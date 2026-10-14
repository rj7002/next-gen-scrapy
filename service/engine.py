"""Live data engine for the app - everything comes from the next_gen_scrapy package, nothing from a database.

A request for one player's season of charts runs the package's own pipeline:

  1. the chart listing straight from Next Gen Stats (scrape.fetch_charts), to find that player's charts
  2. each chart image downloaded into memory and read by the package's chart readers (extract.PROCESSORS)
  3. nflverse play-by-play + FTN charting + NGS participation, plus NGS Game Center's per-receiver air yards and
     per-zone passing yards for every game (load_pbp_with_ftn)
  4. every pass / route / carry matched to its play (match_to_pbp); routes get a route-type read too
  5. passes only: the receivers' route charts from the same games are read as well, and the passes matched
     again with them - each targeted receiver's catch point pins its throw (pass `route_matches=`). This runs
     after the first result is already shown, and the result is swapped in when it's done.

Results live in memory (a small LRU), keyed by (kind, player, season). Nothing is written to disk.
"""

from __future__ import annotations

import io
import math
import os
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import requests

from next_gen_scrapy import load_pbp_with_ftn, match_to_pbp
from next_gen_scrapy import route_type as RT
from next_gen_scrapy import scrape as S
from next_gen_scrapy.extract import PROCESSORS

from . import derived
from .filters import BY_COL, for_table

KINDS = ("pass", "route", "carry")
TABLE = {"pass": "passes", "route": "routes", "carry": "carries"}
FIRST_SEASON = 2018
HEADSHOT = "t_headshot_desktop"
DOWNLOAD_THREADS = 6


def current_season(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


def seasons() -> list[int]:
    return list(range(current_season(), FIRST_SEASON - 1, -1))


# --------------------------------------------------------------------------------------------- listings
_lock = threading.Lock()
_listings: dict[tuple[str, int], tuple[float, list[dict]]] = {}
_abbrs: dict | None = None
_targets: dict[tuple, dict] = {}


def _team_abbrs() -> dict:
    global _abbrs
    if _abbrs is None:
        _abbrs = S.get_team_abbrs()
    return _abbrs


def listing(kind: str, season: int) -> list[dict]:
    """Every chart of one kind in one season, from the NGS chart API (cached; the current season refreshes)."""
    key = (kind, int(season))
    ttl = 600 if season >= current_season() else math.inf
    with _lock:
        hit = _listings.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    abbrs = _team_abbrs()
    charts = [dict(c, team=abbrs.get(c["teamId"], c["teamId"])) for c in S.fetch_charts(kind, season)]
    with _lock:
        _listings[key] = (time.time(), charts)
    return charts


def headshot(url: str | None) -> str | None:
    return url.replace("{formatInstructions}", HEADSHOT) if url else None


def players(season: int) -> list[dict]:
    """Every player with any chart this season, with per-kind games and season totals from the charts' own
    stat lines (passing / receiving / rushing)."""
    out: dict[str, dict] = {}
    for kind in KINDS:
        for c in listing(kind, season):
            p = out.setdefault(c["esbId"], dict(esb_id=c["esbId"], name=c["playerName"], position=c.get("position"),
                                                team=c["team"], headshot=headshot(c.get("headshot")), kinds={}))
            k = p["kinds"].setdefault(kind, dict(games=0))
            k["games"] += 1
            if kind == "pass":
                for f in ("completions", "attempts", "passingYards", "touchdowns", "interceptions"):
                    k[f] = k.get(f, 0) + (c.get(f) or 0)
            elif kind == "route":
                for f in ("receptions", "receivingYards", "touchdowns"):
                    k[f] = k.get(f, 0) + (c.get(f) or 0)
            else:
                for f in ("carries", "rushingYards", "touchdowns"):
                    k[f] = k.get(f, 0) + (c.get(f) or 0)
    volume = lambda p: (p["kinds"].get("pass", {}).get("attempts", 0) + 3 * p["kinds"].get("route", {}).get("receptions", 0)
                        + 1.5 * p["kinds"].get("carry", {}).get("carries", 0))
    return sorted(out.values(), key=volume, reverse=True)


def _route_targets(chart: dict) -> int | None:
    """Targets aren't in a route chart's metadata; the NGS receiving statboard has them, per week."""
    key = (chart["season"], chart["seasonType"], chart["week"])
    with _lock:
        hit = _targets.get(key)
    if hit is None:
        hit = S.fetch_targets(*key)
        with _lock:
            _targets[key] = hit
    return hit.get(chart["esbId"])


# --------------------------------------------------------------------------------------------- season data store
# Pre-read, pre-matched seasons published by the data repo's pipeline (pipeline/update.py in NGS_DATA_REPO, as assets
# of its `data` release; the season in progress is refreshed daily). A chart the store has - same image, with its
# matches - loads from there in a second; anything newer is read live as before. NGS_DATA_TOKEN: a GitHub token that
# can read NGS_DATA_REPO (needed while that repo is private); without it the app reads everything live.
DATA_REPO = os.environ.get("NGS_DATA_REPO", "rj7002/NFL-NGS-Tracking-Data")
DATA_TOKEN = os.environ.get("NGS_DATA_TOKEN")
DATA_CHECK = 10 * 60          # how often to look for a newer release
_assets: tuple[float, dict] = (0.0, {})
data_status = "not checked yet"      # why the season data store is (un)available - shown in job notes and logged
_store: dict[tuple, tuple] = {}       # (kind, season) -> (asset stamp, charts, readings, matched)
_store_lock = threading.Lock()


def _gh_headers(raw=False) -> dict:
    h = {"Accept": "application/octet-stream" if raw else "application/vnd.github+json"}
    if DATA_TOKEN:
        h["Authorization"] = f"Bearer {DATA_TOKEN}"
    return h


def _release_assets() -> dict:
    """{asset name: (api url, updated_at)} of the data release, re-checked every DATA_CHECK seconds."""
    global _assets, data_status
    if time.time() - _assets[0] < DATA_CHECK:
        return _assets[1]
    try:
        r = requests.get(f"https://api.github.com/repos/{DATA_REPO}/releases/tags/data", headers=_gh_headers(), timeout=20)
        found = {a["name"]: (a["url"], a["updated_at"]) for a in r.json().get("assets", [])} if r.ok else {}
        if r.ok:
            status = f"{len(found)} files in {DATA_REPO}'s data release"
        else:
            hint = "no NGS_DATA_TOKEN set" if not DATA_TOKEN else "NGS_DATA_TOKEN can't read it"
            status = f"can't read {DATA_REPO}'s data release (GitHub said {r.status_code}: {hint})"
    except requests.RequestException as e:
        found, status = _assets[1], f"couldn't reach GitHub ({type(e).__name__})"
    if status != data_status:
        print(f"season data: {status}", flush=True)
        data_status = status
    _assets = (time.time(), found)
    return found


def season_store(kind: str, season: int):
    """(charts, readings, matched) for a season from the data release, or None if it isn't there."""
    assets = _release_assets()
    names = [f"{part}_{kind}_{season}.parquet" for part in ("charts", "readings", "matched")]
    if not all(n in assets for n in names):
        return None
    stamp = tuple(assets[n][1] for n in names)
    with _store_lock:
        hit = _store.get((kind, season))
    if hit and hit[0] == stamp:
        return hit[1:]
    try:
        frames = []
        for n in names:
            r = requests.get(assets[n][0], headers=_gh_headers(raw=True), timeout=120)
            r.raise_for_status()
            frames.append(pd.read_parquet(io.BytesIO(r.content)))
    except Exception:
        traceback.print_exc()
        return hit[1:] if hit else None
    with _store_lock:
        _store[(kind, season)] = (stamp, *frames)
    return tuple(frames)


def _stored(kind: str, charts: list[dict], p: pd.DataFrame):
    """(chart rows, matched rows with play-by-play attached, covered chart keys) for the charts the data release
    already holds (same image, matched), or None."""
    st = season_store(kind, int(charts[0]["season"])) if charts else None
    if st is None:
        return None
    ch, rd, mt = st
    image = dict(zip(zip(ch["game_id"].astype(int), ch["esb_id"]), ch["image"]))
    have_m = set(zip(mt["chart_game_id"].astype(int), mt["chart_esb_id"])) if len(mt) else set()
    covered = {(int(c["gameId"]), c["esbId"]) for c in charts
               if image.get((int(c["gameId"]), c["esbId"])) == c.get("imageName") and (int(c["gameId"]), c["esbId"]) in have_m}
    if not covered:
        return None
    rows = rd[pd.Series(list(zip(rd["game_id"].astype(int), rd["esb_id"])), index=rd.index).isin(covered)]
    m = mt[pd.Series(list(zip(mt["chart_game_id"].astype(int), mt["chart_esb_id"])), index=mt.index).isin(covered)].copy()
    for c in ("path_x", "path_y"):
        if c in m:
            m[c] = [np.asarray(v, float) if v is not None else None for v in m[c]]
    from next_gen_scrapy.pbp_match import _attach_pbp_columns
    m = _attach_pbp_columns(m, p) if len(m) else m
    return rows.reset_index(drop=True), m.reset_index(drop=True), covered


# --------------------------------------------------------------------------------------------- pbp cache
_pbp: OrderedDict[int, tuple[float, pd.DataFrame]] = OrderedDict()   # season -> (loaded at, table)
_pbp_locks: dict[int, threading.Lock] = {}
PBP_KEEP = 3
PBP_TTL_CURRENT = 15 * 60     # nflverse adds each game's play-by-play a few hours after it ends: reload the
                              # season in progress this often


NGS_DIST_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "ngsdist")   # finished weeks' lists
PBP_RETRY_WAITS = (2, 5, 10, 20)   # seconds; GitHub now and then resets the connection mid-download (nflverse's
                                   # files are release assets) - a retry nearly always goes through


def _with_retries(fn):
    """fn(), tried again after network errors (nflreadpy raises ConnectionError; requests' errors are OSErrors too).
    Files that did come down stay in nflreadpy's cache, so a retry fetches only what failed."""
    for wait in PBP_RETRY_WAITS:
        try:
            return fn()
        except OSError as e:
            print(f"play-by-play download failed ({e}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    return fn()


_refreshing: set[int] = set()
_unreadable: set[tuple] = set()      # (gameId, esbId, imageName) of charts that failed to read: not retried per load


def pbp(season: int, need_games=(), min_age: float = 120) -> tuple[float, pd.DataFrame]:
    """(loaded-at, play-by-play) for a season, cached - the light version (pbp + FTN + participation + next-snap
    context): enough to show stored matches and every filter. Matching charts live needs more (`for_matching`).
    The season in progress is refreshed in the background once its copy is PBP_TTL_CURRENT old - callers get the
    cached copy meanwhile - and reloaded before returning only when a chart needs a game it doesn't have yet (once
    the copy is min_age old)."""
    with _lock:
        hit = _pbp.get(season)
        if hit:
            _pbp.move_to_end(season)
    if hit:
        age = time.time() - hit[0]
        have = set(hit[1]["old_game_id"].astype(str))
        missing = any(str(g) not in have for g in need_games)
        if season < current_season() or not (missing and age > min_age):
            if season >= current_season() and age >= PBP_TTL_CURRENT:
                _refresh_in_background(season)
            return hit
    return _load_pbp(season)


def _load_pbp(season: int):
    with _lock:
        lk = _pbp_locks.setdefault(season, threading.Lock())
        old = _pbp.get(season)
    with lk:
        with _lock:
            hit = _pbp.get(season)
        if hit is not None and hit is not old:          # someone else loaded it while we waited
            return hit
        if old is not None:
            import nflreadpy
            nflreadpy.clear_cache()                       # its own in-memory cache would hand back the same stale copy
        hit = (time.time(), _with_retries(lambda: load_pbp_with_ftn([season], gamecenter=False, ngs_distance=False)))
        with _lock:
            _pbp[season] = hit
            while len(_pbp) > PBP_KEEP:
                _pbp.popitem(last=False)
        return hit


def _refresh_in_background(season: int):
    with _lock:
        if season in _refreshing:
            return
        _refreshing.add(season)

    def run():
        try:
            _load_pbp(season)
        except Exception:
            traceback.print_exc()
        finally:
            with _lock:
                _refreshing.discard(season)
    threading.Thread(target=run, daemon=True).start()


GC_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "gamecenter")


def for_matching(p: pd.DataFrame, games) -> pd.DataFrame:
    """The pbp rows of these games with what matching needs on top of the light copy: NGS Game Center (one request a
    game) and NGS ball-carrier distances (their team-weeks only) - a handful of requests, not a season's ~1,700."""
    from next_gen_scrapy.pbp_match import add_gamecenter
    from next_gen_scrapy.ngs_distance import add_ngs_distance
    sub = p[p["old_game_id"].astype(str).isin({str(g) for g in games})]
    if not len(sub):
        return sub
    try:
        sub = add_gamecenter(sub, cache_dir=GC_CACHE)
    except Exception:
        traceback.print_exc()
    try:
        sub = add_ngs_distance(sub, cache_dir=NGS_DIST_CACHE)
    except Exception:
        traceback.print_exc()
    return sub


def warm():
    """Load the season in progress (play-by-play and its stored season data) in the background at startup, so the
    first visitor doesn't wait for the downloads."""
    def run():
        try:
            for season in (current_season(), current_season() - 1):     # this season and last: most visits
                pbp(season)
                for kind in KINDS:
                    season_store(kind, season)
        except Exception:
            traceback.print_exc()
    threading.Thread(target=run, daemon=True).start()


# --------------------------------------------------------------------------------------------- reading
def _carry_plays(p: pd.DataFrame, esb_id: str):
    """chart -> this back's pbp runs in that chart's game (None if the game isn't in play-by-play yet)."""
    from next_gen_scrapy.pbp_match import _esb_to_gsis
    gsis = _esb_to_gsis([esb_id]).get(esb_id)
    runs = p[(p["play_type"] == "run") & (p["rusher_player_id"] == gsis)] if gsis else p.iloc[:0]
    by_game = {str(g): d for g, d in runs.groupby(runs["old_game_id"].astype(str))}
    return lambda chart: by_game.get(str(chart["gameId"]))


def _read_charts(kind: str, charts: list[dict], progress, plays_for=None, keep=None) -> tuple[pd.DataFrame, list[str]]:
    """Download and read charts in parallel; progress(i) after each. Returns (rows, notes). plays_for(chart) (carries):
    the back's pbp runs in that game, so a chart whose line colours don't add up to them is read again. keep (dict,
    optional): filled with {(game_id, esb_id): (image, metadata)} for re-reading later."""
    process = PROCESSORS[kind]
    notes, rows, done = [], [], [0]

    def one(chart):
        meta = dict(chart)
        if kind == "route" and meta.get("targets") is None:
            meta["targets"] = _route_targets(meta)
        img = S.fetch_chart_image(meta)
        if keep is not None:
            keep[(meta["gameId"], meta["esbId"])] = (img, meta)
        if kind == "carry" and plays_for is not None:
            r, _qc = process(img, meta, plays=plays_for(chart))
        else:
            r, _qc = process(img, meta)
        return r

    with ThreadPoolExecutor(DOWNLOAD_THREADS) as ex:
        futs = {ex.submit(one, c): c for c in charts}
        for f in futs:
            c = futs[f]
            try:
                rows += f.result()
            except Exception as e:                      # a blank / unreadable chart: skip it, say so
                notes.append(f"week {S.week_label(c)}: chart couldn't be read ({type(e).__name__})")
                _unreadable.add((c["gameId"], c["esbId"], c.get("imageName")))
            done[0] += 1
            progress(done[0])
    return pd.DataFrame(rows), notes


# --------------------------------------------------------------------------------------------- JSON
CORE = ["game_id", "play_id", "week", "season_type", "posteam", "defteam", "home_team", "away_team", "qtr", "down",
        "ydstogo", "yardline_100", "time", "desc", "epa", "wpa", "success", "first_down", "yards_gained", "bucket",
        "location_prob", "match_prob", "location_confident", "n_candidates", "swap_yds", "play_type"]
EXTRA = {
    "pass": ["x_coord", "y_coord", "pass_type", "receiver_player_name", "air_yards", "yards_after_catch",
             "complete_pass", "interception", "pass_touchdown", "cpoe", "xpass"],
    "route": ["route_type", "passer_player_name", "air_yards", "yards_after_catch", "complete_pass", "pass_touchdown",
              "start_ok", "gap", "chart_touchdown", "path_x", "path_y", "path_segment"],
    "carry": ["gain_class", "chart_touchdown", "chart_fumble_lost", "handoff_ok", "gap", "rush_touchdown",
              "path_x", "path_y"],
}


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (np.ndarray, list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if not math.isfinite(f) else round(f, 3)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _week(week, season_type) -> str | None:
    """Same labels as the game list (scrape.week_label): "7", or "post-20" for the playoffs (NGS week numbers)."""
    w = _clean(week)
    if w is None:
        return None
    return str(int(w)) if season_type in (None, "REG") else f"post-{int(w)}"


def _path(v, digits=2):
    a = np.asarray(v, float) if v is not None else np.zeros(0)
    return [round(float(x), digits) for x in a]


_route_guesses: dict[int, tuple] = {}     # season -> (store stamp, {(game_id, play_id): top route guess})


def _routes_by_play(season: int) -> dict:
    """{(game_id, play_id): route type read from the targeted receiver's route chart} for every route chart the
    season data holds - what a QB's throws were thrown to."""
    st = season_store("route", season)
    if st is None:
        return {}
    with _store_lock:
        stamp = _store.get(("route", season), (None,))[0]
    hit = _route_guesses.get(season)
    if hit and hit[0] == stamp:
        return hit[1]
    m = st[2]
    out = {}
    if len(m):
        m = m.assign(path_x=[np.asarray(v, float) if v is not None else None for v in m["path_x"]],
                     path_y=[np.asarray(v, float) if v is not None else None for v in m["path_y"]])
        P = RT.predict_proba(m)
        for g, pid, row in zip(m["game_id"], m["play_id"], P):
            if np.isfinite(row).all():
                out[(str(g), int(float(pid)))] = RT.CLASSES[int(np.argmax(row))]
    _route_guesses[season] = (stamp, out)
    return out


def _route_tree(kind: str, matched: pd.DataFrame, guesses, season: int | None) -> list:
    """(route, source) per matched pass / route for the route tree: NGS's own label for the targeted receiver's route
    (participation `target_route`, unified to route_type's classes) where play-by-play has it, else the route read
    from the chart line's shape - the route's own line, or for a pass the targeted receiver's route chart."""
    by_play = _routes_by_play(season) if kind == "pass" and season else {}
    out = []
    for i, (_, r) in enumerate(matched.iterrows()):
        key = (str(r.get("game_id")), int(float(r["play_id"]))) if pd.notna(r.get("play_id")) else None
        ngs = r.get("target_route")
        ngs = RT.UNIFY.get(ngs, ngs) if isinstance(ngs, str) else None
        if ngs in RT.CLASSES:
            out.append((ngs, "ngs"))
        elif kind == "route" and guesses is not None and guesses[i]:
            out.append((guesses[i][0][0], "shape"))
        elif kind == "pass" and key in by_play:
            out.append((by_play[key], "shape"))
        else:
            out.append((None, None))
    return out


def _receiver_positions(matched: pd.DataFrame) -> list[str | None]:
    """Each pass's target's position (WR / TE / RB / FB), for "throws to tight ends" on a quarterback's chart."""
    if not len(matched) or "receiver_player_id" not in matched:
        return [None] * len(matched)
    try:
        pos = _people_table()["position"]
    except Exception:   # the players table is a download; without it the filter just has nothing to offer
        return [None] * len(matched)
    out = matched["receiver_player_id"].map(pos).replace({"HB": "RB"})
    return [v if isinstance(v, str) and v in ("WR", "TE", "RB", "FB") else None for v in out]


def _panel_cols(kind: str) -> list[str]:
    """The filter panel's columns: in every play row the app loads (the Ask-bar-only ones come via play_columns)."""
    return [f["col"] for f in for_table(TABLE[kind]) if f.get("panel", True)]


def play_columns(job: "Job", cols: list[str]) -> dict:
    """{col: {play id: value}} for the job's plays - any filter column, stored or derived - for the Ask bar's filters
    on columns the play rows leave out."""
    allowed = {f["col"] for f in for_table(TABLE[job.kind])}
    out = {}
    for c in cols:
        if c not in allowed or c not in BY_COL:
            continue
        vals = {}
        for frame, est in ((job.frame, False), (job.undrawn, True)):
            s = derived.column(frame, c)
            if s is None:
                continue
            ids = [f"{g}:{float(p) if est else p}" for g, p in zip(frame["game_id"], frame["play_id"])]
            for i, v in zip(ids, s):
                v = _clean(v)
                if v is not None and v != "":
                    vals[i] = v
        out[c] = vals
    return out


def _plays_json(kind: str, matched: pd.DataFrame, chart_df: pd.DataFrame, season: int | None = None) -> list[dict]:
    filt_cols = _panel_cols(kind)
    matched = derived.with_columns(matched, filt_cols) if len(matched) else matched
    cols = list(dict.fromkeys(CORE + EXTRA[kind] + filt_cols))
    out = []
    pred = None
    no_fit = set()
    if kind == "carry" and len(matched) and "implausible" in matched:
        # the matcher had to put these lines somewhere, but no play fits them (a misread line, usually):
        # show them as lines without a play rather than with a wrong play's details
        bad = matched["implausible"].fillna(False).astype(bool)
        no_fit = set(zip(matched.loc[bad, "chart_game_id"], matched.loc[bad, "chart_carry_id"]))
        matched = matched[~bad]
    if kind == "route" and len(matched):
        P = RT.predict_proba(matched)
        pred = [None if not np.isfinite(r).all() else sorted(zip(RT.CLASSES, r.round(3).tolist()), key=lambda t: -t[1])[:4]
                for r in P]
    tree = _route_tree(kind, matched, pred, season) if kind in ("pass", "route") and len(matched) else None
    rpos = _receiver_positions(matched) if kind == "pass" else None
    for i, (_, r) in enumerate(matched.iterrows()):
        d = {"id": f"{r.get('game_id')}:{r.get('play_id')}", "matched": True}
        for c in cols:
            if c not in matched.columns:
                continue
            v = r[c]
            if c in ("path_x", "path_y"):
                d[c] = _path(v)
            elif c == "path_segment":
                d[c] = [str(s) for s in v] if v is not None else []
            else:
                d[c] = _clean(v)
        d["week"] = _week(r.get("week"), r.get("season_type"))
        if pred is not None:
            d["route_guess"] = pred[i]
        if tree is not None and tree[i][0]:
            d["tree_route"], d["tree_route_src"] = tree[i]
        if rpos is not None and rpos[i]:
            d["receiver_position"] = rpos[i]
        out.append(d)
    # chart lines / dots the matcher couldn't place - still drawn, flagged unmatched
    out += _unmatched(kind, matched, chart_df, no_fit)
    return out


UNDRAWN_DEPTH_OFFSET = 0.56   # an incompletion's dot sits this much deeper than pbp air yards (median, 2018-25)


def _undrawn(job, charts, pb: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """
    The QB's pass attempts in his charted games that NGS didn't draw - ~1 in 5 incompletions (every pass with no
    intended receiver, and some others), spikes, and the odd completion - as pbp rows with an estimated spot: pbp
    air yards for depth, the pass direction's usual spot across the field (shifted for the hash it was snapped from) -
    the matcher's own calibration. Depth is usually within a yard; across the field it's only known to the side.
    With them, attempts and completion % match the box score.
    """
    from next_gen_scrapy import pbp_match as PM
    gsis = PM._esb_to_gsis([job.esb_id]).iloc[0]
    if not isinstance(gsis, str) or not len(pb):
        return pd.DataFrame()
    week_of = {str(c["gameId"]): S.week_label(c) for c in charts}
    two_pt = pb["two_point_attempt"].fillna(0) if "two_point_attempt" in pb else 0
    sack = pb["sack"].fillna(0) if "sack" in pb else 0
    cand = pb[pb["old_game_id"].astype(str).isin(week_of) & (pb["passer_player_id"] == gsis)
              & pb["play_type"].isin(["pass", "qb_spike"]) & (pb["pass_attempt"] == 1) & (sack != 1) & (two_pt != 1)]
    have = set(zip(matched["game_id"], matched["play_id"])) if len(matched) else set()
    und = cand[[(g, q) not in have for g, q in zip(cand["game_id"], cand["play_id"])]].copy()
    if not len(und):
        return und
    params = PM._league_params("pass", pd.DataFrame({"season": [job.season]})) or PM._DEFAULTS["pass"]
    side = {k: v[0] for k, v in params["side"].items()}
    shift = params.get("hash_shift", PM._HASH_SHIFT)
    hx = und["starting_hash"].map({"L": -1.0, "R": 1.0, "M": 0.0}).fillna(0.0) if "starting_hash" in und else 0.0
    spike = und["play_type"] == "qb_spike"
    und["x_coord"] = (und["pass_location"].map(side) + shift * hx).where(~spike, 0.0).fillna(0.0)
    und["y_coord"] = (und["air_yards"] + UNDRAWN_DEPTH_OFFSET).where(~spike, -1.0).fillna(-1.0)
    flag = lambda c: und[c].fillna(0) == 1 if c in und else pd.Series(False, index=und.index)
    und["pass_type"] = np.select([flag("pass_touchdown"), flag("interception"), flag("complete_pass")],
                                 ["TOUCHDOWN", "INTERCEPTION", "COMPLETE"], "INCOMPLETE")
    und["chart_week"] = und["old_game_id"].astype(str).map(week_of)
    return und


def _undrawn_json(kind: str, und: pd.DataFrame) -> list[dict]:
    if und is None or not len(und):
        return []
    filt_cols = _panel_cols(kind)
    und = derived.with_columns(und, filt_cols)
    cols = list(dict.fromkeys(CORE + EXTRA[kind] + filt_cols))
    out = []
    for _, r in und.iterrows():
        d = {"id": f"{r['game_id']}:{float(r['play_id'])}", "matched": True, "estimated": True}
        for c in cols:
            if c in und.columns and c not in ("path_x", "path_y", "path_segment"):
                d[c] = _clean(r[c])
        d["week"] = r["chart_week"]
        d["location_prob"] = None
        ngs = r.get("target_route")
        ngs = RT.UNIFY.get(ngs, ngs) if isinstance(ngs, str) else None
        if ngs in RT.CLASSES:
            d["tree_route"], d["tree_route_src"] = ngs, "ngs"
        out.append(d)
    return out


def _unmatched(kind, matched, chart_df, no_fit=()):
    if chart_df is None or not len(chart_df):
        return []
    rows = []
    if kind == "pass":
        have = set(zip(matched.get("chart_game_id", []), matched.get("x_coord", []).round(2), matched.get("y_coord", []).round(2))) \
            if len(matched) else set()
        for _, c in chart_df.iterrows():
            if pd.isna(c["x_coord"]) or (c["game_id"], round(c["x_coord"], 2), round(c["y_coord"], 2)) in have:
                continue
            rows.append({"id": f"u:{c['game_id']}:{c['x_coord']}:{c['y_coord']}", "matched": False, "week": _week(c["week"], c.get("season_type")),
                         "x_coord": _clean(c["x_coord"]), "y_coord": _clean(c["y_coord"]), "pass_type": c["pass_type"]})
        return rows
    idc = "route_id" if kind == "route" else "carry_id"
    have = set(zip(matched.get("chart_game_id", []), matched.get(f"chart_{idc}", []))) if len(matched) else set()
    for (g, i), grp in chart_df.groupby(["game_id", idc], sort=False):
        if (g, i) in have:
            continue
        grp = grp.sort_values("point")
        d = {"id": f"u:{g}:{i}", "matched": False,
             "week": _week(grp["week"].iloc[0], grp["season_type"].iloc[0] if "season_type" in grp else None),
             "path_x": _path(grp["x_coord"]), "path_y": _path(grp["y_coord"])}
        if kind == "route":
            d.update(route_type=grp["route_type"].iloc[0], path_segment=[str(s) for s in grp["segment"]],
                     chart_touchdown=bool(grp["touchdown"].iloc[0]))
        else:
            d.update(gain_class=grp["gain_class"].iloc[0], chart_touchdown=bool(grp["touchdown"].iloc[0]))
            if (g, i) in no_fit:
                d["reason"] = "no_fit"
        rows.append(d)
    if kind == "route" and rows:
        # no play, but the line still has a shape: its route goes in the route tree
        season = int(chart_df["season"].iloc[0]) if "season" in chart_df else 2023
        P = RT.predict_proba(pd.DataFrame(dict(path_x=[np.asarray(d["path_x"]) for d in rows],
                                               path_y=[np.asarray(d["path_y"]) for d in rows],
                                               path_segment=[d["path_segment"] for d in rows],
                                               route_type=[d["route_type"] for d in rows], season=season)))
        for d, r in zip(rows, P):
            if np.isfinite(r).all():
                d["tree_route"], d["tree_route_src"] = RT.CLASSES[int(np.argmax(r))], "shape"
    return rows


def _games(charts: list[dict], matched: pd.DataFrame) -> list[dict]:
    opp = {}
    if len(matched) and "chart_game_id" in matched:
        for g, grp in matched.groupby("chart_game_id"):
            d = grp["defteam"].dropna()
            opp[str(g)] = d.iloc[0] if len(d) else None
    out = []
    for c in charts:
        g = dict(week=S.week_label(c), game_id=c["gameId"], opponent=opp.get(str(c["gameId"])), season_type=c["seasonType"])
        for f in ("completions", "attempts", "passingYards", "interceptions", "receptions", "receivingYards",
                  "carries", "rushingYards", "touchdowns"):
            if c.get(f) is not None:
                g[f] = c[f]
        out.append(g)
    return out


# --------------------------------------------------------------------------------------------- tracking
_bdb: dict[int, pd.DataFrame] = {}
BDB_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


def _bdb_tracking(season: int):
    """Every tracked player at the key moments of each play, from a local Big Data Bowl 2027 copy (2023-2025), if
    there is one."""
    from next_gen_scrapy import bdb
    if season not in _bdb:
        try:
            _bdb[season] = bdb.load_play_tracking(season, cache_dir=BDB_CACHE)
        except Exception:
            traceback.print_exc()
            _bdb[season] = pd.DataFrame()
    return _bdb[season] if len(_bdb[season]) else None


# --------------------------------------------------------------------------------------------- jobs
@dataclass
class Job:
    id: str
    kind: str
    esb_id: str
    name: str
    season: int
    status: str = "queued"          # queued -> running -> ready (shown) -> refining -> done | error
    stage: str = "Queued"
    done: int = 0
    total: int = 0
    version: int = 0
    data: dict | None = None
    frame: pd.DataFrame | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    pending: list = field(default_factory=list)   # games charted but not in nflverse's play-by-play yet
    undrawn: pd.DataFrame | None = None           # passes only: incompletions NGS didn't draw (estimated spots)
    all_charts: list = field(default_factory=list)  # every chart of the job (some may come from the season store)
    stored: tuple | None = None                   # (chart rows, matched rows) from the season data store
    pbp_ref: pd.DataFrame | None = None

    def public(self):
        return dict(id=self.id, kind=self.kind, name=self.name, season=self.season, status=self.status,
                    stage=self.stage, done=self.done, total=self.total, version=self.version, error=self.error)


_jobs: OrderedDict[str, Job] = OrderedDict()
_by_key: dict[tuple, str] = {}
_pool = ThreadPoolExecutor(2)
KEEP_JOBS = 24


def start(kind: str, esb_id: str, season: int) -> Job:
    key = (kind, esb_id, int(season))
    with _lock:
        jid = _by_key.get(key)
        j = _jobs.get(jid) if jid else None
        # the season in progress keeps changing (new charts, newly published play-by-play): redo an old result
        stale = j is not None and j.status == "done" and int(season) >= current_season() and (
            j.pending or time.time() - j.created > PBP_TTL_CURRENT)
        if j is not None and j.status != "error" and not stale:
            _jobs.move_to_end(jid)
            return j
    charts = [c for c in listing(kind, season) if c["esbId"] == esb_id]
    if not charts:
        raise LookupError(f"no {kind} charts for that player in {season}")
    job = Job(id=uuid.uuid4().hex[:12], kind=kind, esb_id=esb_id, name=charts[0]["playerName"], season=int(season))
    with _lock:
        _jobs[job.id] = job
        _by_key[key] = job.id
        while len(_jobs) > KEEP_JOBS:
            old, j = _jobs.popitem(last=False)
            _by_key.pop((j.kind, j.esb_id, j.season), None)
    _pool.submit(_run, job, charts)
    return job


def get(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def _run(job: Job, charts: list[dict]):
    try:
        from next_gen_scrapy.player import _week_sort_key
        charts = sorted(charts, key=lambda c: _week_sort_key(S.week_label(c)))
        job.status, job.all_charts = "running", charts
        job.stage, job.done, job.total = "Loading play-by-play", 0, 0
        _, p = pbp(job.season, need_games=[c["gameId"] for c in charts])
        job.pbp_ref = p
        # charts the season data store already read and matched load from there; only the rest are read live
        job.stage = "Loading season data"
        stored = _stored(job.kind, charts, p)
        covered = stored[2] if stored else set()
        job.stored = stored[:2] if stored else None
        live = [c for c in charts if (int(c["gameId"]), c["esbId"]) not in covered
                and (c["gameId"], c["esbId"], c.get("imageName")) not in _unreadable]
        notes = [f"{len({k[0] for k in covered})} games loaded from the season data (updated daily)"] if covered else \
            ([] if data_status.endswith("data release") else [f"Season data unavailable - {data_status}; charts read live"])
        if not live:
            _publish(job, charts, pd.DataFrame(), pd.DataFrame(), notes)
            job.status, job.stage = "done", "Done"
            return
        p = for_matching(p, {c["gameId"] for c in live})   # live charts only: matching's extra data for their games
        plays_for = _carry_plays(p, job.esb_id) if job.kind == "carry" else None   # carries are read with plays in hand
        job.stage, job.total, job.done = "Reading charts", len(live), 0
        images = {} if job.kind == "route" else None
        chart_df, read_notes = _read_charts(job.kind, live, lambda i: setattr(job, "done", i), plays_for, keep=images)
        notes += read_notes
        have = set(p["old_game_id"].astype(str))
        job.pending = [dict(week=S.week_label(c), game_id=c["gameId"],
                            plays=int((chart_df["game_id"].astype(str) == str(c["gameId"])).sum()) if len(chart_df) else 0)
                       for c in live if str(c["gameId"]) not in have]
        job.stage = "Matching to play-by-play"
        trk = _bdb_tracking(job.season) if job.kind == "carry" else None
        matched = match_to_pbp(chart_df, p, job.kind, tracking=trk) if len(chart_df) else pd.DataFrame()
        _publish(job, charts, chart_df, matched, notes)
        if job.kind == "pass" and len(matched):
            job.status, job.stage = "refining", "Pinning passes with receivers' route charts"
            routes, route_images = _receiver_routes(job, live)
            if routes is not None and len(routes):
                from next_gen_scrapy.retrace import retrace_routes
                job.stage, job.done, job.total = "Re-reading tangled route charts", 0, 0
                routes, _rep = retrace_routes(routes, p, images=route_images, pass_matches=matched)
                job.stage = "Re-matching passes"
                rm = match_to_pbp(routes, p, "route")
                matched2 = match_to_pbp(chart_df, p, "pass", route_matches=rm)
                notes = notes + [f"{rm['chart_esb_id'].nunique()} receivers' route charts used to pin passes"]
                _publish(job, charts, chart_df, matched2, notes)
        if job.kind == "route" and len(matched):
            _refine_routes(job, live, chart_df, p, images, notes)
        job.status, job.stage = "done", "Done"
    except Exception as e:
        job.status, job.error, job.stage = "error", f"{type(e).__name__}: {e}", "Failed"
        traceback.print_exc()


def _refine_routes(job: Job, charts: list[dict], chart_df: pd.DataFrame, p: pd.DataFrame, images: dict, notes: list):
    """Routes, second pass: re-read tangled charts (retrace_routes: the untangling that fits the receiver's real
    targets best), and pin each completion with the QB's pass chart dot for the same throw."""
    from next_gen_scrapy.retrace import retrace_routes
    job.status, job.stage = "refining", "Reading the QB's pass charts"
    qb = _qb_passes(job, charts)
    pm = match_to_pbp(qb, p, "pass") if qb is not None and len(qb) else None
    notes = list(notes)
    if pm is not None and len(pm):
        notes.append(f"{len(pm[['chart_game_id', 'chart_esb_id']].drop_duplicates())} QB pass charts used to pin catches")
    job.stage, job.done, job.total = "Re-reading tangled routes", 0, 0
    fixed, rep = retrace_routes(chart_df, p, images=images, pass_matches=pm)
    if rep.get("reread"):
        notes.append(f"{rep['reread']} tangled route charts re-read")
    job.stage = "Re-matching routes"
    matched = match_to_pbp(fixed, p, "route", pass_matches=pm)
    _publish(job, charts, fixed, matched, notes)


def _qb_passes(job: Job, route_charts: list[dict]) -> pd.DataFrame | None:
    games = {c["gameId"] for c in route_charts}
    teams = {c["team"] for c in route_charts}
    qc = [c for c in listing("pass", job.season) if c["gameId"] in games and c["team"] in teams]
    if not qc:
        return None
    job.total, job.done = len(qc), 0
    df, _ = _read_charts("pass", qc, lambda i: setattr(job, "done", i))
    return df


def _receiver_routes(job: Job, qb_charts: list[dict]) -> tuple[pd.DataFrame | None, dict]:
    games = {c["gameId"] for c in qb_charts}
    team = qb_charts[0]["team"]
    rc = [c for c in listing("route", job.season) if c["gameId"] in games and c["team"] == team]
    if not rc:
        return None, {}
    job.total, job.done = len(rc), 0
    images = {}
    df, _ = _read_charts("route", rc, lambda i: setattr(job, "done", i), keep=images)
    return df, images


def _publish(job: Job, charts, chart_df, matched, notes):
    """Show the job's current result: what was read live (chart_df, matched) plus whatever came from the season
    data store, for every chart of the job."""
    charts = job.all_charts or charts
    if job.stored is not None:
        s_rows, s_matched = job.stored
        chart_df = pd.concat([s_rows, chart_df], ignore_index=True) if len(chart_df) else s_rows
        matched = pd.concat([s_matched, matched], ignore_index=True) if len(matched) else s_matched
    if job.kind == "pass" and job.pbp_ref is not None:
        job.undrawn = _undrawn(job, charts, job.pbp_ref, matched)
    plays = _plays_json(job.kind, matched, chart_df, job.season) + _undrawn_json(job.kind, job.undrawn)
    first = charts[0]
    job.frame = matched
    job.data = dict(
        kind=job.kind, table=TABLE[job.kind], season=job.season,
        player=dict(esb_id=job.esb_id, name=job.name, position=first.get("position"), team=first["team"],
                    headshot=headshot(first.get("headshot"))),
        games=_games(charts, matched), plays=plays, notes=notes, pending=job.pending,
        counts=dict(charts=len(charts), matched=int(len(matched)), unmatched=sum(not p["matched"] for p in plays),
                    undrawn=0 if job.undrawn is None else int(len(job.undrawn))),
        version=job.version + 1)
    job.version += 1
    if job.status == "running":
        job.status = "ready"


# --------------------------------------------------------------------------------------------- one play, in full
_people: pd.DataFrame | None = None

# who did what on the play: (pbp id column, role), in the order they're listed
ROLES = [("passer_player_id", "Passer"), ("receiver_player_id", "Target"), ("rusher_player_id", "Ball carrier"),
         ("interception_player_id", "Interception"), ("sack_player_id", "Sack"), ("half_sack_1_player_id", "Half sack"),
         ("half_sack_2_player_id", "Half sack"), ("qb_hit_1_player_id", "QB hit"), ("qb_hit_2_player_id", "QB hit"),
         ("pass_defense_1_player_id", "Pass defended"), ("pass_defense_2_player_id", "Pass defended"),
         ("tackle_for_loss_1_player_id", "Tackle for loss"), ("solo_tackle_1_player_id", "Tackle"),
         ("solo_tackle_2_player_id", "Tackle"), ("tackle_with_assist_1_player_id", "Tackle"),
         ("tackle_with_assist_2_player_id", "Tackle"), ("assist_tackle_1_player_id", "Assist"),
         ("assist_tackle_2_player_id", "Assist"), ("assist_tackle_3_player_id", "Assist"),
         ("forced_fumble_player_1_player_id", "Forced fumble"), ("fumble_recovery_1_player_id", "Fumble recovery"),
         ("penalty_player_id", "Penalty")]


def _people_table() -> pd.DataFrame:
    global _people
    if _people is None:
        import nflreadpy
        pl = nflreadpy.load_players().to_pandas().dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")
        _people = pl.set_index("gsis_id")[["display_name", "position", "headshot", "jersey_number", "latest_team"]]
    return _people


def _pick(row, cols):
    return {c: _clean(row[c]) for c in cols if c in row.index and _clean(row[c]) is not None}


def play_detail(job: Job, play_id: str) -> dict | None:
    """Everything known about one play of a job: the game and score, situation, result and its expectations,
    every player involved, scheme and charting, NGS Game Center numbers, the chart's own drawing, the match, and
    the whole drive around it."""
    play = next((p for p in (job.data or {}).get("plays", []) if p["id"] == play_id), None)
    if play is None:
        return None
    out = dict(id=play_id, kind=job.kind, matched=play["matched"], week=play.get("week"),
               chart={k: play.get(k) for k in ("x_coord", "y_coord", "path_x", "path_y", "path_segment", "pass_type",
                                                "route_type", "gain_class", "chart_touchdown", "route_guess", "gap")})
    if not play["matched"] or job.frame is None:
        return out
    g, pid = play_id.split(":")
    fr = job.frame
    hit = fr[(fr["game_id"].astype(str) == g) & (fr["play_id"].astype(float) == float(pid))]
    if not len(hit) and job.undrawn is not None and len(job.undrawn):
        u = job.undrawn
        hit = u[(u["game_id"].astype(str) == g) & (u["play_id"].astype(float) == float(pid))]
        out["estimated"] = bool(len(hit))
    if not len(hit):
        return out
    r = hit.iloc[0]
    out["game"] = _pick(r, ["game_date", "home_team", "away_team", "stadium", "roof", "surface", "temp", "wind",
                            "weather", "home_coach", "away_coach", "spread_line", "total_line", "season_type", "week"])
    out["situation"] = _pick(r, ["posteam", "defteam", "qtr", "time", "down", "ydstogo", "yrdln", "yardline_100",
                                 "goal_to_go", "posteam_score", "defteam_score", "score_differential",
                                 "half_seconds_remaining", "game_seconds_remaining", "shotgun", "no_huddle",
                                 "posteam_timeouts_remaining", "defteam_timeouts_remaining"])
    out["result"] = _pick(r, ["desc", "yards_gained", "epa", "wpa", "wp", "vegas_wp", "success", "first_down",
                              "touchdown", "interception", "fumble_lost", "sack", "qb_hit", "qb_scramble", "penalty",
                              "penalty_type", "penalty_yards", "air_yards", "yards_after_catch", "pass_length",
                              "pass_location", "run_location", "run_gap", "cp", "cpoe", "xpass", "pass_oe",
                              "xyac_mean_yardage", "xyac_success", "xyac_fd", "air_epa", "yac_epa", "comp_air_epa",
                              "comp_yac_epa", "posteam_score_post", "defteam_score_post"])
    out["scheme"] = _pick(r, ["offense_formation", "offense_personnel", "defense_personnel", "n_defense_box",
                              "n_pass_rushers", "n_blitzers", "n_offense_backfield", "defense_coverage_type",
                              "defense_man_zone_type", "time_to_throw", "was_pressure", "target_route", "qb_location",
                              "starting_hash", "is_play_action", "is_rpo", "is_screen_pass", "is_motion", "is_no_huddle",
                              "is_qb_out_of_pocket", "is_trick_play"])
    out["charting"] = _pick(r, ["read_thrown", "is_catchable_ball", "is_contested_ball", "is_drop",
                                "is_created_reception", "is_interception_worthy", "is_throw_away"])
    if job.kind == "pass" and pd.isna(r.get("ngs_receiver_targets", np.nan)):
        try:                                  # stored plays come with the light pbp: this game's Game Center, once
            from next_gen_scrapy.pbp_match import add_gamecenter
            g = add_gamecenter(pd.DataFrame([r]), cache_dir=GC_CACHE).iloc[0]
            r = r.copy()
            r["ngs_receiver_targets"], r["ngs_receiver_avg_air_yards"] = g.get("ngs_receiver_targets"), g.get("ngs_receiver_avg_air_yards")
        except Exception:
            pass
    out["gamecenter"] = _pick(r, ["ngs_receiver_targets", "ngs_receiver_avg_air_yards"])
    out["match"] = _pick(r, ["location_prob", "match_prob", "n_candidates", "swap_yds", "bucket"])
    # the people, with headshots
    try:
        ppl = _people_table()
    except Exception:
        ppl = None
    seen, people = set(), []
    for col, role in ROLES:
        gid = r.get(col)
        if not isinstance(gid, str) or not gid or (gid, role) in seen:
            continue
        seen.add((gid, role))
        name_col = col.replace("_player_id", "_player_name")
        p = dict(id=gid, role=role, name=_clean(r.get(name_col)))
        if ppl is not None and gid in ppl.index:
            q = ppl.loc[gid]
            p.update(name=q.display_name, position=_clean(q.position), headshot=_clean(q.headshot),
                     jersey=_clean(q.jersey_number))
        p["team"] = _clean(r.get("penalty_team") if role == "Penalty" else r.get("posteam") if role in ("Passer", "Target", "Ball carrier")
                           else r.get("defteam"))
        people.append(p)
    out["people"] = people
    # the drive this play belongs to
    try:
        _, pb = pbp(job.season)
        d = pb[(pb["game_id"] == r["game_id"]) & (pb["fixed_drive"] == r.get("fixed_drive")) & pb["play_type"].notna()]
        d = d.sort_values("play_id")
        out["drive"] = dict(
            number=_clean(r.get("fixed_drive")), result=_clean(r.get("fixed_drive_result")),
            plays_count=_clean(r.get("drive_play_count")), time=_clean(r.get("drive_time_of_possession")),
            first_downs=_clean(r.get("drive_first_downs")), start=_clean(r.get("drive_start_yard_line")),
            end=_clean(r.get("drive_end_yard_line")),
            plays=[dict(play_id=_clean(x.play_id), qtr=_clean(x.qtr), time=_clean(x.time), down=_clean(x.down),
                        ydstogo=_clean(x.ydstogo), yrdln=_clean(x.yrdln), play_type=_clean(x.play_type),
                        yards=_clean(x.yards_gained), epa=_clean(x.epa), desc=_clean(x.desc),
                        current=float(x.play_id) == float(pid)) for x in d.itertuples()])
    except Exception:
        traceback.print_exc()
    return out
