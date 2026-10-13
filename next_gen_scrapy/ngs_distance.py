"""
How far the ball carrier went on each play, from NGS's public "ball carrier distance" leader lists
(nextgenstats.nfl.com/api/leaders/distance/ballCarrier), for every game 2018 on.

The list answers per team and week with that team's 50 longest ball-carrier distances (yards the runner or the
receiver covered with the ball and before it, from player tracking). A team-week has ~40-60 runs, catches and returns,
so the list holds ~88% of runs and ~96% of catches; a play missing from a full list went less far than its shortest
entry (its `ngs_dist_below`).

The same API's max-speed list (per team-week, ranked by speed) adds the short fast plays and each play's top speed:
distance / speed predicts how long a run took to ~0.4 s, the duration FTN's clock gives only from 2022, and late.

For matching, the distance is the length of the carry's chart line: on confident matches the line is within ~6% of
it, and among runs that gained the same yards the two agree on which run wandered more (residual correlation ~0.8) -
the information that separates a back's look-alike runs.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import scrape as _S

DIST_API = _S.BASE + "/api/leaders/distance/ballCarrier"
SPEED_API = _S.BASE + "/api/leaders/speed/ballCarrier"   # the same per play, ranked by max speed: adds the short, fast
                                                          # plays the distance list cuts off, and the speed itself
TEAMS_API = _S.BASE + "/api/league/teams"
LIST_LIMIT = 50           # the API's maximum


def _get(url, params):
    return _S.get(url, params=params, headers={"Referer": f"{_S.BASE}/stats/top-plays"}).json()


def teams(season):
    """NGS teamIds for a season (the two Pro Bowl sides left out)."""
    d = _get(TEAMS_API, {"season": int(season)})
    rows = d if isinstance(d, list) else next(v for v in d.values() if isinstance(v, list))
    return [t["teamId"] for t in rows if t.get("abbr") not in ("AFC", "NFC")]


def fetch_list(season, season_type, week, team_id, cache_dir=None, cache=True, api=DIST_API):
    """One team-week's list (the API's own rows). Cached as <season>_<type>_<week>_<team>.json under cache_dir
    (cache=False: always ask, and don't store - a week still being played)."""
    tag = "" if api == DIST_API else "speed_"
    path = os.path.join(cache_dir, f"{tag}{season}_{season_type}_{week}_{team_id}.json") if cache_dir and cache else None
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    data = _get(api, {"limit": LIST_LIMIT, "season": int(season), "seasonType": season_type, "week": int(week),
                           "teamId": team_id}).get("leaders", [])
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
    return data


def carrier_distances(season, weeks, cache_dir=None, workers=8, live_weeks=()):
    """One row per play in the lists: old_game_id, play_id, ngs_carrier_dist, the carrier's ids and the play's
    type - plus per team-week list the shortest distance it holds and whether it was full (list_min, list_full).

    weeks: (season_type, ngs_week) pairs, e.g. [("REG", 1), ..., ("POST", 19)]; live_weeks: those not to cache yet."""
    ids = teams(season)
    jobs = [(st, wk, t, api) for st, wk in weeks for t in ids for api in (DIST_API, SPEED_API)]

    def one(job):
        st, wk, t, api = job
        try:
            return job, fetch_list(season, st, wk, t, cache_dir, cache=(st, wk) not in live_weeks, api=api)
        except Exception:
            return job, None

    rows = []
    with ThreadPoolExecutor(workers) as pool:
        for (st, wk, t, api), lst in pool.map(one, jobs):
            if not lst:
                continue
            dists = [a["leader"].get("inPlayDist") for a in lst if a.get("leader")]
            full = len(lst) >= LIST_LIMIT and api == DIST_API
            lo = float(np.nanmin([d for d in dists if d is not None])) if dists else np.nan
            for a in lst:
                p, ld = a.get("play") or {}, a.get("leader") or {}
                rows.append(dict(old_game_id=str(p.get("gameId")), play_id=float(p.get("playId")),
                                 ngs_carrier_dist=ld.get("inPlayDist"), ngs_max_speed=ld.get("maxSpeed"),
                                 ngs_carrier_gsis=ld.get("gsisId"),
                                 ngs_carrier_esb=ld.get("esbId"), ngs_play_type=p.get("playType"),
                                 team_id=t, season_type=st, ngs_week=wk, list_min=lo, list_full=full))
    cols = ["old_game_id", "play_id", "ngs_carrier_dist", "ngs_max_speed", "ngs_carrier_gsis", "ngs_carrier_esb", "ngs_play_type",
            "team_id", "season_type", "ngs_week", "list_min", "list_full"]
    d = pd.DataFrame(rows, columns=cols)
    # a play in both lists: one row, with the speed list's max speed and the distance list's list facts
    speed = d.dropna(subset=["ngs_max_speed"]).drop_duplicates(["old_game_id", "play_id"]).set_index(["old_game_id", "play_id"])["ngs_max_speed"]
    d = d.sort_values("list_full", ascending=False).drop_duplicates(["old_game_id", "play_id"])
    d["ngs_max_speed"] = [speed.get(k) for k in zip(d["old_game_id"], d["play_id"])]
    return d


def _ngs_weeks(pbp):
    """(season, season_type, NGS week) for a pbp table's games. Playoff weeks are asked for under both nflverse's
    number and the next one (NGS skips the Pro Bowl week before the Super Bowl); an empty answer costs nothing."""
    out = set()
    for season, st, wk in pbp[["season", "season_type", "week"]].drop_duplicates().itertuples(index=False):
        out.add((int(season), st, int(wk)))
        if st == "POST":
            out.add((int(season), st, int(wk) + 1))
    return out


def add_ngs_distance(pbp, cache_dir=None, workers=8):
    """pbp with `ngs_carrier_dist` (yards the ball carrier went, from NGS) and `ngs_dist_below` (a run or catch
    missing from its team-week's full list went less far than this), for every game in it."""
    pbp = pbp.copy()
    want = _ngs_weeks(pbp)
    # the newest two weeks can still change (Monday night, late corrections): never cached
    recent = pbp.sort_values(["season", "game_date"] if "game_date" in pbp else ["season"])[["season", "season_type", "week"]] \
        .drop_duplicates().tail(2)
    live = {(int(a), b, int(c)) for a, b, c in recent.itertuples(index=False)}
    if "game_date" in pbp and pd.to_datetime(pbp["game_date"]).max() < pd.Timestamp.now() - pd.Timedelta(days=10):
        live = set()                                     # a finished season: everything can be cached
    live |= {(a, b, c + 1) for a, b, c in live if b == "POST"}
    frames = [carrier_distances(season, sorted({(st, wk) for s, st, wk in want if s == season}), cache_dir, workers,
                                live_weeks={(st, wk) for s, st, wk in live if s == season})
              for season in sorted({s for s, _, _ in want})]
    d = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["old_game_id", "play_id"])
    key = list(zip(pbp["old_game_id"].astype(str), pbp["play_id"].astype(float)))
    dist = dict(zip(zip(d["old_game_id"], d["play_id"]), d["ngs_carrier_dist"]))
    pbp["ngs_carrier_dist"] = pd.to_numeric(pd.Series([dist.get(k) for k in key], index=pbp.index), errors="coerce")
    speed = dict(zip(zip(d["old_game_id"], d["play_id"]), d["ngs_max_speed"]))
    pbp["ngs_max_speed"] = pd.to_numeric(pd.Series([speed.get(k) for k in key], index=pbp.index), errors="coerce")
    # which pbp team each NGS list belongs to: the posteam of the plays it holds
    pos = dict(zip(key, pbp["posteam"]))
    d["posteam"] = [pos.get(k) for k in zip(d["old_game_id"], d["play_id"])]
    full = d[d["list_full"] & d["posteam"].notna() & d["ngs_play_type"].isin(["play_type_rush", "play_type_pass"])]
    lo = full.groupby(["old_game_id", "posteam"])["list_min"].min().to_dict()
    below = pd.Series([lo.get((g, t)) for g, t in zip(pbp["old_game_id"].astype(str), pbp["posteam"])], index=pbp.index)
    pbp["ngs_dist_below"] = pd.to_numeric(below, errors="coerce").where(pbp["ngs_carrier_dist"].isna())
    return pbp
