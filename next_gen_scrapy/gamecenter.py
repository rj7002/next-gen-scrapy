"""
NGS Game Center data (the per-game overview behind nextgenstats.nfl.com/stats/game-center/<gameId>).

What matters for matching, published for every game 2018 on:
- for each team's top receivers, their targets and average air yards - NGS's own measurement, the one
  participation's `ngs_air_yards` carried per play through 2022 (per play it's gone from public data since 2023).
  A receiver's targets must add up to his total.
- for each team's main passer, completions and yards gained per zone (behind the line / 0-9 / 10-19 / 20+ air
  yards x left / middle / right third of the field). The completions drawn in a zone must add up to its yards.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import scrape as _S

OVERVIEW_API = _S.BASE + "/api/gamecenter/overview"


def fetch_overview(game_id, cache_dir=None):
    """The Game Center overview of one game (NGS gameId = nflverse old_game_id), as a dict. With cache_dir, each
    game is stored there as <gameId>.json and read back next time."""
    game_id = int(game_id)
    path = os.path.join(cache_dir, f"{game_id}.json") if cache_dir else None
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # the API answers only with a Game Center page as the Referer
    data = _S.get(OVERVIEW_API, params={"gameId": game_id},
                  headers={"Referer": f"{_S.BASE}/stats/game-center/{game_id}"}).json()
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
    return data


ZONE_DEPTH = {"behindLOS": 0, "losTo9": 1, "10To19": 2, "20Plus": 3}   # NGS air yards < 0, 0-9, 10-19, 20+
ZONE_SECTION = {"leftThird": 0, "middleThird": 1, "rightThird": 2}     # thirds of the field's width


def _overviews(game_ids, cache_dir, workers):
    ids = sorted({int(g) for g in game_ids if pd.notna(g)})

    def one(gid):
        try:
            return gid, fetch_overview(gid, cache_dir)
        except Exception:
            return gid, None

    with ThreadPoolExecutor(workers) as pool:
        yield from pool.map(one, ids)


def game_tables(game_ids, cache_dir=None, workers=8):
    """
    (receivers, zones) for every game in `game_ids` (NGS / nflverse old_game_id); games that fail to load are skipped.

    receivers: one row per listed receiver per game (each team's top five by targets) - old_game_id, esb_id,
    targets, receptions, avg_air_yards (NGS's measured air yards, averaged over his targets).

    zones: one row per passer, game and zone - old_game_id, esb_id, depth (0 behind the line, 1 0-9 air yards,
    2 10-19, 3 20+), section (0 left, 1 middle, 2 right third of the field), attempts, completions, yards (yards
    gained on those completions), touchdowns, interceptions. Each team's main passer is listed.
    """
    rec, zon = [], []
    for gid, d in _overviews(game_ids, cache_dir, workers):
        for side in ("home", "visitor"):
            for r in ((d or {}).get("receivers") or {}).get(side) or []:
                info = r.get("receptionInfo") or {}
                if info.get("avgAirYards") is None or not r.get("targets"):
                    continue
                rec.append(dict(old_game_id=str(gid), esb_id=r["esbId"], targets=int(r["targets"]),
                                receptions=int(r.get("receptions") or 0), avg_air_yards=float(info["avgAirYards"])))
            p = ((d or {}).get("passers") or {}).get(side)
            for z in (p or {}).get("zones") or []:
                if z.get("lineOfScrimmageDistance") not in ZONE_DEPTH or z.get("section") not in ZONE_SECTION:
                    continue
                zon.append(dict(old_game_id=str(gid), esb_id=p["esbId"], depth=ZONE_DEPTH[z["lineOfScrimmageDistance"]],
                                section=ZONE_SECTION[z["section"]], attempts=int(z.get("attempts") or 0),
                                completions=int(z.get("completions") or 0), yards=int(z.get("yards") or 0),
                                touchdowns=int(z.get("touchdowns") or 0), interceptions=int(z.get("interceptions") or 0)))
    rec = pd.DataFrame(rec, columns=["old_game_id", "esb_id", "targets", "receptions", "avg_air_yards"])
    zon = pd.DataFrame(zon, columns=["old_game_id", "esb_id", "depth", "section", "attempts", "completions", "yards",
                                     "touchdowns", "interceptions"])
    if len(zon):
        # now and then (~1% of passer-games) NGS lists a zone twice with different numbers: drop those passer-games
        dup = zon.duplicated(["old_game_id", "esb_id", "depth", "section"], keep=False)
        bad = set(zip(zon.old_game_id[dup], zon.esb_id[dup]))
        zon = zon[[k not in bad for k in zip(zon.old_game_id, zon.esb_id)]].reset_index(drop=True)
    return rec, zon


def receiver_air_yards(game_ids, cache_dir=None, workers=8):
    """The receivers table of game_tables()."""
    return game_tables(game_ids, cache_dir, workers)[0]
