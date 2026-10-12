"""
Optional: tracking from your own copy of the NFL Big Data Bowl 2027 data (Kaggle) - it isn't shipped with the package.

It tracks the players drafted 2023-2025 on every play they were in (2023-2025 seasons) - no ball and only one
running back, but someone tracked is on the field for most runs (64% in 2023, 90% in 2024, 98% in 2025). For runs
the useful part is where those players were when the run ended: the tackler is at the runner, and the others have
converged on him - something play-by-play doesn't record. Tracked offensive linemen also say where the ball was at
each snap, so where the previous run ended. match_to_pbp(..., tracking=load_play_tracking(...)) uses all of it to
pin carries; load_tackle_points is the older, tackler-only version.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

DEFAULT_DIRS = [os.environ.get("NGS_BDB_DIR", ""), os.path.expanduser("~/Downloads/nfl-big-data-bowl-2027")]


def find_dir(path=None):
    """The Big Data Bowl 2027 folder (path, $NGS_BDB_DIR or ~/Downloads/nfl-big-data-bowl-2027), or None."""
    for d in ([path] if path else []) + DEFAULT_DIRS:
        if d and os.path.exists(os.path.join(d, "player_play.csv")):
            return d
    return None


def _los_x(n, own, direction):
    if direction == "right":
        return 10 + n if own or n == 50 else 110 - n
    return 110 - n if own or n == 50 else 10 + n


def load_tackle_points(season, path=None, cache_dir=None):
    """
    One row per run play where a tracked defender made (or assisted) the tackle or pushed the runner out of bounds:
    old_game_id, play_id, end_x, end_y - the tackler's position at that moment in chart coordinates (x across the
    field, negative = the offense's left; y yards past the line of scrimmage). Empty if the data isn't there.
    """
    import pyarrow.csv as pv
    d = find_dir(path)
    cols = ["old_game_id", "play_id", "end_x", "end_y"]
    if d is None or not os.path.exists(os.path.join(d, f"game_tracking_{season}.csv")):
        return pd.DataFrame(columns=cols)
    cache = os.path.join(cache_dir, f"bdb_tackles_{season}.parquet") if cache_dir else None
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    pp = pd.read_csv(os.path.join(d, "player_play.csv"), low_memory=False,
                     usecols=["game_id", "play_id", "nfl_id", "team_abbr", "possession_team", "play_direction",
                              "yardline_side", "yardline_number", "tackle", "assist"])
    pp = pp[pp["game_id"].astype(str).str.startswith(str(season))]
    tk = pp[((pp["tackle"] == 1) | (pp["assist"] == 1)) & (pp["team_abbr"] != pp["possession_team"])]
    t = pv.read_csv(os.path.join(d, f"game_tracking_{season}.csv"),
                    convert_options=pv.ConvertOptions(include_columns=["game_id", "play_id", "nfl_id", "x", "y", "event"])).to_pandas()
    t = t.merge(tk[["game_id", "play_id", "nfl_id", "tackle"]], on=["game_id", "play_id", "nfl_id"])
    ev = t[t["event"].isin(["tackle", "out_of_bounds"])].groupby(["game_id", "play_id", "nfl_id"]).first().reset_index()
    ev = ev.sort_values("tackle", ascending=False).groupby(["game_id", "play_id"]).first().reset_index()
    meta = pp.drop_duplicates(["game_id", "play_id"]).set_index(["game_id", "play_id"])
    m = meta.loc[list(zip(ev["game_id"], ev["play_id"]))]
    los = [_los_x(n, side == pos, dr) for n, side, pos, dr in zip(m["yardline_number"], m["yardline_side"],
                                                                     m["possession_team"], m["play_direction"])]
    right = (m["play_direction"] == "right").to_numpy()
    x, y = ev["x"].to_numpy(float), ev["y"].to_numpy(float)
    out = pd.DataFrame({"old_game_id": ev["game_id"].astype(str), "play_id": ev["play_id"].astype(float),
                        "end_x": np.where(right, 26.65 - y, y - 26.65),
                        "end_y": np.where(right, x - np.array(los), np.array(los) - x)})
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        out.to_parquet(cache)
    return out


MOMENTS = ("snap", "snap05", "snap10", "handoff", "contact", "end")
_END_EVENTS = ("tackle", "out_of_bounds", "touchdown", "fumble")
_GROUPS = {"DE": "DL", "DT": "DL", "NT": "DL", "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB", "CB": "DB", "SS": "DB",
           "FS": "DB", "DB": "DB", "S": "DB", "T": "OL", "G": "OL", "C": "OL", "TE": "TE", "WR": "WR", "RB": "RB", "FB": "RB"}


def load_play_tracking(season, path=None, cache_dir=None):
    """
    Every tracked player on every play, at the moments that say where a run went: the snap, 0.5 s and 1 s after it,
    the handoff, first contact and the end (tackle / out of bounds / touchdown / fumble). One row per
    (old_game_id, play_id, nfl_id, moment) with x, y in chart coordinates (x across the field, negative = the
    offense's left; y yards past the line of scrimmage), `offense`, `group` (DL / LB / DB / OL / TE / WR / RB) and
    `tackler` (made or assisted the tackle). Empty if the data isn't there. match_to_pbp(..., tracking=) takes it.
    """
    import pyarrow.csv as pv
    d = find_dir(path)
    cols = ["old_game_id", "play_id", "nfl_id", "moment", "x", "y", "offense", "group", "tackler"]
    if d is None or not os.path.exists(os.path.join(d, f"game_tracking_{season}.csv")):
        return pd.DataFrame(columns=cols)
    cache = os.path.join(cache_dir, f"bdb_players_{season}.parquet") if cache_dir else None
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    pp = pd.read_csv(os.path.join(d, "player_play.csv"), low_memory=False,
                     usecols=["game_id", "play_id", "nfl_id", "team_abbr", "possession_team", "play_direction",
                              "yardline_side", "yardline_number", "tackle", "assist"])
    pp = pp[pp["game_id"].astype(str).str.startswith(str(season))]
    pl = pd.read_csv(os.path.join(d, "players.csv"), usecols=["nfl_id", "nfl_position"])
    t = pv.read_csv(os.path.join(d, f"game_tracking_{season}.csv"), convert_options=pv.ConvertOptions(
        include_columns=["game_id", "play_id", "nfl_id", "time", "x", "y", "event"])).to_pandas()
    t["time"] = pd.to_datetime(t["time"])
    ev = t[t["event"].notna()].groupby(["game_id", "play_id", "event"])["time"].min().unstack()
    when = pd.DataFrame({"snap": ev.get("ball_snap"), "handoff": ev.get("handoff"), "contact": ev.get("first_contact"),
                         "end": ev.reindex(columns=list(_END_EVENTS)).min(axis=1)})
    when["snap05"] = when["snap"] + pd.Timedelta(seconds=0.5)
    when["snap10"] = when["snap"] + pd.Timedelta(seconds=1.0)
    t = t.join(when, on=["game_id", "play_id"])
    parts = []
    for k in MOMENTS:
        q = t[t[k].notna()]
        dt = (q["time"] - q[k]).dt.total_seconds().abs()
        q = q.assign(dt=dt)[dt <= 0.06].sort_values("dt").drop_duplicates(["game_id", "play_id", "nfl_id"])
        parts.append(q[["game_id", "play_id", "nfl_id", "x", "y"]].assign(moment=k))
    P = pd.concat(parts, ignore_index=True)
    meta = pp.drop_duplicates(["game_id", "play_id"]).set_index(["game_id", "play_id"])
    P = P.join(meta[["possession_team", "play_direction", "yardline_side", "yardline_number"]], on=["game_id", "play_id"])
    P = P.merge(pp[["game_id", "play_id", "nfl_id", "team_abbr", "tackle", "assist"]], on=["game_id", "play_id", "nfl_id"], how="left")
    P = P.merge(pl, on="nfl_id", how="left").dropna(subset=["play_direction", "yardline_number"])
    los = np.array([_los_x(n, s == p, dr) for n, s, p, dr in zip(P["yardline_number"], P["yardline_side"],
                                                                  P["possession_team"], P["play_direction"])], float)
    right = (P["play_direction"] == "right").to_numpy()
    x, y = P["x"].to_numpy(float), P["y"].to_numpy(float)
    out = pd.DataFrame({"old_game_id": P["game_id"].astype(str), "play_id": P["play_id"].astype(float),
                        "nfl_id": P["nfl_id"], "moment": P["moment"],
                        "x": np.where(right, 26.65 - y, y - 26.65), "y": np.where(right, x - los, los - x),
                        "offense": (P["team_abbr"] == P["possession_team"]).to_numpy(),
                        "group": P["nfl_position"].map(_GROUPS).fillna("other"),
                        "tackler": ((P["tackle"] == 1) | (P["assist"] == 1)).to_numpy()})
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        out.to_parquet(cache)
    return out
