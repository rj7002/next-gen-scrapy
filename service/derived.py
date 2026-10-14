"""Columns worked out from a play's other columns - for filters that need them (defensive look, tacklers, ...).
column(frame, col) gives the column for a frame of plays, whether the frame has it or it's derived here."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

def _names(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    have = [c for c in cols if c in frame]
    if not have:
        return pd.Series([None] * len(frame), index=frame.index, dtype=object)
    return frame[have].apply(lambda r: [v for v in r if isinstance(v, str) and v] or None, axis=1)


def _look(p) -> str | None:
    if not isinstance(p, str) or not p:
        return None
    dbs = sum(int(n) for n, pos in re.findall(r"(\d+)\s*(CB|FS|SS|S|DB)\b", p))
    return None if not dbs else "BASE" if dbs <= 4 else "NICKEL" if dbs == 5 else "DIME"


def _favored(frame):
    if not {"spread_line", "posteam", "home_team"} <= set(frame.columns):
        return pd.Series([None] * len(frame), index=frame.index, dtype=object)
    home = frame["posteam"] == frame["home_team"]
    s = pd.to_numeric(frame["spread_line"], errors="coerce")
    fav = np.where(home, s > 0, s < 0)
    return pd.Series(np.where(s.isna() | (s == 0), None, fav), index=frame.index, dtype=object)


DERIVED = {
    "defense_look": lambda f: f["defense_personnel"].map(_look) if "defense_personnel" in f else None,
    "tackler": lambda f: _names(f, ["solo_tackle_1_player_name", "solo_tackle_2_player_name", "assist_tackle_1_player_name",
                                    "assist_tackle_2_player_name", "assist_tackle_3_player_name", "assist_tackle_4_player_name",
                                    "tackle_with_assist_1_player_name", "tackle_with_assist_2_player_name"]),
    "pass_defender": lambda f: _names(f, ["pass_defense_1_player_name", "pass_defense_2_player_name"]),
    "qb_hitter": lambda f: _names(f, ["qb_hit_1_player_name", "qb_hit_2_player_name"]),
    "tfl_by": lambda f: _names(f, ["tackle_for_loss_1_player_name", "tackle_for_loss_2_player_name"]),
    "forced_fumble_by": lambda f: _names(f, ["forced_fumble_player_1_player_name", "forced_fumble_player_2_player_name"]),
    "penalty_on": lambda f: (pd.Series(np.where(f["penalty_team"].isna(), None,
                                                np.where(f["penalty_team"] == f["posteam"], "offense", "defense")),
                                       index=f.index, dtype=object)
                             if {"penalty_team", "posteam"} <= set(f.columns) else None),
    "team_favored": _favored,
    "yac_over_expected": lambda f: (pd.to_numeric(f["yards_after_catch"], errors="coerce")
                                    - pd.to_numeric(f["xyac_mean_yardage"], errors="coerce")).round(1)
                                   if {"yards_after_catch", "xyac_mean_yardage"} <= set(f.columns) else None,
    "ngs_top_speed": lambda f: pd.to_numeric(f["ngs_max_speed"], errors="coerce").round(1)   # NGS gives mph
                               if "ngs_max_speed" in f else None,
}


def column(frame: pd.DataFrame | None, col: str) -> pd.Series | None:
    """The column `col` of `frame` (as stored, or derived above), or None if this frame can't give it."""
    if frame is None or not len(frame):
        return None
    if col in DERIVED:
        return DERIVED[col](frame)
    return frame[col] if col in frame else None


def with_columns(frame: pd.DataFrame, cols) -> pd.DataFrame:
    """frame plus whichever of `cols` it lacks and can derive."""
    add = {c: column(frame, c) for c in cols if c in DERIVED and c not in frame}
    add = {c: v for c, v in add.items() if v is not None}
    return frame.assign(**add) if add else frame
