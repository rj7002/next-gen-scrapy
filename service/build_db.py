"""Build the SQLite database the API serves from the pbp-matched parquet files.

Run this after service/build_matches.py (which does the actual scrape+match work):

    .venv/bin/python service/build_matches.py     # once per season, needs network (nflreadpy)
    .venv/bin/python service/build_db.py           # rebuild the queryable DB from those parquets
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MATCHED_DIR = REPO_ROOT / "service" / "matched"
DB_PATH = REPO_ROOT / "service" / "ngs.db"

KINDS = ["pass", "route", "carry"]
TABLE_NAME = {"pass": "passes", "route": "routes", "carry": "carries"}

# Indexed for the filters/sorts an analytics UI actually uses - the generic filter endpoint can
# still filter on any column, these just keep the common ones fast.
INDEX_COLS = [
    "season", "season_type", "week", "team", "posteam", "defteam", "name", "position",
    "down", "qtr", "play_type", "pass_type", "route_type", "gain_class",
]

ARRAY_COLS = ["path_x", "path_y", "path_segment"]


def _load_kind(kind: str) -> pd.DataFrame:
    files = sorted(MATCHED_DIR.glob(f"{kind}_*.parquet"))
    if not files:
        raise RuntimeError(f"no matched parquet files for '{kind}' in {MATCHED_DIR} - run build_matches.py first")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True, sort=False)

    for col in ARRAY_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: json.dumps(list(v)) if isinstance(v, np.ndarray) else None)

    # sqlite3/pandas can't bind pandas' nullable StringDtype/boolean directly - normalize to plain
    # python objects (None for NA) so to_sql's executemany doesn't choke on pd.NA.
    for col in df.columns:
        if str(df[col].dtype) in ("boolean", "Int64", "Int32"):
            df[col] = df[col].astype(object).where(df[col].notna(), None)
        elif pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype(object).where(df[col].notna(), None)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str).where(df[col].notna(), None)

    return df


def build() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()

    con = sqlite3.connect(DB_PATH)
    try:
        for kind in KINDS:
            table = TABLE_NAME[kind]
            print(f"loading matched {kind} parquet files...")
            df = _load_kind(kind)
            print(f"{table}: {len(df)} rows, {len(df.columns)} columns")

            df.to_sql(table, con, if_exists="replace", index=False, chunksize=5000)

            for col in INDEX_COLS:
                if col in df.columns:
                    con.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON {table} ("{col}")')
            con.commit()
            print(f"  indexed {table}")

        con.execute("PRAGMA optimize")
    finally:
        con.close()

    print(f"done: {DB_PATH} ({DB_PATH.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    build()
