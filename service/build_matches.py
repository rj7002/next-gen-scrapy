"""Match every pass/carry/route chart to its pbp+FTN play, one season at a time, and cache the
result to parquet so build_db.py doesn't need network access or re-run the (slow) Hungarian match.

Run once per season you care about, or after ALL DATA/ is refreshed with new games:

    .venv/bin/python service/build_matches.py                 # all seasons found in ALL DATA/
    .venv/bin/python service/build_matches.py --seasons 2025   # just one
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from next_gen_scrapy import load_pbp_with_ftn, match_to_pbp
from next_gen_scrapy.retrace import retrace_routes

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "ALL DATA"
OUT_DIR = REPO_ROOT / "service" / "matched"

SOURCES = {
    "pass": "pass_locations.csv",
    "route": "route_coords.csv",
    "carry": "carry_coords.csv",
}


def _seasons_in(csv_path: Path) -> list[int]:
    return sorted(pd.read_csv(csv_path, usecols=["season"])["season"].unique().tolist())


def build(seasons: list[int] | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_seasons = sorted(set().union(*(_seasons_in(DATA_DIR / f) for f in SOURCES.values())))
    seasons = seasons or all_seasons
    print(f"seasons to build: {seasons}")

    # Load each chart CSV once; slice per season below rather than re-reading from disk each time.
    charts = {kind: pd.read_csv(DATA_DIR / fname, low_memory=False) for kind, fname in SOURCES.items()}
    chart_qc = pd.read_csv(DATA_DIR / "chart_qc.csv", low_memory=False)

    for season in seasons:
        print(f"\n=== {season} ===")
        pbp = load_pbp_with_ftn([season])
        print(f"pbp+ftn: {len(pbp)} rows")

        # routes first: a targeted receiver's route chart shows the same throw as the QB's pass chart,
        # so the matched routes are extra evidence for placing each pass
        route_matches = None
        for kind in ("route", "pass", "carry"):
            season_df = charts[kind][charts[kind]["season"] == season]
            if season_df.empty:
                continue
            if kind == "route":
                # re-read tangled charts: pick the untangling that fits the receiver's real targets
                season_df, report = retrace_routes(season_df, pbp, chart_qc, root=str(REPO_ROOT))
                print(f"route: re-read {report['reread']} of {report['unsure_charts']} unsure charts")
            extra = {"route_matches": route_matches} if kind == "pass" else {}
            matched = match_to_pbp(season_df, pbp, kind, **extra)
            if kind == "route":
                route_matches = matched
            conf = matched["location_confident"].mean() if len(matched) else 0
            est = matched["location_prob"].mean() if len(matched) else 0
            print(f"{kind}: {len(season_df)} chart rows -> {len(matched)} matched plays "
                  f"(est. location accuracy {est:.0%}, {conf:.0%} location-confident)")
            if len(matched):
                out_path = OUT_DIR / f"{kind}_{season}.parquet"
                matched.to_parquet(out_path, index=False)
                print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=None)
    args = ap.parse_args()
    build(args.seasons)


if __name__ == "__main__":
    main()
