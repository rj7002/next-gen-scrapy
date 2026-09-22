"""
Turn scraped Next Gen Stats chart images into coordinate data.

Reads the charts downloaded by `ngs-scrape` ({Pass,Route,Carry}_Charts/<team>/<season>/<week>/images/*.jpeg,
with the chart metadata next to each image in ../data/*.json) and writes, in yards on the field
(x: -26.67 left sideline .. +26.67 right sideline, 0 = middle; y: yards past the line of scrimmage):

    pass_locations.csv   one row per pass          (COMPLETE / INCOMPLETE / INTERCEPTION / TOUCHDOWN)
    route_coords.csv     one row per point along each route (COMPLETE incl. after-catch / INCOMPLETE)
    carry_coords.csv     one row per point along each carry
    chart_qc.csv         one row per chart: expected vs. detected counts, so bad extractions are easy to find

Example:
    ngs-scrape --type pass -s 2025
    ngs-extract --type pass -s 2025
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from . import calib as K
from . import carries as ngs_carries
from . import passes as ngs_passes
from . import routes as ngs_routes

TYPES = {"pass": "Pass_Charts", "route": "Route_Charts", "carry": "Carry_Charts"}


def chart_info(meta, image_path):
    return dict(game_id=meta["gameId"], season=meta["season"], season_type=meta["seasonType"],
                week=meta["week"], team=meta.get("team", meta["teamId"]), esb_id=meta["esbId"],
                name=meta["playerName"], position=meta["position"])


def process_pass(image_path, meta):
    td = meta["touchdowns"]
    expected = {"COMPLETE": meta["completions"] - td, "TOUCHDOWN": td, "INTERCEPTION": meta["interceptions"]}
    # Incompletions are NOT constrained: the chart does not draw every attempted pass.
    im, found, lay = ngs_passes.detect_passes(image_path, expected)
    pts = lay.img_to_field([[f["cx"], f["cy"]] for f in found]) if found else []
    base = chart_info(meta, image_path)
    rows = [dict(base, pass_type=f["pass_type"], located=True,
                 x_coord=round(float(p[0]), 2), y_coord=round(float(p[1]), 2))
            for f, p in zip(found, pts)]
    got = {k: sum(f["pass_type"] == k for f in found) for k in ("COMPLETE", "TOUCHDOWN", "INTERCEPTION", "INCOMPLETE")}
    # Every completion, touchdown and interception IS drawn on the chart, so if one was not found it is
    # hidden under another ring. Emit it with no coordinates rather than dropping it, so the row counts
    # still match the box score. (Incompletions are not padded: the chart genuinely omits some.)
    for ptype, n in expected.items():
        for _ in range(max(0, n - got[ptype])):
            rows.append(dict(base, pass_type=ptype, located=False, x_coord=None, y_coord=None))
    qc = dict(base, depth_yd=round(float(lay.row_to_yard(0.0)), 1), expected_complete=expected["COMPLETE"], detected_complete=got["COMPLETE"],
              expected_td=td, detected_td=got["TOUCHDOWN"], expected_int=meta["interceptions"],
              detected_int=got["INTERCEPTION"], attempts=meta["attempts"],
              detected_incomplete=got["INCOMPLETE"],
              expected_incomplete=meta["attempts"] - meta["completions"] - meta["interceptions"])
    qc["counts_ok"] = (qc["expected_complete"] == qc["detected_complete"] and td == got["TOUCHDOWN"]
                       and meta["interceptions"] == got["INTERCEPTION"])
    return rows, qc


def process_route(image_path, meta):
    routes, info = ngs_routes.detect_routes(
        image_path, {"receptions": meta["receptions"], "touchdowns": meta["touchdowns"],
                     "targets": meta.get("targets")})
    base = chart_info(meta, image_path)
    rows = []
    for i, r in enumerate(routes, 1):
        for j, ((x, y), seg) in enumerate(zip(r["pts"], r["seg"])):
            rows.append(dict(base, route_id=i, route_type=r["route_type"], touchdown=r["td"], start_ok=r["start_ok"], point=j,
                             segment=seg, x_coord=round(float(x), 2), y_coord=round(float(y), 2)))
    n_c = sum(r["route_type"] == "COMPLETE" for r in routes)
    qc = dict(base, depth_yd=info["depth_yd"], expected_receptions=meta["receptions"], detected_complete=n_c,
              expected_targets=meta.get("targets"), detected_routes=len(routes),
              detected_incomplete=len(routes) - n_c, expected_td=meta["touchdowns"], detected_td_rings=info["n_td_rings"],
              counts_ok=(n_c == meta["receptions"] and info["n_td_rings"] == meta["touchdowns"]))
    return rows, qc


def process_carry(image_path, meta):
    carries, info = ngs_carries.detect_carries(
        image_path, {"carries": meta["carries"], "touchdowns": meta["touchdowns"]})
    base = chart_info(meta, image_path)
    rows = []
    for i, c in enumerate(carries, 1):
        for j, (x, y) in enumerate(c["pts"]):
            rows.append(dict(base, carry_id=i, gain_class=c["color"], touchdown=c["td"], fumble_lost=c["fumble"],
                             handoff_ok=c["handoff_ok"], point=j, x_coord=round(float(x), 2), y_coord=round(float(y), 2)))
    qc = dict(base, depth_yd=info["depth_yd"], expected_carries=meta["carries"], detected_carries=len(carries),
              expected_td=meta["touchdowns"], detected_td_rings=info["n_td_rings"],
              counts_ok=(len(carries) == meta["carries"] and info["n_td_rings"] == meta["touchdowns"]))
    return rows, qc


PROCESSORS = {"pass": process_pass, "route": process_route, "carry": process_carry}
OUTPUTS = {"pass": "pass_locations.csv", "route": "route_coords.csv", "carry": "carry_coords.csv"}


def work(args):
    kind, image_path = args
    with open(image_path.replace(os.sep + "images" + os.sep, os.sep + "data" + os.sep)
              .replace(".jpeg", ".json")) as f:
        meta = json.load(f)
    try:
        rows, qc = PROCESSORS[kind](image_path, meta)
        qc["error"] = ""
    except Exception as e:      # keep going: one odd chart should not stop a season
        rows, qc = [], dict(chart_info(meta, image_path), counts_ok=False, error=repr(e))
    qc["type"] = kind
    qc["image"] = image_path
    return rows, qc


def main():
    ap = argparse.ArgumentParser(description="Extract coordinates from scraped Next Gen Stats charts")
    ap.add_argument("--type", choices=list(TYPES) + ["all"], default="all")
    ap.add_argument("--root", default=".", help="folder holding the *_Charts folders")
    ap.add_argument("--out", default=".", help="folder to write the CSVs to")
    ap.add_argument("-s", "--seasons", nargs="+", default=None)
    ap.add_argument("-t", "--teams", nargs="+", default=None)
    ap.add_argument("-w", "--weeks", nargs="+", default=None)
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    kinds = list(TYPES) if args.type == "all" else [args.type]
    qcs = []
    for kind in kinds:
        images = sorted(glob.glob(os.path.join(args.root, TYPES[kind], "*", "*", "*", "images", "*.jpeg")))
        keep = []
        for p in images:
            team, season, week = p.split(os.sep)[-5:-2]
            if (args.teams and team not in {t.upper() for t in args.teams}) or \
               (args.seasons and season not in args.seasons) or (args.weeks and week not in args.weeks):
                continue
            keep.append((kind, p))
        print("%s: %d charts" % (kind, len(keep)), flush=True)
        if not keep:
            continue
        with ProcessPoolExecutor(args.jobs) as ex:
            results = list(ex.map(work, keep, chunksize=4))
        rows = [r for rs, _ in results for r in rs]
        qcs += [q for _, q in results]
        pd.DataFrame(rows).to_csv(os.path.join(args.out, OUTPUTS[kind]), index=False)
        bad = sum(1 for _, q in results if not q["counts_ok"])
        print("  wrote %s (%d rows); %d/%d charts have count mismatches vs. the chart metadata"
              % (OUTPUTS[kind], len(rows), bad, len(keep)))
    if qcs:
        pd.DataFrame(qcs).to_csv(os.path.join(args.out, "chart_qc.csv"), index=False)
        print("wrote chart_qc.csv")


if __name__ == "__main__":
    sys.exit(main())
