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
from . import statbox as ngs_statbox

TYPES = {"pass": "Pass_Charts", "route": "Route_Charts", "carry": "Carry_Charts"}


def chart_info(meta, image_path):
    return dict(game_id=meta["gameId"], season=meta["season"], season_type=meta["seasonType"],
                week=meta["week"], team=meta.get("team", meta["teamId"]), esb_id=meta["esbId"],
                name=meta["playerName"], position=meta["position"])


STAT_KEYS = {"pass": ("completions", "attempts", "passingYards", "touchdowns", "interceptions"),
             "route": ("receptions", "receivingYards", "touchdowns", "targets"),
             "carry": ("carries", "rushingYards", "touchdowns")}
_READERS = {"pass": ngs_statbox.read_pass_stats, "route": ngs_statbox.read_route_stats, "carry": ngs_statbox.read_carry_stats}


def chart_stats(kind, image, meta):
    """
    (decoded image, stats, source): the counts the chart itself prints in its stat box, which is what is drawn on
    it. NGS's listing sometimes carries no stats for a chart (mostly 2021-23 playoff games), and sometimes the
    game's final line while the chart was made before the game ended - in both cases the image is the truth. Keys
    the stat box doesn't show (e.g. interceptions on the oldest layout) come from the listing. Stats the listing
    lacks are also filled into `meta` itself, so whoever holds the chart dict sees them.
    """
    im = K.read_image(image)
    read = _READERS[kind](im) or {}
    stats = {k: meta.get(k) for k in STAT_KEYS[kind]}
    for k, v in read.items():
        if v is not None:
            stats[k] = v
            meta.setdefault(k, v)
    missing = [k for k in STAT_KEYS[kind] if k not in ("targets", "attempts", "passingYards", "receivingYards", "rushingYards")
               and stats.get(k) is None]
    if missing:
        raise ValueError(f"chart stats unreadable and not in the listing: {', '.join(missing)}")
    return im, stats, "image" if read else "listing"


def process_pass(image_path, meta):
    im0, st, src = chart_stats("pass", image_path, meta)
    td = st["touchdowns"]
    expected = {"COMPLETE": st["completions"] - td, "TOUCHDOWN": td, "INTERCEPTION": st["interceptions"]}
    # Incompletions are NOT constrained: the chart does not draw every attempted pass - only capped
    if st.get("attempts") is not None:
        expected["INCOMPLETE_MAX"] = st["attempts"] - st["completions"] - st["interceptions"]
    im, found, lay = ngs_passes.detect_passes(im0, expected)
    pts = lay.img_to_field([[f["cx"], f["cy"]] for f in found]) if found else []
    base = chart_info(meta, image_path)
    # A ring past the chart's deepest gridline (the art stops at about +50) has its depth extrapolated from the
    # calibration; checked against play-by-play air yards those are as accurate as the rest (median 1.4 yd), so
    # they're kept, flagged `extrapolated`.
    rows = [dict(base, pass_type=f["pass_type"], located=True, extrapolated=bool(p[1] > lay.max_reliable_yard),
                 x_coord=round(float(p[0]), 2), y_coord=round(float(p[1]), 2))
            for f, p in zip(found, pts)]
    n_extrapolated = sum(r["extrapolated"] for r in rows)
    got = {k: sum(f["pass_type"] == k for f in found) for k in ("COMPLETE", "TOUCHDOWN", "INTERCEPTION", "INCOMPLETE")}
    # Every completion, touchdown and interception IS drawn on the chart, so if one was not found it is
    # hidden under another ring. Emit it with no coordinates rather than dropping it, so the row counts
    # still match the box score. (Incompletions are not padded: the chart genuinely omits some.)
    for ptype, n in expected.items():
        if ptype not in got:
            continue
        for _ in range(max(0, n - got[ptype])):
            rows.append(dict(base, pass_type=ptype, located=False, extrapolated=False, x_coord=None, y_coord=None))
    qc = dict(base, depth_yd=round(lay.max_reliable_yard, 1), expected_complete=expected["COMPLETE"], detected_complete=got["COMPLETE"],
              expected_td=td, detected_td=got["TOUCHDOWN"], expected_int=st["interceptions"],
              detected_int=got["INTERCEPTION"], attempts=st.get("attempts"),
              detected_incomplete=got["INCOMPLETE"],
              expected_incomplete=(st["attempts"] - st["completions"] - st["interceptions"]) if st.get("attempts") is not None else None,
              n_deep_dropped=0, n_extrapolated=n_extrapolated, stats_from=src)
    qc["counts_ok"] = (qc["expected_complete"] == qc["detected_complete"] and td == got["TOUCHDOWN"]
                       and st["interceptions"] == got["INTERCEPTION"])
    return rows, qc


def process_route(image_path, meta):
    im0, st, src = chart_stats("route", image_path, meta)
    routes, info = ngs_routes.detect_routes(
        im0, {"receptions": st["receptions"], "touchdowns": st["touchdowns"],
              "targets": st.get("targets"), "max_yards": st.get("receivingYards")})
    base = chart_info(meta, image_path)
    rows = []
    for i, r in enumerate(routes, 1):
        for j, ((x, y), seg) in enumerate(zip(r["pts"], r["seg"])):
            rows.append(dict(base, route_id=i, route_type=r["route_type"], touchdown=r["td"], start_ok=r["start_ok"],
                             gap=bool(r.get("gap", False)), point=j, segment=seg,
                             x_coord=round(float(x), 2), y_coord=round(float(y), 2)))
    n_c = sum(r["route_type"] == "COMPLETE" for r in routes)
    qc = dict(base, depth_yd=info["depth_yd"], expected_receptions=st["receptions"], detected_complete=n_c,
              expected_targets=st.get("targets"), detected_routes=len(routes),
              detected_incomplete=len(routes) - n_c, expected_td=st["touchdowns"], detected_td_rings=info["n_td_rings"],
              n_deep_truncated=info["n_deep_truncated"], n_deep_dropped=info["n_deep_dropped"],
              detected_td_routes=sum(bool(r["td"]) for r in routes),
              n_gap_routes=sum(bool(r.get("gap", False)) for r in routes),
              counts_ok=(n_c == st["receptions"] and sum(bool(r["td"]) for r in routes) == st["touchdowns"]),
              reader=info.get("method"), stats_from=src)
    return rows, qc


def carry_colours(plays):
    """{'LOSS', 'SHORT', 'LONG': carries} a carry chart should draw for these pbp runs: red for lost yards, yellow 0-5,
    green 6+ and every touchdown (checked on tracking-identified runs: 5-yd runs are yellow, 6-yd green)."""
    y = plays["yards_gained"].astype(float)
    td = plays["touchdown"].fillna(0).astype(float) == 1 if "touchdown" in plays else pd.Series(False, index=plays.index)
    return {"LOSS": int(((y < 0) & ~td).sum()), "SHORT": int(((y >= 0) & (y <= 5) & ~td).sum()),
            "LONG": int(((y > 5) | td).sum())}


def _carry_rows(base, carries):
    rows = []
    for i, c in enumerate(carries, 1):
        for j, (x, y) in enumerate(c["pts"]):
            rows.append(dict(base, carry_id=i, gain_class=c["color"], touchdown=c["td"], fumble_lost=c["fumble"],
                             handoff_ok=c["handoff_ok"], gap=bool(c.get("gap", False)), point=j,
                             x_coord=round(float(x), 2), y_coord=round(float(y), 2)))
    return rows


def process_carry(image_path, meta, plays=None, recheck="mismatch"):
    """
    plays (optional): this back's pbp runs in the game. When the reading's colours don't add up to them (a yellow
    run read as part of a green one at a crossing, ...), the chart is read again with pbp's colour counts held, by
    every reader, and the reading that best fits the plays (pbp_match.score_chart) is kept. recheck="always" tries
    the other readings even when the colours add up.
    """
    im0, st, src = chart_stats("carry", image_path, meta)
    expected = {"carries": st["carries"], "touchdowns": st["touchdowns"], "max_yards": st.get("rushingYards")}
    carries, info = ngs_carries.detect_carries(im0, expected)
    base = chart_info(meta, image_path)
    rows = _carry_rows(base, carries)
    rechecked = False
    if plays is not None and len(plays) == st["carries"]:
        want = carry_colours(plays)
        have = {k: sum(c["color"] == k for c in carries) for k in want}
        if have != want or recheck == "always":
            from . import pbp_match as PM
            params = PM._league_params("carry", pd.DataFrame([base])) or PM._calibrate("carry", [])
            scored = [(PM.score_chart(pd.DataFrame(rows), plays, "carry", params), rows, carries, info)]
            for method in ("layers", "graph", "tracer"):
                try:
                    alt, alt_info = ngs_carries.detect_carries(im0, dict(expected, by_color=want), method=method)
                except Exception:
                    continue
                if len(alt) != st["carries"]:
                    continue
                alt_rows = _carry_rows(base, alt)
                scored.append((PM.score_chart(pd.DataFrame(alt_rows), plays, "carry", params), alt_rows, alt, alt_info))
            best = min(scored, key=lambda t: t[0])
            rechecked = best[1] is not rows
            _, rows, carries, info = best
    qc = dict(base, depth_yd=info["depth_yd"], expected_carries=st["carries"], detected_carries=len(carries),
              expected_td=st["touchdowns"], detected_td_rings=info["n_td_rings"],
              n_deep_truncated=info.get("n_deep_truncated", 0), n_deep_dropped=info.get("n_deep_dropped", 0),
              detected_td_carries=sum(bool(c["td"]) for c in carries),
              n_gap_carries=sum(bool(c.get("gap", False)) for c in carries),
              counts_ok=(len(carries) == st["carries"] and sum(bool(c["td"]) for c in carries) == st["touchdowns"]),
              reader=info.get("method"), stats_from=src, reread_with_pbp=rechecked)
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
