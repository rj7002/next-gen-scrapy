"""
Re-read tangled route charts, picking the untangling that fits the real plays best.

Where several routes cross, the tracer can join the wrong pieces and hand one route another route's
catch point. Geometry alone can't always tell the readings apart - but the plays can: each receiver's
targets in pbp say how deep, to which side, and with how much yards-after-catch every catch was made.
So for every chart where the matcher is unsure, the alternative readings from
routes.route_alternatives() are each scored against that receiver's plays (plus a small penalty for
less natural joins), and the best-fitting reading replaces the default one.

    from next_gen_scrapy.retrace import retrace_routes
    routes_fixed, report = retrace_routes(route_df, pbp_df, chart_qc_df, root=".")

Because this step uses pbp to choose, judge it on evidence it doesn't use (e.g. the QB's pass chart
dot for the same throw), not on the route matches' own confidence.
"""
import json
import os

import pandas as pd

from .pbp_match import match_to_pbp, score_chart
from .routes import route_alternatives

GEO_WEIGHT = 0.05            # nats per pixel of extra join cost: geometry breaks near-ties only
UNSURE_BELOW = 0.6           # re-read charts with any catch below this location confidence
_BASE_COLS = ["game_id", "season", "season_type", "week", "team", "esb_id", "name", "position"]


def _rows(base, routes):
    rows = []
    for i, r in enumerate(routes, 1):
        for j, ((x, y), seg) in enumerate(zip(r["pts"], r["seg"])):
            rows.append(dict(base, route_id=i, route_type=r["route_type"], touchdown=r["td"],
                             start_ok=r["start_ok"], point=j, segment=seg,
                             x_coord=round(float(x), 2), y_coord=round(float(y), 2)))
    return pd.DataFrame(rows)


def retrace_routes(route_df, pbp_df, chart_qc, root=".", geo_weight=GEO_WEIGHT, unsure_below=UNSURE_BELOW,
                   max_alts=40):
    """
    route_df: point-per-row route data (route_coords.csv format). chart_qc: chart_qc.csv (for image
    paths). Returns (route_df with re-read charts swapped in, report dict).
    """
    matched, params = match_to_pbp(route_df, pbp_df, "route", return_params=True)
    if not len(matched):
        return route_df, dict(unsure_charts=0, reread=0)
    unsure = matched[(matched["bucket"] != "INCOMPLETE") & (matched["location_prob"] < unsure_below)]
    charts = unsure[["chart_game_id", "chart_esb_id"]].drop_duplicates()
    keys = matched.groupby(["chart_game_id", "chart_esb_id"])[["game_id", "receiver_player_id"]].first()
    images = chart_qc[chart_qc["type"] == "route"].set_index(["game_id", "esb_id"])["image"]
    passes = pbp_df[pbp_df["play_type"] == "pass"]

    changed, replacements = [], []
    for g, esb in charts.itertuples(index=False):
        img = images.get((g, esb))
        if not isinstance(img, str):
            continue
        img = os.path.join(root, img)
        meta_path = img.replace(os.sep + "images" + os.sep, os.sep + "data" + os.sep).rsplit(".", 1)[0] + ".json"
        if not (os.path.exists(img) and os.path.exists(meta_path)):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        expected = {"receptions": meta["receptions"], "touchdowns": meta["touchdowns"],
                    "targets": meta.get("targets"), "max_yards": meta.get("receivingYards")}
        try:
            alts = route_alternatives(img, expected, max_alts=max_alts)
        except Exception:
            continue
        if len(alts) <= 1:
            continue
        game_id, pid = keys.loc[(g, esb)]
        plays = passes[(passes["game_id"] == game_id) & (passes["receiver_player_id"] == pid)]
        old = route_df[(route_df["game_id"] == g) & (route_df["esb_id"] == esb)]
        base = old.iloc[0][_BASE_COLS].to_dict()
        scored = []
        for k, (geo, routes, _info) in enumerate(alts):
            rows = _rows(base, routes)
            if not len(rows):
                continue
            scored.append((score_chart(rows, plays, "route", params) + geo_weight * geo, k, rows))
        best = min(scored, key=lambda t: t[0])
        if best[1] != 0:                       # 0 is detect_routes' own reading
            changed.append((g, esb))
            replacements.append(best[2])

    if not changed:
        return route_df, dict(unsure_charts=len(charts), reread=0)
    drop = pd.MultiIndex.from_tuples(changed, names=["game_id", "esb_id"])
    keep = ~route_df.set_index(["game_id", "esb_id"]).index.isin(drop)
    out = pd.concat([route_df[keep]] + replacements, ignore_index=True)
    return out, dict(unsure_charts=len(charts), reread=len(changed))
