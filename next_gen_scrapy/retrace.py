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

import numpy as np
import pandas as pd

from .pbp_match import line_fit, match_to_pbp, score_chart
from .routes import detect_routes, route_alternatives

GEO_WEIGHT = 0.05            # nats per pixel of extra join cost: geometry breaks near-ties only
UNSURE_BELOW = 0.6           # re-read charts with any catch below this location confidence
DOT_DISAGREE_YD = 2.5
ROUTE_FIT_HINT = True        # re-reads score every candidate route against the receiver's targets (pbp_match.line_fit)        # ... or any catch this far from the QB's pass-chart dot for the same throw
_BASE_COLS = ["game_id", "season", "season_type", "week", "team", "esb_id", "name", "position"]


def _rows(base, routes):
    rows = []
    for i, r in enumerate(routes, 1):
        for j, ((x, y), seg) in enumerate(zip(r["pts"], r["seg"])):
            rows.append(dict(base, route_id=i, route_type=r["route_type"], touchdown=r["td"],
                             start_ok=r["start_ok"], gap=bool(r.get("gap", False)), point=j, segment=seg,
                             x_coord=round(float(x), 2), y_coord=round(float(y), 2)))
    return pd.DataFrame(rows)


def retrace_routes(route_df, pbp_df, chart_qc=None, root=".", geo_weight=GEO_WEIGHT, unsure_below=UNSURE_BELOW,
                   max_alts=40, images=None, pass_matches=None):
    """
    route_df: point-per-row route data (route_coords.csv format). chart_qc: chart_qc.csv (for image
    paths). images (instead of chart_qc, e.g. charts read in memory): {(game_id, esb_id): (image, chart metadata)}.
    pass_matches (optional): the QB charts' passes, matched without route links. Each completion's dot there is a
    second drawing of the catch point: a chart is also re-read when any catch lands 2.5+ yds from its dot, and the
    readings are scored with the dots.
    Returns (route_df with re-read charts swapped in, report dict).
    """
    matched, params = match_to_pbp(route_df, pbp_df, "route", return_params=True, pass_matches=pass_matches)
    if not len(matched):
        return route_df, dict(unsure_charts=0, reread=0)
    unsure = matched[(matched["bucket"] != "INCOMPLETE") & (matched["location_prob"] < unsure_below)]
    link = None
    if pass_matches is not None and len(pass_matches):
        pm = pass_matches[pass_matches["pass_type"].isin(["COMPLETE", "TOUCHDOWN"])]
        link = pd.DataFrame({"game_id": pm["game_id"].values, "play_id": pm["play_id"].values,
                             "link_x": pm["x_coord"].values, "link_y": pm["y_coord"].values}).drop_duplicates(["game_id", "play_id"])
        comp = matched[matched["bucket"] != "INCOMPLETE"].merge(link, on=["game_id", "play_id"])
        if len(comp):
            ci = comp["path_segment"].map(lambda seg: max([i for i, s in enumerate(seg) if s != "after_catch"], default=len(seg) - 1))
            cx = [px[i] for px, i in zip(comp["path_x"], ci)]
            cy = [py[i] for py, i in zip(comp["path_y"], ci)]
            far = np.hypot(np.asarray(cx) - comp["link_x"], np.asarray(cy) - comp["link_y"]) >= DOT_DISAGREE_YD
            unsure = pd.concat([unsure, comp[far.to_numpy()]])
    charts = unsure[["chart_game_id", "chart_esb_id"]].drop_duplicates()
    keys = matched.groupby(["chart_game_id", "chart_esb_id"])[["game_id", "receiver_player_id"]].first()
    paths = chart_qc[chart_qc["type"] == "route"].set_index(["game_id", "esb_id"])["image"] if chart_qc is not None else {}
    passes = pbp_df[pbp_df["play_type"] == "pass"]

    changed, replacements = [], []
    for g, esb in charts.itertuples(index=False):
        if images is not None:
            if (g, esb) not in images:
                continue
            img, meta = images[(g, esb)]
        else:
            img = paths.get((g, esb))
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
            alts = []
        # the stroke-graph readers' readings too (they read every line at once and often untangle differently) - and,
        # with the QB's dots for this receiver's catches, readings that untangle the crossings where the ball went
        game_id, pid = keys.loc[(g, esb)]
        plays = passes[(passes["game_id"] == game_id) & (passes["receiver_player_id"] == pid)]
        old = route_df[(route_df["game_id"] == g) & (route_df["esb_id"] == esb)]
        base = old.iloc[0][_BASE_COLS].to_dict()
        hinted = [expected]
        if link is not None and pass_matches is not None and "receiver_player_id" in pass_matches:
            mine = pass_matches[(pass_matches["game_id"] == game_id) & (pass_matches["receiver_player_id"] == pid)
                                & pass_matches["pass_type"].isin(["COMPLETE", "TOUCHDOWN"])]
            if len(mine):
                hinted.append(dict(expected, catches=mine[["x_coord", "y_coord"]].to_numpy(float).tolist()))
        if ROUTE_FIT_HINT and len(plays):
            # every candidate route scored against this receiver's real targets while the chart is read
            fit = lambda routes, _p=plays, _b=base: line_fit(_rows(_b, routes), _p, "route", params, link)
            hinted = hinted + [dict(h, fit=fit) for h in hinted]
        for exp in hinted:
            for method in ("graph", "layers"):
                try:
                    alts.append((0.0,) + tuple(detect_routes(img, exp, method=method)))
                except Exception:
                    pass
        if not alts:
            continue
        # the chart's current reading (detect_routes' graph reader, normally) is the baseline: an alternative
        # replaces it only if it explains this receiver's real targets better. The alternatives come from the
        # original tracer's ways of joining crossings (their geometry cost is relative to each other)
        scored = [(score_chart(old, plays, "route", params, link), -1, old)]
        for k, (geo, routes, _info) in enumerate(alts):
            rows = _rows(base, routes)
            if not len(rows):
                continue
            scored.append((score_chart(rows, plays, "route", params, link) + geo_weight * geo, k, rows))
        best = min(scored, key=lambda t: t[0])
        if best[1] != -1:                      # -1 is the reading the chart already has
            changed.append((g, esb))
            replacements.append(best[2])

    if not changed:
        return route_df, dict(unsure_charts=len(charts), reread=0)
    drop = pd.MultiIndex.from_tuples(changed, names=["game_id", "esb_id"])
    keep = ~route_df.set_index(["game_id", "esb_id"]).index.isin(drop)
    out = pd.concat([route_df[keep]] + replacements, ignore_index=True)
    return out, dict(unsure_charts=len(charts), reread=len(changed))
