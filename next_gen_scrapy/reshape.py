"""
Reduce a carry/route chart DataFrame from one row per point to one row per carry/route, with the
whole traced line folded into array columns instead of spread across rows.

    from next_gen_scrapy import get_carries, get_routes, to_paths

    carries = get_carries("Bijan Robinson", 2025, weeks=[1, 2, 3])
    carry_paths = to_paths(carries, "carry")     # one row per carry: path_x, path_y, y_coord, ...

The point-per-row shape from get_carries/get_routes is the source of truth - it's what
ngs-extract writes to CSV, and what plot_routes/plot_carries want for drawing. Reach for to_paths
when you want one row per entity instead: feature engineering, or matching to pbp (match_to_pbp
uses this same reduction internally).
"""
import pandas as pd

SCALAR_COLS = {
    "carry": ["season", "season_type", "week", "team", "esb_id", "name", "position",
              "gain_class", "touchdown", "fumble_lost", "handoff_ok"],
    "route": ["season", "season_type", "week", "team", "esb_id", "name", "position",
              "route_type", "touchdown", "start_ok"],
}


def to_paths(chart_df, kind):
    """
    kind: 'pass' | 'carry' | 'route'

    - pass: a pass chart is already one row per pass (a single point, not a path) - returned
      unchanged.
    - carry: one row per (game_id, carry_id): the scalar columns (name/team/gain_class/touchdown/
      fumble_lost/handoff_ok/...), plus path_x/path_y (the whole traced line, in point order) and
      y_coord (the final point). y_coord is the last point's own value, not
      path_y[-1] - path_y[0]: the chart's y-axis is already zeroed at the line of scrimmage, so a
      carry drawn starting behind the LOS (a shotgun handoff, say) would have its gain overstated
      by computing a delta from the line's own start instead of from the LOS.
    - route: one row per (game_id, route_id): the scalar columns (name/team/route_type/touchdown/
      start_ok/...), plus path_x/path_y/path_segment (segment: 'route' or 'after_catch') and
      catch_y (the target's depth - the last pre-catch point's own y value, i.e. air_yards; not
      the same as the route's final y, which also includes yards after the catch).
    """
    assert kind in ("pass", "carry", "route")
    if kind == "pass":
        return chart_df.copy()

    id_col = "carry_id" if kind == "carry" else "route_id"
    scalar_cols = SCALAR_COLS[kind]

    def _agg(g):
        g = g.sort_values("point")
        out = {c: g[c].iloc[0] for c in scalar_cols}
        out["path_x"] = g["x_coord"].to_numpy()
        out["path_y"] = g["y_coord"].to_numpy()
        if kind == "route":
            out["path_segment"] = g["segment"].to_numpy()
            pre = g[g["segment"] != "after_catch"]
            out["catch_y"] = pre["y_coord"].iloc[-1] if len(pre) else g["y_coord"].iloc[-1]
        else:
            out["y_coord"] = g["y_coord"].iloc[-1]
        return pd.Series(out)

    return (
        chart_df.groupby(["game_id", id_col], sort=False)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
