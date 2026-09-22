"""
Match a chart DataFrame (from get_passes / get_carries / get_routes) to the play-by-play rows
for the same plays, optionally enriched with FTN charting.

There is no shared play id between an NGS chart and nflverse's pbp - a chart is just dots/lines
per player-game, in field yards, with no ordering tying a given dot to a given play. Matching them
is an assignment problem, not a join:

    from next_gen_scrapy import get_passes, load_pbp_with_ftn, match_to_pbp

    passes = get_passes("Tyler Shough", 2026, [1, 2])
    pbp = load_pbp_with_ftn(2026)                  # or your own pbp DataFrame, FTN-merged or not
    matches = match_to_pbp(passes, pbp, "pass")

This needs nflreadpy (`pip install "next-gen-scrapy[pbp]"`), which is not a core dependency of the
scrape/extract pipeline - only of this module.
"""
import numpy as np
import pandas as pd

from .reshape import to_paths

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as e:                             # pragma: no cover - scipy is a core dependency already
    raise ImportError("match_to_pbp needs scipy (already a core dependency - reinstall next-gen-scrapy)") from e

CHART_ID_COL = {"pass": "passer_player_id", "carry": "rusher_player_id", "route": "receiver_player_id"}
CHART_PLAY_TYPE = {"pass": "pass", "carry": "run", "route": "pass"}

# Chart columns promoted to their natural (unprefixed) name in match_to_pbp's output, since they ARE
# the feature this whole match exists to attach real-world context to. Verified against a real season:
# these never collide with a pbp/FTN column - EXCEPT "name" (pbp's own is nflverse's short form, e.g.
# "J.Allen", vs. the chart's display name, e.g. "Josh Allen") and "season"/"season_type"/"week" (equal
# to pbp's own on every matched row, by construction - the match only ever pairs a chart entity with a
# pbp play from the exact same game). Where they do collide, the chart's value wins and pbp's copy is
# dropped (see _attach_pbp_columns) - not kept as name_x/name_y.
#
# "game_id" is deliberately NOT promoted: pbp's game_id is the join key this whole match is built on,
# and chart's own raw NGS id stays available, unambiguously, as chart_game_id. "touchdown"/
# "fumble_lost" are also deliberately left prefixed: pbp's versions are a whole-play flag (was there
# a touchdown/fumble on this play, by anyone), not specifically this target's/this carry's - usually
# equal, but collapsing them would throw away a real, if usually-agreeing, cross-check between the
# chart and the box score.
_PROMOTE_COLS = ["name", "team", "position", "season", "season_type", "week", "pass_type",
                "route_type", "gain_class", "located", "start_ok", "handoff_ok",
                "x_coord", "y_coord", "catch_y", "path_x", "path_y", "path_segment"]


def _require_nflreadpy():
    try:
        import nflreadpy as nfl
    except ImportError as e:
        raise ImportError(
            "load_pbp_with_ftn and match_to_pbp need nflreadpy: pip install \"next-gen-scrapy[pbp]\""
        ) from e
    return nfl


def load_pbp_with_ftn(seasons):
    """
    Convenience loader: play-by-play left-joined with FTN charting for the given season(s), as a
    pandas DataFrame. FTN charting doesn't cover every play (roughly 94% of a season, in practice -
    it never charts non-play admin rows, and it lags real time by a few days for the newest games),
    so expect nulls in the FTN columns rather than a row being dropped.

    FTN's own `season`/`week` columns are dropped before merging - identical to pbp's own on every
    matched row (verified against a real season), so keeping both would only produce pandas'
    auto-suffixed season_x/season_y and week_x/week_y instead of one column apiece.
    """
    nfl = _require_nflreadpy()
    seasons = [seasons] if isinstance(seasons, (int, str)) else list(seasons)
    pbp = nfl.load_pbp(seasons=seasons).to_pandas()
    ftn = nfl.load_ftn_charting(seasons=seasons).to_pandas()
    ftn["nflverse_play_id"] = ftn["nflverse_play_id"].astype("float64")
    ftn = ftn.drop(columns=["season", "week"])
    return pbp.merge(ftn, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"],
                      how="left")


def _game_id_crosswalk(pbp_df):
    """An NGS chart's game_id (numeric, e.g. '2026091302') equals pbp's old_game_id."""
    return pbp_df[["game_id", "old_game_id"]].drop_duplicates().set_index("old_game_id")["game_id"]


def _esb_to_gsis(esb_ids):
    """esb_id (the chart's player id) -> gsis_id (pbp's passer/rusher/receiver_player_id)."""
    nfl = _require_nflreadpy()
    players = nfl.load_players().to_pandas().dropna(subset=["esb_id"]).drop_duplicates(subset=["esb_id"])
    return players.set_index("esb_id")["gsis_id"].reindex(esb_ids)


def _hungarian_match(pbp_sub, chart_sub, pbp_col, chart_col):
    """
    Optimal 1:1 assignment minimizing |pbp_col - chart_col|. Handles unequal set sizes - excess
    rows on the larger side are simply left unmatched. Rows with a NaN match-value (a handful of
    pbp plays have null air_yards) are dropped first - they can't be matched by yardage, and
    linear_sum_assignment can't accept NaN in the cost matrix anyway.
    """
    pbp_sub = pbp_sub[pbp_sub[pbp_col].notna()]
    chart_sub = chart_sub[chart_sub[chart_col].notna()]
    if len(pbp_sub) == 0 or len(chart_sub) == 0:
        return []
    pbp_sub = pbp_sub.reset_index()
    chart_sub = chart_sub.reset_index()
    cost = np.abs(pbp_sub[pbp_col].to_numpy(dtype=float)[:, None]
                  - chart_sub[chart_col].to_numpy(dtype=float)[None, :])
    rows, cols = linear_sum_assignment(cost)
    return [(pbp_sub.loc[r, "index"], chart_sub.loc[c, "index"], cost[r, c]) for r, c in zip(rows, cols)]


def _buckets(kind, grp_pbp, grp_chart):
    """(pbp subset, chart subset, pbp column, chart column) per outcome bucket, both measured from
    the line of scrimmage - so a route or carry line drawn starting behind the LOS needs no
    adjustment, only the chart's absolute y_coord/catch_y matters, never the line's own start
    point."""
    if kind == "pass":
        return {
            "TOUCHDOWN": (grp_pbp[grp_pbp["pass_touchdown"] == 1],
                          grp_chart[grp_chart["pass_type"] == "TOUCHDOWN"], "air_yards", "y_coord"),
            "INTERCEPTION": (grp_pbp[grp_pbp["interception"] == 1],
                              grp_chart[grp_chart["pass_type"] == "INTERCEPTION"], "air_yards", "y_coord"),
            "COMPLETE": (grp_pbp[(grp_pbp["complete_pass"] == 1) & (grp_pbp["pass_touchdown"] == 0)],
                         grp_chart[grp_chart["pass_type"] == "COMPLETE"], "air_yards", "y_coord"),
            "INCOMPLETE": (grp_pbp[grp_pbp["incomplete_pass"] == 1],
                           grp_chart[grp_chart["pass_type"] == "INCOMPLETE"], "air_yards", "y_coord"),
        }
    if kind == "carry":
        return {
            "TOUCHDOWN": (grp_pbp[grp_pbp["rush_touchdown"] == 1],
                          grp_chart[grp_chart["touchdown"] == True], "yards_gained", "y_coord"),
            "OTHER": (grp_pbp[grp_pbp["rush_touchdown"] != 1],
                      grp_chart[grp_chart["touchdown"] != True], "yards_gained", "y_coord"),
        }
    return {  # route
        "TOUCHDOWN": (grp_pbp[grp_pbp["pass_touchdown"] == 1],
                      grp_chart[grp_chart["touchdown"] == True], "air_yards", "catch_y"),
        "COMPLETE": (grp_pbp[(grp_pbp["complete_pass"] == 1) & (grp_pbp["pass_touchdown"] == 0)],
                     grp_chart[(grp_chart["route_type"] == "COMPLETE") & (grp_chart["touchdown"] != True)],
                     "air_yards", "catch_y"),
        "INCOMPLETE": (grp_pbp[grp_pbp["complete_pass"] == 0],
                       grp_chart[grp_chart["route_type"] == "INCOMPLETE"], "air_yards", "catch_y"),
    }


def match_to_pbp(chart_df, pbp_df, kind):
    """
    Match a chart DataFrame (from get_passes / get_carries / get_routes) to the pbp rows for the
    same plays. `pbp_df` can be plain play-by-play or the FTN-enriched table from
    load_pbp_with_ftn - either way it just needs nflverse's standard pbp columns.

      1. joins the game exactly: chart['game_id'] (NGS's numeric id) == pbp_df['old_game_id']
      2. resolves the player exactly via an esb_id -> gsis_id crosswalk (nfl.load_players())
      3. within each (game, player), buckets rows by outcome - touchdown / interception /
         complete / incomplete for passes, touchdown / other for carries - and solves an optimal
         assignment (Hungarian algorithm) minimizing |pbp yardage - chart y_coord|.

    kind: 'pass' | 'carry' | 'route'

    Returns one row per matched play, with every column from the chart data, from pbp_df, and (if
    pbp_df is the table from load_pbp_with_ftn) from FTN - minus duplicate features:

      - kind, bucket, residual (yards) - this function's own columns.
      - game_id, play_id - pbp's play key (also the join used to pull in the rest of pbp_df/FTN
        below), plus every other pbp/FTN column from pbp_df (EPA, WP, FTN's is_play_action /
        is_screen_pass / n_pass_rushers / ..., whatever pbp_df carries).
      - every chart column. Most are promoted to their natural, unprefixed name - name, team,
        position, season, season_type, week, pass_type/route_type/gain_class, located/start_ok/
        handoff_ok, x_coord/y_coord/catch_y, path_x/path_y/path_segment (the full traced line, not
        just its endpoint - see reshape.to_paths for exactly which columns a carry/route chart
        reduces to). A few of these share a name with a pbp column (only "name" differs in value -
        pbp's own is nflverse's short form, e.g. "J.Allen", vs. the chart's "Josh Allen"; season/
        week are identical on every matched row by construction) - there the chart's value wins and
        pbp's copy is dropped, rather than silently suffixing both _x/_y.
      - chart_game_id, chart_touchdown, chart_fumble_lost stay prefixed rather than promoted:
        pbp's own game_id is the join key (chart's raw NGS numeric id is a different id system, kept
        separately rather than overwriting it); pbp's touchdown/fumble_lost are a whole-play flag
        (was there a TD/fumble on this play, by anyone), not specifically this target's/this
        carry's - usually equal to the chart's, but collapsing them would throw away a real, if
        usually-agreeing, cross-check between the chart and the box score.

    Chart columns are embedded directly rather than returned as an index to look up later, since for
    carry/route the match happens against a one-row-per-carry/-route reduction of chart_df (see
    reshape.to_paths), not chart_df itself - an index into that reduction can't be used to slice
    chart_df.

    Sort by residual - large residuals (roughly >3 yards) are the ones worth treating with
    suspicion.

    Caveats:
      - pass: TOUCHDOWN / INTERCEPTION / COMPLETE are guaranteed 1:1 with pbp (every one is drawn
        on the chart). INCOMPLETE is not - NGS charts don't draw every incompletion - so some
        incomplete pbp attempts will simply go unmatched.
      - carry: every carry is guaranteed drawn, so counts should line up exactly with the
        rusher's run plays for that game.
      - route: the least reliable of the three (in practice, roughly a 1-in-10 rate of residuals
        over 5 yards vs. roughly 1-in-25 for pass/carry). NGS's own target/reception counts for a
        route chart don't always equal nflverse's target/reception counts for the same
        player-game, so bucket sizes can mismatch and some matches will be low-confidence.
    """
    assert kind in ("pass", "carry", "route")
    id_col = CHART_ID_COL[kind]
    play_type = CHART_PLAY_TYPE[kind]

    chart_reduced = to_paths(chart_df, kind)      # one row per pass/carry/route, every original column kept
    gmap = _game_id_crosswalk(pbp_df)
    chart_reduced["game_id_nflverse"] = chart_reduced["game_id"].astype(str).map(gmap)
    chart_reduced["gsis_id"] = chart_reduced["esb_id"].map(_esb_to_gsis(chart_reduced["esb_id"].unique()))

    pbp_typed = pbp_df[pbp_df["play_type"] == play_type]

    matches = []
    for (game_id, gsis_id), grp_chart in chart_reduced.groupby(["game_id_nflverse", "gsis_id"]):
        if pd.isna(game_id) or pd.isna(gsis_id):
            continue
        grp_pbp = pbp_typed[(pbp_typed["game_id"] == game_id) & (pbp_typed[id_col] == gsis_id)]

        for bucket, (p_sub, c_sub, pcol, ccol) in _buckets(kind, grp_pbp, grp_chart).items():
            for pbp_idx, chart_idx, resid in _hungarian_match(p_sub, c_sub, pcol, ccol):
                chart_row = chart_reduced.loc[chart_idx].to_dict()
                matches.append(dict(
                    kind=kind, bucket=bucket, game_id=game_id,
                    play_id=pbp_df.loc[pbp_idx, "play_id"], residual=round(float(resid), 2),
                    **{f"chart_{k}": v for k, v in chart_row.items() if k not in ("game_id_nflverse", "gsis_id")}
                ))

    matched = pd.DataFrame(matches)
    if not len(matched):
        return matched
    matched = matched.rename(columns={f"chart_{c}": c for c in _PROMOTE_COLS if f"chart_{c}" in matched.columns})
    return _attach_pbp_columns(matched, pbp_df)


def _attach_pbp_columns(matched, pbp_df):
    """
    Bring in every column of pbp_df (already including FTN's, if pbp_df is load_pbp_with_ftn's
    output) onto the matched rows, joining on (game_id, play_id) - the exact keys match_to_pbp
    already pulled from pbp_df for every matched row, so this is an exact join, not a fuzzy one,
    and every row is guaranteed to find its match.

    Most chart-derived columns on `matched` were promoted to their natural name by match_to_pbp
    (_PROMOTE_COLS) specifically because they're expected to collide with a pbp_df column of the
    same name (name, season, season_type, week) - matched's copy (the chart's value) wins in every
    case, and pbp_df's is dropped, rather than letting pandas silently rename both to _x/_y. The
    remaining chart_-prefixed columns (chart_game_id, chart_touchdown, chart_fumble_lost, ...)
    aren't expected to collide, but this drops pbp_df's copy of those too if one somehow exists.
    """
    dupes = [c for c in pbp_df.columns if c in matched.columns and c not in ("game_id", "play_id")]
    pbp_slim = pbp_df.drop(columns=dupes) if dupes else pbp_df
    return matched.merge(pbp_slim, on=["game_id", "play_id"], how="left", validate="m:1")
