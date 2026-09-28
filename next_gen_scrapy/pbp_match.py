"""
Match a chart DataFrame (from get_passes / get_carries / get_routes) to the play-by-play rows
for the same plays, optionally enriched with FTN charting and NGS participation data.

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

# NGS participation columns worth carrying onto every play. `route` is renamed: it's the TARGETED
# receiver's route, not necessarily the charted player's.
_PARTICIPATION_COLS = {
    "ngs_air_yards": "ngs_air_yards", "time_to_throw": "time_to_throw", "was_pressure": "was_pressure",
    "route": "target_route", "defense_man_zone_type": "defense_man_zone_type",
    "defense_coverage_type": "defense_coverage_type", "offense_formation": "offense_formation",
    "offense_personnel": "offense_personnel", "defense_personnel": "defense_personnel",
}


def _require_nflreadpy():
    try:
        import nflreadpy as nfl
    except ImportError as e:
        raise ImportError(
            "load_pbp_with_ftn and match_to_pbp need nflreadpy: pip install \"next-gen-scrapy[pbp]\""
        ) from e
    return nfl


def load_pbp_with_ftn(seasons, participation=True):
    """
    Convenience loader: play-by-play left-joined with FTN charting (and, by default, NGS
    participation data) for the given season(s), as a pandas DataFrame. FTN charting doesn't cover
    every play (roughly 94% of a season, in practice - it never charts non-play admin rows, and it
    lags real time by a few days for the newest games), so expect nulls in the FTN columns rather
    than a row being dropped. FTN charting itself only covers 2022 onward - seasons before that get
    pbp with every FTN column present but null, rather than an error, so a multi-season call spanning
    the cutoff still returns one consistent schema.

    participation=True adds NGS's own per-play fields (see _PARTICIPATION_COLS): coverage type,
    formation, personnel, pressure, time to throw, the targeted receiver's route, and - for
    2018-2022 - `ngs_air_yards`, NGS's measured air yards. match_to_pbp prefers ngs_air_yards over
    pbp's scorer-spotted air_yards when it's there: it agrees with the chart dots ~2.5x more tightly.

    FTN's own `season`/`week` columns are dropped before merging - identical to pbp's own on every
    matched row (verified against a real season), so keeping both would only produce pandas'
    auto-suffixed season_x/season_y and week_x/week_y instead of one column apiece.
    """
    nfl = _require_nflreadpy()
    seasons = [seasons] if isinstance(seasons, (int, str)) else list(seasons)
    pbp = nfl.load_pbp(seasons=seasons).to_pandas()

    ftn_seasons = [s for s in seasons if int(s) >= 2022]
    if ftn_seasons:
        ftn = nfl.load_ftn_charting(seasons=ftn_seasons).to_pandas()
        ftn["nflverse_play_id"] = ftn["nflverse_play_id"].astype("float64")
        ftn = ftn.drop(columns=["season", "week"])
        pbp = pbp.merge(ftn, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"],
                        how="left")

    if participation:
        part = nfl.load_participation(seasons=seasons).to_pandas()
        keep = {k: v for k, v in _PARTICIPATION_COLS.items() if k in part.columns}
        part = part[["nflverse_game_id", "play_id", *keep]].rename(columns=keep)
        part = part.rename(columns={"nflverse_game_id": "_part_game_id", "play_id": "_part_play_id"})
        part["_part_play_id"] = part["_part_play_id"].astype("float64")
        part = part.drop_duplicates(subset=["_part_game_id", "_part_play_id"])
        pbp = pbp.merge(part, left_on=["game_id", "play_id"], right_on=["_part_game_id", "_part_play_id"],
                        how="left").drop(columns=["_part_game_id", "_part_play_id"])
    return pbp


def _game_id_crosswalk(pbp_df):
    """An NGS chart's game_id (numeric, e.g. '2026091302') equals pbp's old_game_id."""
    return pbp_df[["game_id", "old_game_id"]].drop_duplicates().set_index("old_game_id")["game_id"]


def _esb_to_gsis(esb_ids):
    """esb_id (the chart's player id) -> gsis_id (pbp's passer/rusher/receiver_player_id)."""
    nfl = _require_nflreadpy()
    players = nfl.load_players().to_pandas().dropna(subset=["esb_id"]).drop_duplicates(subset=["esb_id"])
    return players.set_index("esb_id")["gsis_id"].reindex(esb_ids)


# ---------------------------------------------------------------------------------------------------
# Matching model
#
# Within one (game, player, outcome) group, every chart entity is scored against every pbp play by a
# negative log-likelihood built from the evidence the chart and pbp share, and the optimal 1:1
# assignment (Hungarian algorithm) minimises the total:
#
#   pass   depth  chart dot y      vs  ngs_air_yards (2018-22) or pbp air_yards
#          side   chart dot x      vs  pbp pass_location (left / middle / right)
#   route  depth  catch point y    vs  ngs_air_yards or pbp air_yards
#          side   catch point x    vs  pass_location
#          YAC    after-catch run  vs  pbp yards_after_catch            (completions)
#   carry  gain   end point y      vs  pbp yards_gained
#          gap    where the run crosses the LOS, relative to the handoff
#                                  vs  pbp run_location + run_gap (guard ~2 yds out, tackle ~4-5, end ~10)
#          depth  handoff depth    vs  participation offense_formation (shotgun ~-4.4 yds, under center ~-6.4)
#          hash   start x          vs  FTN starting_hash                  (2022+)
#   pass   (optional) link   chart dot vs the matched route chart's catch point for the same play,
#                            when the targeted receiver has a route chart (pass route_matches=...)
#
# Every term's offset/spread is self-calibrated on the groups that can only match one way (one chart
# entity, one pbp play), falling back to values measured on 2022 when there are too few of those.
# Each term is a robust mixture (90% Laplace/Gaussian + 10% uniform) so one wild measurement can't
# dominate, and a missing pbp field costs the same as "no information" rather than zero.
# ---------------------------------------------------------------------------------------------------

_DEFAULTS = {
    "pass": {"ngs": (0.72, 0.58), "air": (0.74, 1.41),
             "side": {"left": (-17.3, 5.8), "middle": (-1.0, 6.2), "right": (17.0, 6.3)}},
    "route": {"ngs": (1.44, 1.18), "air": (2.04, 2.60), "yac": (0.0, 0.8),
              "side": {"left": (-16.3, 8.5), "middle": (0.6, 7.6), "right": (18.7, 6.4)}},
    "carry": {"gain": (-0.45, 0.87),
              # crossing point minus handoff x, keyed "location/gap" (measured on unambiguous 2022 carries)
              "gap": {"left/end": (-9.8, 5.4), "left/tackle": (-5.0, 3.1), "left/guard": (-2.4, 2.6),
                      "middle/": (0.0, 2.9), "left/": (-5.5, 5.5), "right/": (5.5, 5.5),
                      "right/guard": (1.7, 2.0), "right/tackle": (4.1, 4.6), "right/end": (10.8, 5.4)},
              "form": {"SHOTGUN": (-4.4, 1.8), "SINGLEBACK": (-6.4, 1.9), "I_FORM": (-6.4, 1.4),
                       "PISTOL": (-6.3, 2.5), "JUMBO": (-5.2, 1.5), "WILDCAT": (-4.6, 1.5)},
              "hash": {"L": (-2.1, 2.5), "M": (-0.6, 2.4), "R": (1.9, 3.3)}},
}
_USE_ROUTE_X = True      # switch for A/B evaluation of the route-type term
_USE_OOB = True          # switch for A/B evaluation of the out-of-bounds / fumble terms

# P(chart line ends within 2.7 yds of a sideline | pbp out_of_bounds), measured on confident matches
_SIDELINE_YD = 24.0
_P_END_AT_SIDELINE = {"carry": {1: 0.98, 0: 0.10}, "route": {1: 0.89, 0: 0.17}}
_P_FUMBLE_AGREE = 0.97
# carry chart colour (red / yellow / green) vs pbp yards_gained: LOSS < 0, SHORT 0-5, LONG 6+ -
# agrees 97% of the time on confidently matched carries
_P_GAIN_CLASS_AGREE = 0.97
_TWO_PASS = True         # recalibrate on the first pass's confident matches, then match again
# Temperature applied to the costs when turning them into match probabilities (not to the assignment
# itself): < 1 sharpens, > 1 softens. Fitted per kind so the reported probabilities agree with
# held-out evidence the model doesn't use (see the notes on calibration in the README).
_TEMP = {"pass": 1.0, "route": 1.0, "carry": 0.4}   # carry: fitted on FTN starting hash, 2022-25
_LINK_SD = 2.0           # pass dot vs route catch point for the same throw (two separately calibrated charts)

# Where a throw lands across the field, given pbp's side AND NGS's route type for the target
# (participation `route`): (median x, spread) in yards, measured on ~52k confidently matched passes,
# 2018-25. Sharper than side alone: an out / go / corner lands near the sideline, a slant / in / post
# inside the numbers. Groups with < 80 plays fall back to side only.
_ROUTE_X = {
    'left/ANGLE': (-10.5, 3.9), 'left/CORNER': (-20.6, 4.1), 'left/CROSS': (-13.6, 5.7), 'left/DEEP OUT': (-22.0, 3.1),
    'left/FLAT': (-16.1, 4.7), 'left/GO': (-23.4, 2.7), 'left/HITCH': (-16.0, 7.6), 'left/HITCH/CURL': (-14.9, 7.2),
    'left/IN': (-10.6, 3.7), 'left/IN/DIG': (-11.5, 4.3), 'left/OUT': (-20.6, 4.0), 'left/POST': (-10.8, 4.1),
    'left/QUICK OUT': (-19.3, 4.3), 'left/SCREEN': (-12.0, 3.8), 'left/SHALLOW CROSS/DRAG': (-12.4, 4.4),
    'left/SLANT': (-11.1, 2.9), 'left/SWING': (-15.0, 4.8), 'left/WHEEL': (-22.0, 3.3),
    'middle/ANGLE': (0.3, 4.4), 'middle/CROSS': (-0.9, 6.6), 'middle/FLAT': (0.9, 6.8), 'middle/GO': (1.0, 7.0),
    'middle/HITCH': (0.2, 4.9), 'middle/HITCH/CURL': (0.1, 5.8), 'middle/IN': (-0.4, 5.9), 'middle/IN/DIG': (0.2, 6.2),
    'middle/OUT': (1.0, 6.6), 'middle/POST': (-0.2, 6.4), 'middle/SCREEN': (-0.1, 5.3),
    'middle/SHALLOW CROSS/DRAG': (-0.1, 5.1), 'middle/SLANT': (-0.4, 7.0),
    'right/ANGLE': (10.9, 3.7), 'right/CORNER': (20.5, 4.0), 'right/CROSS': (13.4, 5.7), 'right/DEEP OUT': (22.0, 3.3),
    'right/FLAT': (15.8, 4.9), 'right/GO': (23.2, 2.9), 'right/HITCH': (15.6, 7.9), 'right/HITCH/CURL': (14.2, 7.3),
    'right/IN': (10.8, 3.8), 'right/IN/DIG': (13.3, 5.9), 'right/OUT': (20.4, 4.3), 'right/POST': (10.7, 4.1),
    'right/QUICK OUT': (18.6, 4.6), 'right/SCREEN': (11.6, 3.5), 'right/SHALLOW CROSS/DRAG': (12.7, 4.7),
    'right/SLANT': (11.0, 2.8), 'right/SWING': (14.7, 5.4), 'right/WHEEL': (22.1, 2.8),
}
_MIN_CALIB = 40          # forced pairs needed before trusting a self-calibrated term
_FLOOR = 0.10            # weight of the uniform "outlier" component in every term
_RANGE = 60.0            # yards the uniform component spreads over
_LOCATION_OK_YDS = 3.0   # a play's location counts as right if its dot is within this many yards of the true one


def _nll_laplace(d, b):
    b = np.maximum(b, 0.25)
    return -np.log((1 - _FLOOR) * np.exp(-np.abs(d) / b) / (2 * b) + _FLOOR / _RANGE)


def _nll_gauss(d, s):
    s = np.maximum(s, 1.0)
    return -np.log((1 - _FLOOR) * np.exp(-0.5 * (d / s) ** 2) / (s * np.sqrt(2 * np.pi)) + _FLOOR / _RANGE)


_NLL_MISSING = -np.log(1.0 / _RANGE)    # "this pbp field tells us nothing" - flat over the field


def _robust(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    med = float(np.median(x))
    return med, float(np.median(np.abs(x - med)) / np.log(2))      # Laplace scale from the MAD


def _catch_index(seg):
    pre = [i for i, s in enumerate(seg) if s != "after_catch"]
    return pre[-1] if pre else len(seg) - 1


def _chart_features(chart, kind):
    """Per chart entity: the measurements the model scores, plus the point used for location checks."""
    f = pd.DataFrame(index=chart.index)
    if kind == "pass":
        f["depth"], f["x"] = chart["y_coord"], chart["x_coord"]
        f["loc_x"], f["loc_y"] = chart["x_coord"], chart["y_coord"]
    elif kind == "route":
        ci = chart["path_segment"].map(_catch_index)
        f["x"] = [px[i] for px, i in zip(chart["path_x"], ci)]
        f["depth"] = [py[i] for py, i in zip(chart["path_y"], ci)]
        f["yac"] = [py[-1] - py[i] for py, i in zip(chart["path_y"], ci)]
        f["end_sideline"] = [abs(px[-1]) >= _SIDELINE_YD for px in chart["path_x"]]
        f["loc_x"], f["loc_y"] = f["x"], f["depth"]
    else:
        f["gain"] = chart["y_coord"]
        f["x0"] = [px[0] for px in chart["path_x"]]
        f["y0"] = [py[0] for py in chart["path_y"]]
        f["cross"] = [_crossing_x(px, py) - px[0] for px, py in zip(chart["path_x"], chart["path_y"])]
        f["loc_x"] = [px[-1] for px in chart["path_x"]]
        f["loc_y"] = chart["y_coord"]
        f["end_sideline"] = [abs(px[-1]) >= _SIDELINE_YD for px in chart["path_x"]]
        f["fumble"] = chart["fumble_lost"].astype(str).str.lower().eq("true") if "fumble_lost" in chart else False
        f["gclass"] = chart["gain_class"] if "gain_class" in chart else None
    return f


def _crossing_x(px, py):
    """x where a carry first crosses the line of scrimmage (y = 0); NaN if it never does."""
    py = np.asarray(py, float)
    k = np.flatnonzero((py[:-1] < 0) & (py[1:] >= 0))
    if not len(k):
        return np.nan
    k = k[0]
    t = -py[k] / (py[k + 1] - py[k])
    return float(px[k] + t * (px[k + 1] - px[k]))


def _pbp_features(pbp, kind):
    f = pd.DataFrame(index=pbp.index)
    if kind in ("pass", "route"):
        ngs = pbp["ngs_air_yards"] if "ngs_air_yards" in pbp else pd.Series(np.nan, index=pbp.index)
        f["ngs"], f["air"] = ngs.astype(float), pbp["air_yards"].astype(float)
        f["side"] = pbp["pass_location"]
        if "target_route" in pbp:
            key = pbp["pass_location"].fillna("") + "/" + pbp["target_route"].fillna("")
            f["side_route"] = key.where(key.isin(list(_ROUTE_X)), None)
        if kind == "route":
            f["yac"] = pbp["yards_after_catch"].astype(float)
            f["oob"] = pbp["out_of_bounds"].astype(float) if "out_of_bounds" in pbp else np.nan
    else:
        f["gain"] = pbp["yards_gained"].astype(float)
        loc = pbp["run_location"].fillna("")
        gap = pbp["run_gap"].fillna("") if "run_gap" in pbp else ""
        f["gap"] = np.where(loc == "", None, loc + "/" + np.where(loc == "middle", "", gap))
        f["form"] = pbp["offense_formation"] if "offense_formation" in pbp else None
        f["hash"] = pbp["starting_hash"] if "starting_hash" in pbp else None
        f["oob"] = pbp["out_of_bounds"].astype(float) if "out_of_bounds" in pbp else np.nan
        f["fumble"] = pbp["fumble_lost"].astype(float) if "fumble_lost" in pbp else np.nan
        y = pbp["yards_gained"].astype(float)
        f["gclass"] = np.where(y.isna(), None, np.where(y < 0, "LOSS", np.where(y <= 5, "SHORT", "LONG")))
    return f


def _flag_cost(chart_flag, pbp_flag, p_given):
    """(n_chart, n_pbp) cost of a yes/no chart observation given a yes/no pbp flag:
    p_given = {1: P(chart yes | pbp yes), 0: P(chart yes | pbp no)}. Unknown pbp flag costs nothing."""
    cf = np.asarray(chart_flag, bool)[:, None]
    pf = np.asarray(pbp_flag, float)[None, :]
    p_yes = np.where(pf == 1, p_given[1], p_given[0])
    cost = -np.log(np.where(cf, p_yes, 1 - p_yes))
    return np.where(np.isfinite(pf), cost, 0.0)


def _buckets(kind, grp_pbp, grp_chart):
    """(pbp subset, chart subset) per outcome bucket - an exact partition both sides share."""
    if kind == "pass":
        return {
            "TOUCHDOWN": (grp_pbp[grp_pbp["pass_touchdown"] == 1], grp_chart[grp_chart["pass_type"] == "TOUCHDOWN"]),
            "INTERCEPTION": (grp_pbp[grp_pbp["interception"] == 1], grp_chart[grp_chart["pass_type"] == "INTERCEPTION"]),
            "COMPLETE": (grp_pbp[(grp_pbp["complete_pass"] == 1) & (grp_pbp["pass_touchdown"] == 0)],
                         grp_chart[grp_chart["pass_type"] == "COMPLETE"]),
            "INCOMPLETE": (grp_pbp[grp_pbp["incomplete_pass"] == 1], grp_chart[grp_chart["pass_type"] == "INCOMPLETE"]),
        }
    if kind == "carry":
        return {
            "TOUCHDOWN": (grp_pbp[grp_pbp["rush_touchdown"] == 1], grp_chart[grp_chart["touchdown"] == True]),
            "OTHER": (grp_pbp[grp_pbp["rush_touchdown"] != 1], grp_chart[grp_chart["touchdown"] != True]),
        }
    return {  # route
        "TOUCHDOWN": (grp_pbp[grp_pbp["pass_touchdown"] == 1], grp_chart[grp_chart["touchdown"] == True]),
        "COMPLETE": (grp_pbp[(grp_pbp["complete_pass"] == 1) & (grp_pbp["pass_touchdown"] == 0)],
                     grp_chart[(grp_chart["route_type"] == "COMPLETE") & (grp_chart["touchdown"] != True)]),
        "INCOMPLETE": (grp_pbp[grp_pbp["complete_pass"] == 0], grp_chart[grp_chart["route_type"] == "INCOMPLETE"]),
    }


def _calibrate(kind, pairs):
    """Offsets/spreads for every term, from forced 1:1 pairs [(chart feature row, pbp feature row), ...]."""
    p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DEFAULTS[kind].items()}
    if not pairs:
        return p
    c = pd.DataFrame([a for a, _ in pairs]).reset_index(drop=True)
    b = pd.DataFrame([bb for _, bb in pairs]).reset_index(drop=True)

    def fit(name, resid):
        # forced pairs are mostly touchdowns (e.g. YAC 0 on both sides), so never let a term
        # calibrate tighter than half of what was measured on a broad sample
        resid = np.asarray(resid, float)
        if np.isfinite(resid).sum() >= _MIN_CALIB:
            med, scale = _robust(resid)
            p[name] = (med, max(scale, 0.5 * _DEFAULTS[kind][name][1]))

    def fit_side(name, values, labels):
        for lab, (mu0, sd0) in list(p[name].items()):
            v = np.asarray(values, float)[np.asarray(labels) == lab]
            v = v[np.isfinite(v)]
            if len(v) >= _MIN_CALIB:
                med = float(np.median(v))
                p[name][lab] = (med, max(1.5 * float(np.median(np.abs(v - med))), 2.0))

    if kind in ("pass", "route"):
        fit("ngs", c["depth"] - b["ngs"])
        fit("air", c["depth"] - b["air"])
        fit_side("side", c["x"], b["side"])
        if kind == "route":
            comp = b["yac"].notna()
            fit("yac", (c["yac"] - b["yac"])[comp])
    else:
        fit("gain", c["gain"] - b["gain"])
        fit_side("gap", c["cross"], b["gap"])
        fit_side("form", c["y0"], b["form"])
        fit_side("hash", c["x0"], b["hash"])
    return p


def _side_cost(x, labels, table):
    """x: (n_chart,), labels: (n_pbp,) -> (n_chart, n_pbp) cost of each chart x under each pbp side label."""
    out = np.full((len(x), len(labels)), _NLL_MISSING)
    for j, lab in enumerate(labels):
        if isinstance(lab, str) and lab in table:
            mu, sd = table[lab]
            out[:, j] = _nll_gauss(x - mu, sd)
    return out


def _cost_matrix(kind, cf, pf, params, bucket):
    """(n_chart, n_pbp) total negative log-likelihood."""
    if kind in ("pass", "route"):
        depth = cf["depth"].to_numpy(float)[:, None]
        ngs, air = pf["ngs"].to_numpy(float), pf["air"].to_numpy(float)
        (bn, sn), (ba, sa) = params["ngs"], params["air"]
        use_ngs = np.isfinite(ngs)
        d_cost = np.where(use_ngs[None, :], _nll_laplace(depth - bn - np.nan_to_num(ngs)[None, :], sn),
                          _nll_laplace(depth - ba - np.nan_to_num(air)[None, :], sa))
        d_cost = np.where((~use_ngs & ~np.isfinite(air))[None, :], _NLL_MISSING, d_cost)
        side = _side_cost(cf["x"].to_numpy(float), pf["side"].to_numpy(object), params["side"])
        if "side_route" in pf and _USE_ROUTE_X:
            sharper = _side_cost(cf["x"].to_numpy(float), pf["side_route"].to_numpy(object), _ROUTE_X)
            has = np.array([isinstance(k, str) for k in pf["side_route"]])
            side = np.where(has[None, :], sharper, side)
        C = d_cost + side
        if kind == "route" and bucket != "INCOMPLETE":
            yac = pf["yac"].to_numpy(float)
            by, sy = params["yac"]
            y_cost = _nll_laplace(cf["yac"].to_numpy(float)[:, None] - by - np.nan_to_num(yac)[None, :], sy)
            C = C + np.where(np.isfinite(yac)[None, :], y_cost, _NLL_MISSING)
            if _USE_OOB:
                C = C + _flag_cost(cf["end_sideline"], pf["oob"], _P_END_AT_SIDELINE["route"])
    else:
        gain = pf["gain"].to_numpy(float)
        bg, sg = params["gain"]
        g_cost = _nll_laplace(cf["gain"].to_numpy(float)[:, None] - bg - np.nan_to_num(gain)[None, :], sg)
        C = np.where(np.isfinite(gain)[None, :], g_cost, _NLL_MISSING)
        C = C + _side_cost(cf["cross"].to_numpy(float), pf["gap"].to_numpy(object), params["gap"])
        C = C + _side_cost(cf["y0"].to_numpy(float), pf["form"].to_numpy(object), params["form"])
        C = C + _side_cost(cf["x0"].to_numpy(float), pf["hash"].to_numpy(object), params["hash"])
        if _USE_OOB:
            C = C + _flag_cost(cf["end_sideline"], pf["oob"], _P_END_AT_SIDELINE["carry"])
            C = C + _flag_cost(cf["fumble"], pf["fumble"], {1: _P_FUMBLE_AGREE, 0: 1 - _P_FUMBLE_AGREE})
        if "gclass" in cf and "gclass" in pf:
            same = cf["gclass"].to_numpy(object)[:, None] == pf["gclass"].to_numpy(object)[None, :]
            known = pd.notna(pf["gclass"].to_numpy(object))[None, :] & pd.notna(cf["gclass"].to_numpy(object))[:, None]
            C = C + np.where(known, np.where(same, -np.log(_P_GAIN_CLASS_AGREE), -np.log((1 - _P_GAIN_CLASS_AGREE) / 2)), 0.0)
    if kind == "pass" and "link_x" in pf:
        # the receiver's route chart puts this same throw at (link_x, link_y) - a second, independent
        # measurement of where the ball went, for the plays whose target has a route chart
        lx, ly = pf["link_x"].to_numpy(float), pf["link_y"].to_numpy(float)
        d2 = (cf["x"].to_numpy(float)[:, None] - lx[None, :]) ** 2 + (cf["depth"].to_numpy(float)[:, None] - ly[None, :]) ** 2
        dens = (1 - _FLOOR) * np.exp(-0.5 * d2 / _LINK_SD ** 2) / (2 * np.pi * _LINK_SD ** 2) + _FLOOR / _RANGE ** 2
        C = C + np.where(np.isfinite(lx)[None, :], -np.log(dens), np.log(_RANGE ** 2))
    # a chart entity with no usable measurement at all can still be placed, but gains nothing
    return np.nan_to_num(C, nan=2 * _NLL_MISSING, posinf=2 * _NLL_MISSING)


def _assignment_marginals(C, iters=200):
    """
    Approximate P(chart entity k belongs to play j) under the one-to-one constraint, by Sinkhorn
    balancing of exp(-C). Unequal sides are padded with "unmatched" slots at the group's median
    cost (NGS doesn't draw every incompletion, so a play may have no chart entity at all).
    Returns the (n_chart, n_pbp) block of the balanced matrix.
    """
    n, m = C.shape
    size = max(n, m)
    pad = np.full((size, size), float(np.median(C)))
    pad[:n, :m] = C
    K = np.exp(-(pad - pad.min()))
    K = np.maximum(K, 1e-300)
    for _ in range(iters):
        K /= K.sum(axis=1, keepdims=True)
        K /= K.sum(axis=0, keepdims=True)
    return K[:n, :m]


def match_to_pbp(chart_df, pbp_df, kind, route_matches=None, return_params=False):
    """
    Match a chart DataFrame (from get_passes / get_carries / get_routes) to the pbp rows for the
    same plays. `pbp_df` can be plain play-by-play or the enriched table from load_pbp_with_ftn -
    it needs nflverse's standard pbp columns, and uses FTN `starting_hash` and participation
    `ngs_air_yards` when present.

      1. joins the game exactly: chart['game_id'] (NGS's numeric id) == pbp_df['old_game_id']
      2. resolves the player exactly via an esb_id -> gsis_id crosswalk (nfl.load_players())
      3. within each (game, player), splits rows by outcome - touchdown / interception / complete /
         incomplete for passes and routes, touchdown / other for carries - exactly, on both sides
      4. within each outcome, scores every chart entity against every candidate play using all the
         evidence both sides carry (see the model notes above: depth, side of the field, yards after
         catch, starting hash), self-calibrated on this data, and takes the optimal 1:1 assignment.

    kind: 'pass' | 'carry' | 'route'

    route_matches (pass only, optional): the output of match_to_pbp(..., 'route') for the same
    games. A targeted receiver's route chart shows the same throw a second time (its catch point),
    so for plays whose target has a route chart that point is used as extra evidence of where the
    ball went. Match routes first, then pass them in here.

    Returns one row per matched play, with every column from the chart data, from pbp_df, and (if
    pbp_df is the table from load_pbp_with_ftn) from FTN / participation - minus duplicate features:

      - kind, bucket - this function's own columns.
      - n_candidates - how many pbp plays in this game had this player + outcome. 1 means the match
        is certain.
      - match_cost - the matched pair's negative log-likelihood (lower = better agreement).
      - match_margin - how much worse this chart entity's next-best play would have been (nats).
        Large = the chart clearly singles out this play.
      - location_prob - estimated probability that the chart location attached to this play is within
        3 yards of where the play really happened. Every chart entity in the group is weighted by how
        well it fits this play; rivals sitting right next to the chosen one don't count against it,
        since swapping two dots that are on top of each other doesn't move anything. Averaged over
        many plays it estimates location accuracy (slightly conservatively - it ignores that other
        plays have already claimed some of the rivals).
      - match_prob - estimated probability that this chart entity is exactly this play (not just
        in the same spot). Both probabilities are temperature-calibrated per kind against evidence
        the model doesn't use (_TEMP; carries on FTN's starting hash).
      - location_confident - location_prob >= 0.95. For location-based analysis (heatmaps, charts
        by situation) this is the flag that matters; exact play identity only matters when looking
        up a single play.
      - swap_yds - the farthest any plausible rival (>= 5% weight) sits from the chosen location;
        0 when nothing else competes.
      - residual - |pbp yardage - chart yardage| in yards (air yards for pass/route, yards gained
        for carries), kept for continuity with earlier versions.
      - game_id, play_id - pbp's play key, plus every other pbp/FTN/participation column.
      - every chart column. Most are promoted to their natural, unprefixed name - name, team,
        position, season, season_type, week, pass_type/route_type/gain_class, located/start_ok/
        handoff_ok, x_coord/y_coord/catch_y, path_x/path_y/path_segment. Where one shares a name
        with a pbp column the chart's value wins and pbp's copy is dropped.
      - chart_game_id, chart_touchdown, chart_fumble_lost stay prefixed: pbp's game_id is the join
        key; pbp's touchdown/fumble_lost are whole-play flags, kept separately as a cross-check.

    Caveats:
      - pass: TOUCHDOWN / INTERCEPTION / COMPLETE are drawn 1:1 with pbp. INCOMPLETE is not - NGS
        doesn't draw every incompletion - so some incomplete pbp attempts simply go unmatched.
      - Plays that look the same to both the chart and pbp (same player, outcome, depth and side)
        can't be told apart by anyone; the matcher then picks one, and swap_yds says how much
        that could move the play's location.
    """
    assert kind in ("pass", "carry", "route")
    id_col = CHART_ID_COL[kind]
    play_type = CHART_PLAY_TYPE[kind]

    chart_reduced = to_paths(chart_df, kind)      # one row per pass/carry/route, every original column kept
    if kind == "pass":
        chart_reduced = chart_reduced[chart_reduced["x_coord"].notna() & chart_reduced["y_coord"].notna()]
    gmap = _game_id_crosswalk(pbp_df)
    chart_reduced["game_id_nflverse"] = chart_reduced["game_id"].astype(str).map(gmap)
    chart_reduced["gsis_id"] = chart_reduced["esb_id"].map(_esb_to_gsis(chart_reduced["esb_id"].unique()))

    pbp_typed = pbp_df[pbp_df["play_type"] == play_type]
    cfeat_all = _chart_features(chart_reduced, kind)
    pfeat_all = _pbp_features(pbp_typed, kind)
    if kind == "pass" and route_matches is not None and len(route_matches):
        rm = route_matches
        ci = rm["path_segment"].map(_catch_index)
        link = pd.DataFrame({"game_id": rm["game_id"].values, "play_id": rm["play_id"].values,
                             "link_x": [p[i] for p, i in zip(rm["path_x"], ci)],
                             "link_y": [p[i] for p, i in zip(rm["path_y"], ci)]})
        link = link.drop_duplicates(subset=["game_id", "play_id"])
        keyed = pbp_typed[["game_id", "play_id"]].reset_index().merge(link, on=["game_id", "play_id"], how="left")
        pfeat_all["link_x"] = keyed.set_index("index")["link_x"]
        pfeat_all["link_y"] = keyed.set_index("index")["link_y"]

    # ---- collect groups; forced 1:1 pairs calibrate the model ----
    groups, pairs = [], []
    pbp_by_key = {k: g for k, g in pbp_typed.groupby(["game_id", id_col])}
    for (game_id, gsis_id), grp_chart in chart_reduced.groupby(["game_id_nflverse", "gsis_id"]):
        grp_pbp = pbp_by_key.get((game_id, gsis_id))
        if grp_pbp is None:
            continue
        for bucket, (p_sub, c_sub) in _buckets(kind, grp_pbp, grp_chart).items():
            if len(p_sub) and len(c_sub):
                groups.append((game_id, bucket, p_sub, c_sub))
                if len(p_sub) == 1 and len(c_sub) == 1:
                    pairs.append((cfeat_all.loc[c_sub.index[0]], pfeat_all.loc[p_sub.index[0]]))
    params = _calibrate(kind, pairs)

    def assign(params):
        """One matching pass: per group, the optimal assignment plus each pair's location confidence."""
        out = []
        for game_id, bucket, p_sub, c_sub in groups:
            cf, pf = cfeat_all.loc[c_sub.index], pfeat_all.loc[p_sub.index]
            C = _cost_matrix(kind, cf, pf, params, bucket)
            rows, cols = linear_sum_assignment(C)
            M = _assignment_marginals(C / _TEMP[kind])
            loc = cf[["loc_x", "loc_y"]].to_numpy(float)
            for r, c in zip(rows, cols):
                alt_row = np.delete(C[r], c)
                margin = float(alt_row.min() - C[r, c]) if len(alt_row) else float("inf")
                # which chart entity really belongs to play c (given it's drawn at all)? a rival
                # sitting within _LOCATION_OK_YDS of the one we chose doesn't threaten the location
                w = M[:, c] / M[:, c].sum() if M[:, c].sum() > 0 else np.eye(len(c_sub))[r]
                dist = np.hypot(loc[:, 0] - loc[r, 0], loc[:, 1] - loc[r, 1])
                dist = np.where(np.isfinite(dist), dist, np.inf)
                dist[r] = 0.0
                loc_prob = float(w[dist <= _LOCATION_OK_YDS].sum())
                rivals = w >= 0.05
                rivals[r] = False
                swap = float(dist[rivals].max()) if rivals.any() else 0.0
                out.append((game_id, bucket, len(p_sub), c_sub.index[r], p_sub.index[c], float(C[r, c]),
                            margin, swap, loc_prob, float(w[r])))
        return out

    assigned = assign(params)
    if _TWO_PASS:
        # The forced 1:1 pairs are mostly touchdowns (goal-line runs, end-zone catches), which skews
        # the calibration for ordinary plays. Re-fit every term on the first pass's confident matches,
        # across all outcomes, and match again.
        confident = [(cfeat_all.loc[ci], pfeat_all.loc[pi]) for *_, ci, pi, _c, _m, _s, lp, _mp in assigned if lp >= 0.95]
        if len(confident) >= _MIN_CALIB:
            params = _calibrate(kind, pairs + confident)
            assigned = assign(params)

    # ---- build the output rows ----
    matches = []
    yard_col = "yards_gained" if kind == "carry" else "air_yards"
    chart_yard = cfeat_all["gain"] if kind == "carry" else cfeat_all["depth"]
    for game_id, bucket, n_cand, chart_idx, pbp_idx, cost, margin, swap, loc_prob, match_prob in assigned:
        chart_row = chart_reduced.loc[chart_idx].to_dict()
        pv, cv = pbp_df.loc[pbp_idx, yard_col], chart_yard.loc[chart_idx]
        matches.append(dict(
            kind=kind, bucket=bucket, game_id=game_id, play_id=pbp_df.loc[pbp_idx, "play_id"],
            n_candidates=n_cand, match_cost=round(cost, 3),
            match_margin=round(margin, 3) if np.isfinite(margin) else None,
            swap_yds=round(swap, 2), location_prob=round(loc_prob, 3), match_prob=round(match_prob, 3),
            location_confident=bool(loc_prob >= 0.95),
            residual=round(abs(float(pv) - float(cv)), 2) if pd.notna(pv) and pd.notna(cv) else None,
            **{f"chart_{k}": v for k, v in chart_row.items() if k not in ("game_id_nflverse", "gsis_id")}
        ))

    matched = pd.DataFrame(matches)
    if len(matched):
        matched = matched.rename(columns={f"chart_{c}": c for c in _PROMOTE_COLS if f"chart_{c}" in matched.columns})
        matched = _attach_pbp_columns(matched, pbp_df)
    return (matched, params) if return_params else matched


_UNMATCHED_NATS = 12.0   # a chart line with no play left to pair with (a reading with the wrong split)


def score_chart(chart_rows, pbp_rows, kind, params):
    """
    How well one player-game chart reading fits that player's plays in pbp_rows (already restricted to
    the same game and player): the total negative log-likelihood of the best one-to-one assignment,
    plus a flat penalty for chart lines left without a play. Lower = better. Used to choose between
    alternative readings of a tangled chart (see next_gen_scrapy.retrace).
    """
    red = to_paths(chart_rows, kind)
    if kind == "pass":
        red = red[red["x_coord"].notna() & red["y_coord"].notna()]
    cfeat, pfeat = _chart_features(red, kind), _pbp_features(pbp_rows, kind)
    total = 0.0
    for bucket, (p_sub, c_sub) in _buckets(kind, pbp_rows, red).items():
        if len(c_sub) and len(p_sub):
            C = _cost_matrix(kind, cfeat.loc[c_sub.index], pfeat.loc[p_sub.index], params, bucket)
            r, c = linear_sum_assignment(C)
            total += float(C[r, c].sum())
        total += max(0, len(c_sub) - len(p_sub)) * _UNMATCHED_NATS
    return total


def _attach_pbp_columns(matched, pbp_df):
    """
    Bring in every column of pbp_df (already including FTN's/participation's, if pbp_df is
    load_pbp_with_ftn's output) onto the matched rows, joining on (game_id, play_id) - the exact keys
    match_to_pbp already pulled from pbp_df for every matched row, so this is an exact join.

    Chart-derived columns on `matched` that share a name with a pbp_df column win; pbp_df's copy is
    dropped rather than letting pandas silently rename both to _x/_y.
    """
    dupes = [c for c in pbp_df.columns if c in matched.columns and c not in ("game_id", "play_id")]
    pbp_slim = pbp_df.drop(columns=dupes) if dupes else pbp_df
    return matched.merge(pbp_slim, on=["game_id", "play_id"], how="left", validate="m:1")
