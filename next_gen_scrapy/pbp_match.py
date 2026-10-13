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
import functools
import json
import os
import warnings

import numpy as np
import pandas as pd

from .reshape import to_paths
from . import route_type as _RT
from . import carry_type as _CT
from . import carry_pair as _CP

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
                "route_type", "gain_class", "located", "start_ok", "handoff_ok", "gap",
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


GAMECENTER_COLS = ["ngs_receiver_targets", "ngs_receiver_avg_air_yards", "ngs_passer_zones"]


def add_gamecenter(pbp, cache_dir=None):
    """
    Add NGS Game Center's per-game numbers to every pass play (needs network access: one request per game;
    cache_dir keeps them on disk):
      - ngs_receiver_targets, ngs_receiver_avg_air_yards: the targeted receiver's targets in that game and NGS's
        average air yards over them (each team's top five receivers by targets are listed). A receiver's targets
        must add up to his NGS total.
      - ngs_passer_zones: the passer's completions and yards per zone ("depth,section,completions,yards;..." -
        depth 0 behind the line / 1 0-9 / 2 10-19 / 3 20+ NGS air yards, section 0-2 the left / middle / right
        third of the field). The completions drawn in a zone must add up to its yards.
    match_to_pbp uses both to pin passes down.
    """
    from .gamecenter import game_tables
    pbp = pbp.drop(columns=[c for c in GAMECENTER_COLS if c in pbp.columns])
    rec, zon = game_tables(pbp["old_game_id"].dropna().unique(), cache_dir=cache_dir)
    gsis = _esb_to_gsis(pd.unique(pd.concat([rec["esb_id"], zon["esb_id"]]))) if len(rec) or len(zon) else None
    if len(zon):
        zon["passer_player_id"] = zon["esb_id"].map(gsis)
        zon = zon.dropna(subset=["passer_player_id"])
        txt = zon.groupby(["old_game_id", "passer_player_id"]).apply(
            lambda g: ";".join(f"{d},{q},{c},{y}" for d, q, c, y in zip(g.depth, g.section, g.completions, g.yards)))
        pbp = pbp.merge(txt.rename("ngs_passer_zones").reset_index(), on=["old_game_id", "passer_player_id"], how="left")
        pbp.loc[pbp["play_type"] != "pass", "ngs_passer_zones"] = np.nan
    if len(rec):
        rec["receiver_player_id"] = rec["esb_id"].map(gsis)
        rec = (rec.dropna(subset=["receiver_player_id"]).drop_duplicates(["old_game_id", "receiver_player_id"])
               .rename(columns={"targets": "ngs_receiver_targets", "avg_air_yards": "ngs_receiver_avg_air_yards"}))
        cols = ["ngs_receiver_targets", "ngs_receiver_avg_air_yards"]
        pbp = pbp.merge(rec[["old_game_id", "receiver_player_id", *cols]], on=["old_game_id", "receiver_player_id"], how="left")
        pbp.loc[pbp["play_type"] != "pass", cols] = np.nan
    for c in GAMECENTER_COLS:
        if c not in pbp.columns:
            pbp[c] = np.nan
    return pbp


def load_pbp_with_ftn(seasons, participation=True, gamecenter=True, gamecenter_cache=None, ngs_distance=True,
                      ngs_distance_cache=None):
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

    gamecenter=True adds NGS Game Center's per-receiver air yards to every pass play (see add_gamecenter) - since
    2023 the only public source of NGS's measured air yards. It takes one request per game; if NGS can't be
    reached the columns are left empty, with a warning.

    ngs_distance=True adds how far each play's ball carrier went, from NGS's public leader lists (ngs_distance.py):
    the length of a carry's or a completed route's chart line. About 700 requests a season (one per team-week;
    ngs_distance_cache keeps finished weeks on disk).

    FTN's own `season`/`week` columns are dropped before merging - identical to pbp's own on every
    matched row (verified against a real season), so keeping both would only produce pandas'
    auto-suffixed season_x/season_y and week_x/week_y instead of one column apiece.
    """
    nfl = _require_nflreadpy()
    seasons = [seasons] if isinstance(seasons, (int, str)) else list(seasons)
    pbp = nfl.load_pbp(seasons=seasons).to_pandas()

    ftn_seasons = [s for s in seasons if int(s) >= 2022]
    ftn_frames = []
    for s in ftn_seasons:
        try:
            ftn_frames.append(nfl.load_ftn_charting(seasons=[int(s)]).to_pandas())
        except ValueError as e:                  # not published for this season (yet)
            warnings.warn(f"no FTN charting for {s} yet ({e}); its FTN columns are left empty")
    if ftn_frames:
        ftn = pd.concat(ftn_frames, ignore_index=True)
        ftn["nflverse_play_id"] = ftn["nflverse_play_id"].astype("float64")
        ftn = ftn.drop(columns=["season", "week"])
        pbp = pbp.merge(ftn, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"],
                        how="left")

    if participation:
        # nflverse publishes participation data a season behind (e.g. none yet for the season in
        # progress): load what exists, and leave the columns empty - with a warning - for the rest
        frames = []
        for s in seasons:
            try:
                frames.append(nfl.load_participation(seasons=[int(s)]).to_pandas())
            except ValueError as e:
                warnings.warn(f"no NGS participation data for {s} yet ({e}); its participation columns are left empty")
        if frames:
            part = pd.concat(frames, ignore_index=True)
            keep = {k: v for k, v in _PARTICIPATION_COLS.items() if k in part.columns}
            part = part[["nflverse_game_id", "play_id", *keep]].rename(columns=keep)
            part = part.rename(columns={"nflverse_game_id": "_part_game_id", "play_id": "_part_play_id"})
            part["_part_play_id"] = part["_part_play_id"].astype("float64")
            part = part.drop_duplicates(subset=["_part_game_id", "_part_play_id"])
            pbp = pbp.merge(part, left_on=["game_id", "play_id"], right_on=["_part_game_id", "_part_play_id"],
                            how="left").drop(columns=["_part_game_id", "_part_play_id"])
        for col in _PARTICIPATION_COLS.values():        # same schema whatever was available
            if col not in pbp.columns:
                pbp[col] = np.nan
    if gamecenter:
        try:
            pbp = add_gamecenter(pbp, cache_dir=gamecenter_cache)
        except Exception as e:                      # NGS unreachable: match without it
            warnings.warn(f"no NGS Game Center data ({e}); matching works without it, less precisely")
            for c in GAMECENTER_COLS:
                pbp[c] = np.nan
    try:                                            # next snap's hash per run / completion (for chart readers)
        pbp = add_snap_context(pbp)
    except Exception as e:
        warnings.warn(f"no next-snap context ({e})")
    if ngs_distance:
        from .ngs_distance import add_ngs_distance
        try:
            pbp = add_ngs_distance(pbp, cache_dir=ngs_distance_cache)
        except Exception as e:                      # NGS unreachable: match without it
            warnings.warn(f"no NGS ball-carrier distances ({e}); matching works without them, less precisely")
            pbp["ngs_carrier_dist"], pbp["ngs_dist_below"] = np.nan, np.nan
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
#          type   the line's shape vs  participation target_route (route_type.py: shape -> route probabilities)
#          time   route yards before the catch vs participation time_to_throw (+ air yards)
#   carry  gain   end point y      vs  pbp yards_gained
#          gap    where the run crosses the LOS, relative to the handoff
#                                  vs  pbp run_location + run_gap (guard ~2 yds out, tackle ~4-5, end ~10)
#          dir    end x - start x  vs  pbp run_location
#          depth  handoff depth    vs  participation offense_formation (shotgun ~-4.4 yds, under center ~-6.4)
#          hash   start x          vs  FTN starting_hash                  (2022+)
#          time   the line's shape -> seconds snap to whistle (carry_type.predict_duration)
#                                  vs  pbp end_clock_time - time_of_day   (2022+)
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
    "route": {"ngs": (1.44, 1.18), "air": (2.04, 2.60), "yac": (0.0, 0.8), "ttt": (0.0, 5.3),
              "side": {"left": (-16.3, 8.5), "middle": (0.6, 7.6), "right": (18.7, 6.4)}},
    # chart end y minus pbp yards: ordinary runs end ~0.4 yd past the official spot, runs out of bounds a little
    # further (gain_oob: extra offset), touchdowns ~1.2 yds into the end zone (gain_td) - kept apart, because the
    # plays that can only match one way are mostly touchdowns and would drag the ordinary offset up
    "carry": {"gain": (0.42, 0.65), "gain_td": (1.15, 0.7), "gain_oob": (0.3, 0.0),
              # crossing point minus handoff x, keyed "location/gap" (measured on unambiguous 2022 carries)
              "gap": {"left/end": (-9.8, 5.4), "left/tackle": (-5.0, 3.1), "left/guard": (-2.4, 2.6),
                      "middle/": (0.0, 2.9), "left/": (-5.5, 5.5), "right/": (5.5, 5.5),
                      "right/guard": (1.7, 2.0), "right/tackle": (4.1, 4.6), "right/end": (10.8, 5.4)},
              # participation's formation labels changed in 2023 (UNDER CENTER / SHOTGUN / PISTOL only)
              "form": {"SHOTGUN": (-4.4, 1.8), "SINGLEBACK": (-6.4, 1.9), "I_FORM": (-6.4, 1.4),
                       "PISTOL": (-6.3, 2.5), "JUMBO": (-5.2, 1.5), "WILDCAT": (-4.6, 1.5),
                       "UNDER CENTER": (-6.4, 1.9), "EMPTY": (-4.4, 1.8)},
              "hash": {"L": (-2.1, 2.5), "M": (-0.6, 2.4), "R": (1.9, 3.3)},
              "qbloc": {"S": (-4.4, 1.8), "U": (-6.4, 1.9), "P": (-6.0, 2.0)},
              "rundir": {"left": (-6.0, 6.0), "middle": (0.0, 4.0), "right": (6.0, 6.0)},
              # pbp seconds snap -> end of play minus the line's predicted seconds (Laplace offset, scale)
              "dur": (0.0, 0.6), "dur_ngs": (0.0, 0.75)},
}
_USE_ROUTE_X = True      # switch for A/B evaluation of the route-type term
_USE_OOB = True          # switch for A/B evaluation of the out-of-bounds / fumble terms
_USE_ROUTE_TYPE = True   # route charts: the line's shape vs participation's target_route (route_type.py)
_RT_MODEL = None         # route-type model override (dict from route_type.fit); None = the packaged model
_RT_W, _RT_EPS = 1.0, 0.15   # weight of the route-type term, and the share of route labels it treats as noise
_USE_HASH = False        # carries: start x vs FTN starting_hash on its own - off: the snap-spot term (_spot_cost) scores
                         # the same evidence, and counting it twice made matches worse against tracked tackle spots
_USE_FORM = True         # carries: handoff depth vs participation offense_formation (switch for grading)
_USE_QBLOC = False       # carries: handoff depth vs FTN qb_location - same information as the formation term
_USE_RUNDIR = True       # carries: where the run ends across the field, relative to its start, vs run_location
_GAIN_SCALE = 1.0        # carries: multiplier on the fitted spread of chart end y vs pbp yards
_USE_GAIN_ALT = True     # carries: also accept the description's run yards where a foul enforced at a spot cut pbp's
_USE_DUR = True          # carries: how long the play lasted (pbp, 2022+) vs what the line's shape implies
_DUR_MODEL = None        # duration model override (path or dict with a "dur" head) - for held-out evaluation
_DUR_RANGE = 12.0        # seconds the duration term's outlier component spreads over
_DUR_W = 1.0             # weight of the duration term
# seconds pbp's clock adds for what a carry line doesn't show: a QB's dropback before he scrambles, a fumble
# (measured on confidently matched carries, 2022-25); touchdowns end the moment the ball crosses the plane
_DUR_ADJ = {"qb_scramble": 1.6, "fumble": 1.5, "touchdown": -0.35}
_USE_DEPTH_TAB = True    # pass/route depth: offset/spread per outcome (complete, incomplete, TD, INT) and depth band
_DEPTH_EDGES = (0.0, 5.0, 10.0, 20.0)   # bands of the pbp air yards: <0, 0-5, 5-10, 10-20, 20+
_MIN_SCALE = {"ngs": 0.25, "air": 0.4}  # floor on a fitted band's spread (fits come from confident matches)
_USE_DRAWN = True        # passes: NGS doesn't draw ~22% of pbp incompletions - P(drawn) from pbp per play
# logistic model of P(an incompletion is drawn on the QB's chart), fitted on QB-games whose dots all matched
# confidently, 2018-25 (AUC 0.80 on held-out seasons): no intended receiver is the big one (93% undrawn)
_DRAWN_COEF = {"intercept": 2.531, "no_receiver": -4.591, "defended": 0.22, "hit": -0.032, "pressure": -0.169,
               "oop": -0.115, "air": -0.05, "air_neg": -0.328, "air_deep": -0.157}
_TEMP_BUCKET = {("pass", "INCOMPLETE"): 1.0}   # per-bucket override of _TEMP
_USE_TTT = True          # route charts: route length before the catch vs participation time_to_throw
_USE_GC = True           # passes: NGS Game Center per-receiver air yards (the receiver's targets sum to his total)
_GC_ITERS = 6            # assignment <-> implied-depth rounds (stops early once nothing changes)
_GC_OFF = 0.85           # a pass dot sits this much deeper than NGS's air yards for the throw (measured 2021, 2024)
_GC_B = 0.30             # ... with this Laplace spread
_GC_SD_AIR = (1.0, 2.0)  # spread of pbp air yards as a stand-in for NGS's, completions / other passes
_USE_ZONES = True        # passes: NGS Game Center passer zones (the completions in a zone sum to its yards)
_ZONE_THIRD = 53.33 / 6  # a third of the field's width, either side of the middle
_ZONE_B = 0.5            # spread (yards) of a zone's yards vs pbp yards gained (equal on ~99% of clean zones)
_USE_PASS_HASH = True    # pass/route: shift the side term's expected x by FTN starting_hash (2022+)
_HASH_SHIFT = 1.5        # yards a throw's x moves per hash (L = -1, M = 0, R = +1) before calibration
# ...which depends on the throw (confident 2023-25 matches): short throws move ~2 yd per hash, wide ones 10-20 yds
# downfield ~0.5, deep ones not at all - receivers spread to the field's own landmarks, not the ball's
_HASH_SHIFT_TAB = True
_HASH_SHIFT_TAB_BUCKETS = ("COMPLETE", "TOUCHDOWN")
# FTN's is_catchable_ball: a catchable incompletion's dot sits about twice as close to pbp air yards (depth residual
# MAD 0.63 yd vs 1.21 uncatchable, confident 2023-25 matches) - scale the incompletion depth spread by it
_USE_CATCHABLE = False   # measured: no gain on tracked incompletions (truth set) - kept switchable
_CATCHABLE_SCALE = {1: 0.6, 0: 1.15}
_HASH_SHIFT_BY = {"side": [(10.0, 2.0), (20.0, 0.5), (np.inf, 0.0)], "middle": [(20.0, 1.9), (np.inf, 0.0)]}


def _hash_shift_per_play(side, air, hash_x):
    """Per-play x shift (yd) of the expected throw from the hash, by pass_location and pbp air yards."""
    side, air, hx = np.asarray(side, object), np.asarray(air, float), np.nan_to_num(np.asarray(hash_x, float))
    out = np.zeros(len(hx))
    for kind, bands in _HASH_SHIFT_BY.items():
        sel = (side == "middle") if kind == "middle" else np.isin(side, ["left", "right"])
        lo = -np.inf
        for hi, v in bands:
            m = sel & (np.nan_to_num(air, nan=5.0) >= lo) & (np.nan_to_num(air, nan=5.0) < hi)
            out[m] = v
            lo = hi
    return out * hx
_TTT_COEF = (3.19, 3.73, 0.69)   # route yards before the catch ~ a + b * time_to_throw (s) + c * air yards,
                                 # fitted on ~33k confidently matched routes (2018-25)

# P(chart line ends within 2.7 yds of a sideline | pbp out_of_bounds), measured on confident matches
_SIDELINE_YD = 24.0
_P_END_AT_SIDELINE = {"carry": {1: 0.98, 0: 0.10}, "route": {1: 0.89, 0: 0.17}}
_P_FUMBLE_AGREE = 0.97
_USE_PAIR = True         # carries: the learned (line, run) scorer, carry_pair.py
_PAIR_W = 1.0
# carry chart colour (red / yellow / green) vs pbp yards_gained: LOSS < 0, SHORT 0-5, LONG 6+ -
# agrees 97% of the time on confidently matched carries
_P_GAIN_CLASS_AGREE = 0.97
_TWO_PASS = True         # recalibrate on the first pass's confident matches, then match again
# Temperature applied to the costs when turning them into match probabilities (not to the assignment
# itself): < 1 sharpens, > 1 softens. Fitted per kind so the reported probabilities agree with
# held-out evidence the model doesn't use (see the notes on calibration in the README).
# Fitted against independent checks (2018-25): passes and routes against each other's chart (the QB's dot and the
# receiver's catch point are the same throw), carries against FTN starting_hash and participation formation, each
# check held out of the model it grades. Passes came out underconfident (sharpened), carries overconfident
# (softened - more so without FTN's hash, i.e. before 2022).
_TEMP = {"pass": 0.35, "route": 1.0, "carry": 2.8}
_TEMP_CARRY_WITH_HASH = 2.0
_TEMP_PASS_NO_NGS = 0.5   # pass temperature where pbp has no NGS air yards per play (2023+)
# pass location_prob, as the assignment computes it -> how often that's actually right (completions graded against
# receivers' route charts the match never saw, 2018-25, pooled by bins and made monotone). The raw numbers run
# low at the bottom - they count a rival dot as a threat even when another play has already claimed it - and a
# little high in the upper middle. Separate maps with / without NGS per-play air yards (2023+ has none).
_LOC_CALIBRATION = {
    True: ((0.0, 0.0), (0.165, 0.301), (0.412, 0.606), (0.55, 0.719), (0.754, 0.794), (0.928, 0.887), (0.975, 0.944),
           (0.999, 1.0), (1.0, 1.0)),
    False: ((0.0, 0.0), (0.183, 0.502), (0.419, 0.582), (0.551, 0.716), (0.70, 0.825), (0.852, 0.843), (0.926, 0.869),
            (0.975, 0.888), (0.999, 0.964), (1.0, 0.964)),
}
# carry location_prob, as the assignment computes it -> how often the line really ends within 3 yds of where the play
# ended: graded on 2023-25 runs against the Big Data Bowl's tracked tackle spots (the line ends ~1.15 yds past the
# tackler; even near-certain matches land within 3 yds of him only 90% of the time, so the share within 3 yds is
# divided by that), isotonic. With FTN's hashes (2022+) / without (the same runs with the hashes and play times taken
# away, standing in for 2018-21).
_CARRY_LOC_CALIBRATION = {
    # refit with the learned pair term (carry_pair.py): a 2-parameter logistic of raw location_prob vs the share of
    # matched line ends within 3 yds of the tracked end / 0.90, Big Data Bowl 2023-25, checked season-out; the lowest
    # band follows the data (few runs, right ~10-20%). True: games with FTN's hash, False: without
    True: ((0.0, 0.0), (0.05, 0.1), (0.1, 0.2), (0.2, 0.35), (0.3, 0.567), (0.4, 0.603), (0.5, 0.636), (0.6, 0.669),
           (0.7, 0.703), (0.8, 0.743), (0.85, 0.768), (0.9, 0.8), (0.95, 0.847), (0.98, 0.899), (0.99, 0.933),
           (0.995, 0.961), (1.0, 1.0)),
    False: ((0.0, 0.0), (0.05, 0.1), (0.1, 0.2), (0.2, 0.35), (0.3, 0.498), (0.4, 0.547), (0.5, 0.592), (0.6, 0.637),
            (0.7, 0.684), (0.8, 0.74), (0.85, 0.774), (0.9, 0.816), (0.95, 0.876), (0.98, 0.939), (0.99, 0.976),
            (0.995, 1.0), (1.0, 1.0)),
}
# route location_prob where pbp has no NGS air yards (2023+) -> how often a completion's catch point is really within
# 2 yds of where the tracked receiver caught it (Big Data Bowl 2024-25, divided by the 93% that near-certain matches
# reach), isotonic. Before 2023 the QB chart cross-check says route probabilities are already right.
_ROUTE_LOC_CALIBRATION = ((0.0, 0.0), (0.63, 0.33), (0.885, 0.55), (0.976, 0.84), (0.996, 0.93), (1.0, 1.0))
_GC_PROB_SCALE = 1.0      # weight of the Game Center terms in the probabilities (the assignment uses them fully)
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
        f["end_x"] = [px[-1] for px in chart["path_x"]]
        f["path_len"] = [float(np.hypot(np.diff(np.asarray(px, float)), np.diff(np.asarray(py, float))).sum())
                         for px, py in zip(chart["path_x"], chart["path_y"])]
        f["route_len"] = [float(np.hypot(np.diff(np.asarray(px[:i + 1], float)), np.diff(np.asarray(py[:i + 1], float))).sum())
                          for px, py, i in zip(chart["path_x"], chart["path_y"], ci)]
        if _USE_ROUTE_TYPE:
            P = _RT.predict_proba(chart, model=_RT_MODEL)
            for i in range(P.shape[1]):
                f[f"rt{i}"] = P[:, i]
    else:
        f["gain"] = chart["y_coord"]
        f["x0"] = [px[0] for px in chart["path_x"]]
        f["y0"] = [py[0] for py in chart["path_y"]]
        f["path_len"] = [float(np.hypot(np.diff(px), np.diff(py)).sum()) for px, py in zip(chart["path_x"], chart["path_y"])]
        f["cross"] = [_crossing_x(px, py) - px[0] for px, py in zip(chart["path_x"], chart["path_y"])]
        f["dx_end"] = [px[-1] - px[0] for px in chart["path_x"]]
        f["loc_x"] = [px[-1] for px in chart["path_x"]]
        f["loc_y"] = chart["y_coord"]
        f["end_sideline"] = [abs(px[-1]) >= _SIDELINE_YD for px in chart["path_x"]]
        f["fumble"] = chart["fumble_lost"].astype(str).str.lower().eq("true") if "fumble_lost" in chart else False
        f["gclass"] = chart["gain_class"] if "gain_class" in chart else None
        if _USE_CARRY_TYPE:
            pr = _CT.predict_proba(chart, model=_CT_MODEL)
            for i in range(len(_CT.GAPS)):
                f[f"ct_gap{i}"] = pr["gap"][:, i]
            f["ct_gun"] = pr["shotgun"][:, 1]
        if _USE_DUR:
            f["dur_hat"] = _CT.predict_duration(chart, model=_DUR_MODEL if _DUR_MODEL is not None else _CT_MODEL)
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
        f["catchable"] = pbp["is_catchable_ball"].astype(float) if "is_catchable_ball" in pbp else np.nan
        f["side"] = pbp["pass_location"]
        f["hash_x"] = (pbp["starting_hash"].map({"L": -1.0, "M": 0.0, "R": 1.0}) if "starting_hash" in pbp
                       else pd.Series(np.nan, index=pbp.index))
        if "target_route" in pbp:
            key = pbp["pass_location"].fillna("") + "/" + pbp["target_route"].fillna("")
            f["side_route"] = key.where(key.isin(list(_ROUTE_X)), None)
        if kind == "pass" and "ngs_receiver_avg_air_yards" in pbp:
            f["gc_n"] = pbp["ngs_receiver_targets"].astype(float)
            f["gc_avg"] = pbp["ngs_receiver_avg_air_yards"].astype(float)
            f["gc_key"] = (pbp["game_id"].astype(str) + "|" + pbp["receiver_player_id"].astype(str)).where(
                f["gc_n"].notna() & pbp["receiver_player_id"].notna())
            f["complete"] = pbp["complete_pass"].astype(float)
        if kind == "pass" and "ngs_passer_zones" in pbp:
            f["zones"] = pbp["ngs_passer_zones"]
            f["qb_key"] = pbp["game_id"].astype(str) + "|" + pbp["passer_player_id"].astype(str)
            f["yds"] = pbp["yards_gained"].astype(float)
        if kind == "pass":
            col = lambda c: pbp[c] if c in pbp else pd.Series(np.nan, index=pbp.index)
            air = pbp["air_yards"].astype(float)
            z = (_DRAWN_COEF["intercept"]
                 + _DRAWN_COEF["no_receiver"] * pbp["receiver_player_id"].isna().astype(float)
                 + _DRAWN_COEF["defended"] * col("pass_defense_1_player_id").notna().astype(float)
                 + _DRAWN_COEF["hit"] * (col("qb_hit") == 1).astype(float)
                 + _DRAWN_COEF["pressure"] * col("was_pressure").fillna(0).astype(float)
                 + _DRAWN_COEF["oop"] * col("is_qb_out_of_pocket").fillna(0).astype(float)
                 + _DRAWN_COEF["air"] * air.fillna(0) + _DRAWN_COEF["air_neg"] * (air < 0).astype(float)
                 + _DRAWN_COEF["air_deep"] * (air >= 20).astype(float))
            f["drawn_logit"] = z
        if kind == "route":
            if "target_route" in pbp:
                lab = pbp["target_route"].replace(_RT.UNIFY)
                f["rt_class"] = lab.map({c: i for i, c in enumerate(_RT.CLASSES)}).astype(float)
            f["ngs_dist"] = pbp["ngs_carrier_dist"].astype(float) if "ngs_carrier_dist" in pbp else np.nan
            f["screen"] = pbp["is_screen_pass"].astype(float) if "is_screen_pass" in pbp else np.nan
            f["yac"] = pbp["yards_after_catch"].astype(float)
            if "next_hash_route" in pbp:              # add_snap_context
                f["next_hash"] = pbp["next_hash_route"]
            f["oob"] = pbp["out_of_bounds"].astype(float) if "out_of_bounds" in pbp else np.nan
            ttt = pbp["time_to_throw"].astype(float) if "time_to_throw" in pbp else pd.Series(np.nan, index=pbp.index)
            a, b, c = _TTT_COEF
            f["len_hat"] = a + b * ttt + c * f["ngs"].where(f["ngs"].notna(), f["air"])
    else:
        f["gain"] = pbp["yards_gained"].astype(float)
        # a foul enforced at a spot cuts pbp's yards there, but the chart draws the whole run - which the play's
        # description still gives ("... for 13 yards (tackler). PENALTY on ..., enforced at ...")
        if "desc" in pbp:
            ran = pbp["desc"].fillna("").str.extract(r"for (-?\d+) yards?|for (no gain)")
            ran = pd.to_numeric(ran[0], errors="coerce").where(ran[1].isna(), 0.0)
            f["gain_alt"] = ran.where(ran.notna() & (ran != f["gain"]))
        loc = pbp["run_location"].fillna("")
        gap = pbp["run_gap"].fillna("") if "run_gap" in pbp else ""
        f["gap"] = np.where(loc == "", None, loc + "/" + np.where(loc == "middle", "", gap))
        f["form"] = pbp["offense_formation"] if "offense_formation" in pbp else None
        f["hash"] = pbp["starting_hash"] if "starting_hash" in pbp else None
        f["gun"] = pbp["shotgun"].astype(float) if "shotgun" in pbp else np.nan
        f["qbloc"] = pbp["qb_location"] if "qb_location" in pbp else None
        f["rundir"] = pbp["run_location"]
        f["ngs_dist"] = pbp["ngs_carrier_dist"].astype(float) if "ngs_carrier_dist" in pbp else np.nan
        pair = _CP._load() if _USE_PAIR else None
        if pair is not None:                          # the pbp columns the learned pair scorer reads
            for c in pair["pbp_cols"]:
                if c in pbp:
                    f["pm_" + c] = pbp[c].to_numpy()
        if "next_hash_carry" in pbp:                  # add_snap_context: lets a chart reader score with it too
            f["next_hash"] = pbp["next_hash_carry"]
        f["ngs_below"] = pbp["ngs_dist_below"].astype(float) if "ngs_dist_below" in pbp else np.nan
        f["oob"] = pbp["out_of_bounds"].astype(float) if "out_of_bounds" in pbp else np.nan
        f["fumble"] = pbp["fumble_lost"].astype(float) if "fumble_lost" in pbp else np.nan
        y = pbp["yards_gained"].astype(float)
        f["gclass"] = np.where(y.isna(), None, np.where(y < 0, "LOSS", np.where(y <= 5, "SHORT", "LONG")))
        g_lab, s_lab = _CT.labels(pbp)
        f["ct_gap_lab"], f["ct_gun_lab"] = g_lab.astype(float), s_lab.astype(float)
        adj = sum(w * (pbp[col].fillna(0).to_numpy(float) == 1) for col, w in _DUR_ADJ.items() if col in pbp)
        f["dur"] = _CT.play_duration(pbp) - adj
        # where FTN's clock is missing (before 2022, the newest week), NGS's distance and top speed say it nearly as
        # well: seconds ~ 1.0 + 1.41 x distance / top speed (yd/s), ~0.42 s MAD on 2024 runs (ngs_distance.py)
        if "ngs_carrier_dist" in pbp and "ngs_max_speed" in pbp:
            spd = pbp["ngs_max_speed"].astype(float) * 0.4889             # mph -> yd/s
            prox = _DUR_NGS_COEF[0] + _DUR_NGS_COEF[1] * pbp["ngs_carrier_dist"].astype(float) / spd.where(spd > 0)
            f["dur_ngs"] = prox.where(f["dur"].isna())

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


def _depth_band(v):
    return np.searchsorted(np.asarray(_DEPTH_EDGES), np.asarray(v, float), side="right")


def _calibrate(kind, pairs):
    """Offsets/spreads for every term, from 1:1 pairs [(chart feature row, pbp feature row, bucket), ...]."""
    p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DEFAULTS[kind].items()}
    if not pairs:
        return p
    c = pd.DataFrame([t[0] for t in pairs]).reset_index(drop=True)
    b = pd.DataFrame([t[1] for t in pairs]).reset_index(drop=True)
    bucket = np.array([t[2] if len(t) > 2 else None for t in pairs], dtype=object)

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
        x_side = c["x"]
        if _USE_PASS_HASH and "hash_x" in b:
            # half the gap between throws from the right and the left hash, after each side label's own centre
            resid = c["x"] - c["x"].groupby(b["side"]).transform("median")
            h = b["hash_x"]
            if (h == -1).sum() >= _MIN_CALIB and (h == 1).sum() >= _MIN_CALIB:
                p["hash_shift"] = float((resid[h == 1].median() - resid[h == -1].median()) / 2)
            flat = p.get("hash_shift", _HASH_SHIFT) * b["hash_x"].fillna(0.0)
            tabbed = np.isin(np.asarray(bucket, object), _HASH_SHIFT_TAB_BUCKETS) if _HASH_SHIFT_TAB else False
            x_side = c["x"] - np.where(tabbed, _hash_shift_per_play(b["side"], b["air"], b["hash_x"]), flat)
        fit_side("side", x_side, b["side"])
        # depth agreement differs a lot by outcome (a completion's dot is ~2x tighter to pbp than an
        # incompletion's) and grows with depth: one offset/spread per (source, outcome, depth band)
        tab = {}
        for src in ("ngs", "air"):
            res = (c["depth"] - b[src]).to_numpy(float)
            band = _depth_band(b[src].to_numpy(float))
            for bk in set(bucket) - {None}:
                for bd in range(len(_DEPTH_EDGES) + 1):
                    m = (bucket == bk) & (band == bd) & np.isfinite(res)
                    if m.sum() >= _MIN_CALIB:
                        med, sc = _robust(res[m])
                        tab[(src, bk, bd)] = (med, max(sc, _MIN_SCALE[src]))
        p["depth_tab"] = tab
        if kind == "route":
            comp = b["yac"].notna()
            fit("yac", (c["yac"] - b["yac"])[comp])
            if "len_hat" in b and "route_len" in c:
                fit("ttt", c["route_len"] - b["len_hat"])
    else:
        td = bucket == "TOUCHDOWN"
        oob = b["oob"].to_numpy(float) == 1 if "oob" in b else np.zeros(len(b), bool)
        r = (c["gain"] - b["gain"]).to_numpy(float)
        fit("gain", r[~td & ~oob])
        fit("gain_td", r[td])
        if (~td & oob).sum() >= _MIN_CALIB and np.isfinite(r[~td & oob]).any():
            p["gain_oob"] = (float(np.nanmedian(r[~td & oob]) - p["gain"][0]), 0.0)
        fit_side("gap", c["cross"], b["gap"])
        fit_side("form", c["y0"], b["form"])
        fit_side("hash", c["x0"], b["hash"])
        fit_side("qbloc", c["y0"], b["qbloc"])
        fit_side("rundir", c["dx_end"], b["rundir"])
        if "dur_hat" in c and "dur" in b:
            fit("dur", b["dur"] - c["dur_hat"])
        if "dur_hat" in c and "dur_ngs" in b:
            fit("dur_ngs", b["dur_ngs"] - c["dur_hat"])
    return p


def _side_cost(x, labels, table, shift=None):
    """x: (n_chart,), labels: (n_pbp,) -> (n_chart, n_pbp) cost of each chart x under each pbp side label.
    shift (n_pbp,), optional: moves each play's expected x (e.g. by the hash the ball was snapped from)."""
    out = np.full((len(x), len(labels)), _NLL_MISSING)
    x = np.asarray(x, float)
    known = np.isfinite(x)                 # an unknown chart measurement is "no information", like an unknown label
    for j, lab in enumerate(labels):
        if isinstance(lab, str) and lab in table:
            mu, sd = table[lab]
            out[known, j] = _nll_gauss(x[known] - mu - (shift[j] if shift is not None else 0.0), sd)
    return out


def _hash_shift(pf, params, bucket=None):
    """Per-play shift of the expected throw x from FTN starting_hash (0 where unknown or switched off)."""
    if not _USE_PASS_HASH or "hash_x" not in pf:
        return None
    if _HASH_SHIFT_TAB and "side" in pf and "air" in pf and bucket in _HASH_SHIFT_TAB_BUCKETS:
        return _hash_shift_per_play(pf["side"].to_numpy(object), pf["air"].to_numpy(float), pf["hash_x"].to_numpy(float))
    return params.get("hash_shift", _HASH_SHIFT) * np.nan_to_num(pf["hash_x"].to_numpy(float))


def _cost_matrix(kind, cf, pf, params, bucket, split_link=False):
    """(n_chart, n_pbp) total negative log-likelihood."""
    if kind in ("pass", "route"):
        depth = cf["depth"].to_numpy(float)[:, None]
        ngs, air = pf["ngs"].to_numpy(float), pf["air"].to_numpy(float)
        (bn, sn), (ba, sa) = params["ngs"], params["air"]
        use_ngs = np.isfinite(ngs)
        tab = params.get("depth_tab", {}) if _USE_DEPTH_TAB else {}
        if tab:
            on, sn_, oa, sa_ = (np.full(len(ngs), v, float) for v in (bn, sn, ba, sa))
            for arr_o, arr_s, src, vals in ((on, sn_, "ngs", ngs), (oa, sa_, "air", air)):
                band = _depth_band(vals)
                for j in range(len(vals)):
                    o_s = tab.get((src, bucket, int(band[j])))
                    if o_s is not None:
                        arr_o[j], arr_s[j] = o_s
            d_cost = np.where(use_ngs[None, :], _nll_laplace(depth - on[None, :] - np.nan_to_num(ngs)[None, :], sn_[None, :]),
                              _nll_laplace(depth - oa[None, :] - np.nan_to_num(air)[None, :], sa_[None, :]))
        else:
            d_cost = np.where(use_ngs[None, :], _nll_laplace(depth - bn - np.nan_to_num(ngs)[None, :], sn),
                              _nll_laplace(depth - ba - np.nan_to_num(air)[None, :], sa))
        if kind == "pass" and bucket == "INCOMPLETE" and _USE_CATCHABLE and "catchable" in pf:
            cat = pf["catchable"].to_numpy(float)
            k = np.where(cat == 1, _CATCHABLE_SCALE[1], np.where(cat == 0, _CATCHABLE_SCALE[0], 1.0))
            sa_k = (sa_ if tab else np.full(len(air), sa, float)) * k
            oa_k = oa if tab else np.full(len(air), ba, float)
            d_air = _nll_laplace(depth - oa_k[None, :] - np.nan_to_num(air)[None, :], sa_k[None, :])
            d_cost = np.where(use_ngs[None, :], d_cost, d_air)
        d_cost = np.where((~use_ngs & ~np.isfinite(air))[None, :], _NLL_MISSING, d_cost)
        hs = _hash_shift(pf, params, bucket)
        side = _side_cost(cf["x"].to_numpy(float), pf["side"].to_numpy(object), params["side"], hs)
        if "side_route" in pf and _USE_ROUTE_X:
            sharper = _side_cost(cf["x"].to_numpy(float), pf["side_route"].to_numpy(object), _ROUTE_X, hs)
            has = np.array([isinstance(k, str) for k in pf["side_route"]])
            side = np.where(has[None, :], sharper, side)
        C = d_cost + side
        if kind == "pass" and bucket == "INCOMPLETE" and _USE_DRAWN and "drawn_logit" in pf:
            # matching a dot to a play NGS probably didn't draw costs what that's unlikely to have been drawn
            # (relative to leaving it unmatched): -log P(drawn) + log P(not drawn)
            C = C - pf["drawn_logit"].to_numpy(float)[None, :]
        if kind == "route" and _USE_ROUTE_TYPE and "rt_class" in pf and "rt0" in cf:
            C = C + _route_type_cost(cf, pf)
        if kind == "route" and _USE_SCREEN and "screen" in pf and "rt0" in cf:
            C = C + _screen_cost(cf, pf)
        if kind == "route" and _USE_TTT and "len_hat" in pf:
            lh = pf["len_hat"].to_numpy(float)
            bt, st = params["ttt"]
            t_cost = _nll_laplace(cf["route_len"].to_numpy(float)[:, None] - bt - np.nan_to_num(lh)[None, :], st)
            C = C + np.where(np.isfinite(lh)[None, :], t_cost, _NLL_MISSING)
        if kind == "route" and bucket != "INCOMPLETE":
            yac = pf["yac"].to_numpy(float)
            by, sy = params["yac"]
            y_cost = _nll_laplace(cf["yac"].to_numpy(float)[:, None] - by - np.nan_to_num(yac)[None, :], sy)
            C = C + np.where(np.isfinite(yac)[None, :], y_cost, _NLL_MISSING)
            if _USE_OOB:
                C = C + _flag_cost(cf["end_sideline"], pf["oob"], _P_END_AT_SIDELINE["route"])
    else:
        gain = pf["gain"].to_numpy(float)
        bg, sg = params["gain_td"] if bucket == "TOUCHDOWN" and "gain_td" in params else params["gain"]
        sg = sg * _GAIN_SCALE
        if bucket != "TOUCHDOWN" and "oob" in pf:
            bg = bg + params.get("gain_oob", _DEFAULTS["carry"]["gain_oob"])[0] * (pf["oob"].to_numpy(float) == 1)[None, :]
        g_cost = _nll_laplace(cf["gain"].to_numpy(float)[:, None] - bg - np.nan_to_num(gain)[None, :], sg)
        if "gain_alt" in pf and _USE_GAIN_ALT:
            alt = pf["gain_alt"].to_numpy(float)
            a_cost = _nll_laplace(cf["gain"].to_numpy(float)[:, None] - bg - np.nan_to_num(alt)[None, :], sg)
            both = -np.log(0.5 * np.exp(-g_cost) + 0.5 * np.exp(-a_cost))
            g_cost = np.where(np.isfinite(alt)[None, :], both, g_cost)
        C = np.where(np.isfinite(gain)[None, :], g_cost, _NLL_MISSING)
        if _USE_GAP_TERM:
            C = C + _side_cost(cf["cross"].to_numpy(float), pf["gap"].to_numpy(object), params["gap"])
        if _USE_CARRY_TYPE and "ct_gun" in cf:
            C = C + _carry_type_cost(cf, pf)
        if _USE_FORM:
            C = C + _side_cost(cf["y0"].to_numpy(float), pf["form"].to_numpy(object), params["form"])
        if _USE_SPOT and not _SPOT_IN_G:
            bx = pf["bx"].to_numpy(float) if "bx" in pf else np.full(len(pf), np.nan)
            C = C + _spot_cost(cf["x0"].to_numpy(float), pf["hash"].to_numpy(object), bx, pf["gun"].to_numpy(float))
        elif _USE_HASH:
            C = C + _side_cost(cf["x0"].to_numpy(float), pf["hash"].to_numpy(object), params["hash"])
        if _USE_QBLOC and "qbloc" in pf:
            C = C + _side_cost(cf["y0"].to_numpy(float), pf["qbloc"].to_numpy(object), params["qbloc"])
        if _USE_NGS_DIST and _NGS_DIST_IN_C and "ngs_dist" in pf and "path_len" in cf:
            nd = _ngs_dist_cost(cf["path_len"].to_numpy(float), pf["ngs_dist"].to_numpy(float),
                                pf["ngs_below"].to_numpy(float), pf["gain"].to_numpy(float))
            C = C + _NGS_DIST_W * (nd - nd.mean(axis=1, keepdims=True))
        if _USE_RUNDIR and "rundir" in pf:
            C = C + _side_cost(cf["dx_end"].to_numpy(float), pf["rundir"].to_numpy(object), params["rundir"])
        if _USE_OOB:
            C = C + _flag_cost(cf["end_sideline"], pf["oob"], _P_END_AT_SIDELINE["carry"])
            C = C + _flag_cost(cf["fumble"], pf["fumble"], {1: _P_FUMBLE_AGREE, 0: 1 - _P_FUMBLE_AGREE})
        if _USE_DUR and "dur_hat" in cf and "dur" in pf:
            C = C + _DUR_W * _dur_cost(cf["dur_hat"].to_numpy(float), pf["dur"].to_numpy(float), params.get("dur", _DEFAULTS["carry"]["dur"]))
        if _USE_DUR_NGS and "dur_hat" in cf and "dur_ngs" in pf:
            C = C + _DUR_W * _dur_cost(cf["dur_hat"].to_numpy(float), pf["dur_ngs"].to_numpy(float), params.get("dur_ngs", _DUR_NGS_DEFAULT))
        if "gclass" in cf and "gclass" in pf:
            same = cf["gclass"].to_numpy(object)[:, None] == pf["gclass"].to_numpy(object)[None, :]
            known = pd.notna(pf["gclass"].to_numpy(object))[None, :] & pd.notna(cf["gclass"].to_numpy(object))[:, None]
            C = C + np.where(known, np.where(same, -np.log(_P_GAIN_CLASS_AGREE), -np.log((1 - _P_GAIN_CLASS_AGREE) / 2)), 0.0)
    L = np.zeros_like(C)
    if kind in ("pass", "route") and "link_x" in pf and (kind == "pass" or bucket != "INCOMPLETE"):
        # the receiver's route chart (for a pass) or the QB's pass chart (for a completed route) puts this same throw
        # at (link_x, link_y) - a second, independent measurement of where the ball went
        lx, ly = pf["link_x"].to_numpy(float), pf["link_y"].to_numpy(float)
        d2 = (cf["x"].to_numpy(float)[:, None] - lx[None, :]) ** 2 + (cf["depth"].to_numpy(float)[:, None] - ly[None, :]) ** 2
        dens = (1 - _FLOOR) * np.exp(-0.5 * d2 / _LINK_SD ** 2) / (2 * np.pi * _LINK_SD ** 2) + _FLOOR / _RANGE ** 2
        L = np.where(np.isfinite(lx)[None, :], -np.log(dens), np.log(_RANGE ** 2))
    G = np.zeros_like(C)                    # NGS Game Center terms (and the snap-spot chain, for carries)
    if kind == "carry" and "tk_players" in pf:
        G = G + _track_players_cost(cf["loc_x"].to_numpy(float), cf["loc_y"].to_numpy(float), list(pf["tk_players"]))
    if kind == "carry" and "tk_next_ball" in pf:
        G = G + _next_ball_cost(cf["loc_x"].to_numpy(float), pf["tk_next_ball"].to_numpy(float))
    if kind == "carry" and "tk_x" in pf:
        G = G + _track_end_cost(cf["loc_x"].to_numpy(float), cf["loc_y"].to_numpy(float),
                                pf["tk_x"].to_numpy(float), pf["tk_y"].to_numpy(float))
    if kind == "carry" and "bx" in pf and not _USE_SPOT:
        G = G + _chain_cost(cf["x0"].to_numpy(float), pf["bx"].to_numpy(float))
    if kind == "route" and _USE_NEXT_HASH_ROUTE and bucket == "COMPLETE" and "next_hash" in pf:
        # a completion ends where the receiver went down: the next snap's hash says which side that was
        G = G + _NEXT_W_ROUTE * _next_hash_cost(cf["end_x"].to_numpy(float), pf["next_hash"].to_numpy(object))
    if kind == "route" and _USE_NGS_DIST_ROUTE and bucket in ("COMPLETE", "TOUCHDOWN") and "ngs_dist" in pf and "path_len" in cf:
        # NGS's distance for the receiver: his whole path, route and run after the catch - the line's length
        nd = _ngs_dist_cost(cf["path_len"].to_numpy(float), pf["ngs_dist"].to_numpy(float),
                            np.full(len(pf), np.nan), table=_NGS_DIST_LR_ROUTE)
        G = G + _NGS_DIST_W_ROUTE * (nd - nd.mean(axis=1, keepdims=True))
    if kind == "pass" and _USE_NEXT_HASH_PASS and bucket == "COMPLETE" and "next_hash" in pf:
        G = G + _NEXT_W_PASS * _next_hash_cost(cf["x"].to_numpy(float), pf["next_hash"].to_numpy(object))
    if kind == "carry" and _USE_NEXT_HASH and "next_hash" in pf:
        nh = pf["next_hash"].to_numpy(object)
        if "tk_next_ball" in pf:              # tracking's spot for the next snap says it more precisely
            nh = np.where(np.isfinite(pf["tk_next_ball"].to_numpy(float)), None, nh)
        G = G + _NEXT_W * _next_hash_cost(cf["loc_x"].to_numpy(float), nh)
    if kind == "carry" and _USE_PAIR:
        # the learned (line, run) scorer (carry_pair.py): every chart feature and pbp column at once
        pm = [c for c in pf.columns if c.startswith("pm_")]
        if pm:
            G = G + _PAIR_W * _CP.pair_cost(cf, pf[pm].rename(columns=lambda c: c[3:]))
    if kind == "carry" and _USE_NGS_DIST and not _NGS_DIST_IN_C and "ngs_dist" in pf and "path_len" in cf:
        nd = _ngs_dist_cost(cf["path_len"].to_numpy(float), pf["ngs_dist"].to_numpy(float), pf["ngs_below"].to_numpy(float),
                            pf["gain"].to_numpy(float))
        G = G + _NGS_DIST_W * (nd - nd.mean(axis=1, keepdims=True))
    if kind == "carry" and _USE_SPOT and _SPOT_IN_G:
        bx = pf["bx"].to_numpy(float) if "bx" in pf else np.full(len(pf), np.nan)
        sc = _spot_cost(cf["x0"].to_numpy(float), pf["hash"].to_numpy(object), bx, pf["gun"].to_numpy(float))
        G = G + _SPOT_G_W * (sc - sc.mean(axis=1, keepdims=True))
    if kind == "pass" and "gc_imp" in pf:
        # NGS's air yards for this play as implied by its receiver's Game Center total minus his other targets
        imp, sdj = pf["gc_imp"].to_numpy(float), pf["gc_sd"].to_numpy(float)
        bj = np.sqrt(_GC_B ** 2 + 0.5 * np.nan_to_num(sdj) ** 2)
        g = _nll_laplace(cf["depth"].to_numpy(float)[:, None] - np.nan_to_num(imp)[None, :], bj[None, :])
        G = G + np.where(np.isfinite(imp)[None, :], g, _NLL_MISSING)
    if kind == "pass" and "zy_imp" in cf and "yds" in pf:
        # the yards gained this dot's play must have: its NGS zone's yards less what the zone's other dots took
        zi, zs = cf["zy_imp"].to_numpy(float), cf["zy_sd"].to_numpy(float)
        bz = np.sqrt(_ZONE_B ** 2 + 0.5 * np.nan_to_num(zs) ** 2)
        yds = pf["yds"].to_numpy(float)
        g = _nll_laplace(yds[None, :] - np.nan_to_num(zi)[:, None], bz[:, None])
        g = np.where(np.isfinite(yds)[None, :], g, _NLL_MISSING)
        G = G + np.where(np.isfinite(zi)[:, None], g, 0.0)
    # a chart entity with no usable measurement at all can still be placed, but gains nothing
    C = np.nan_to_num(C, nan=2 * _NLL_MISSING, posinf=2 * _NLL_MISSING)
    L = np.nan_to_num(L, nan=0.0, posinf=0.0)
    G = np.nan_to_num(G, nan=0.0, posinf=0.0)
    if split_link == 3:
        return C, L, G
    return (C, L + G) if split_link else C + L + G


_USE_DUR_NGS = False     # carries without FTN's clock: the duration NGS's distance and top speed imply - measured: no gain once the distance term is on
_DUR_NGS_COEF = (1.011, 1.411)
_DUR_NGS_DEFAULT = (0.0, 0.75)   # (offset, Laplace scale): the line's ~0.42 s and the proxy's ~0.42 s together


def _dur_cost(dur_hat, dur, params):
    """(n_chart, n_pbp): pbp's seconds for the play vs the seconds the line implies, a Laplace with a uniform floor
    over _DUR_RANGE; flat (no information) where either side is unknown."""
    off, b = params
    b = max(b, 0.2)
    d = np.nan_to_num(dur)[None, :] - np.nan_to_num(dur_hat)[:, None] - off
    cost = -np.log((1 - _FLOOR) * np.exp(-np.abs(d) / b) / (2 * b) + _FLOOR / _DUR_RANGE)
    known = np.isfinite(dur)[None, :] & np.isfinite(dur_hat)[:, None]
    return np.where(known, cost, np.log(_DUR_RANGE))


def _carry_type_cost(cf, pf):
    """-log likelihood ratio of each carry line's shape under each run's pbp gap and shotgun labels (0 where unknown)."""
    m = _CT_MODEL if isinstance(_CT_MODEL, dict) else _CT._load(_CT_MODEL)
    K = len(_CT.GAPS)
    out = np.zeros((len(cf), len(pf)))
    P = cf[[f"ct_gap{i}" for i in range(K)]].to_numpy(float)
    k = pf["ct_gap_lab"].to_numpy(float)
    known = k >= 0
    kk = np.where(known, k, 0).astype(int)
    prior = m["gap_prior"] if m is not None else np.full(K, 1.0 / K)
    p = (1 - _CT_EPS) * P[:, kk] + _CT_EPS / K
    out += np.where(known[None, :] & np.isfinite(P).all(1)[:, None], -_CT_W[0] * np.log(p / prior[kk][None, :]), 0.0)
    g = cf["ct_gun"].to_numpy(float)[:, None]
    s = pf["ct_gun_lab"].to_numpy(float)[None, :]
    gp = m["shotgun_prior"] if m is not None else np.array([0.5, 0.5])
    q = np.where(s == 1, g, 1 - g)
    q = (1 - _CT_EPS) * q + _CT_EPS / 2
    pri = np.where(s == 1, gp[1], gp[0])
    # where FTN recorded the snap's hash, the snap-spot term already scores the line's start against it (shotgun
    # offset included) and the shotgun read only double-counts it - measured against tracked tackle spots (2023-25)
    w = np.where(pd.notna(pf["hash"].to_numpy(object)), _CT_GUN_WITH_HASH, _CT_W[1]) if "hash" in pf else np.full(len(pf), _CT_W[1])
    out += np.where((s >= 0) & np.isfinite(g), -w[None, :] * np.log(q / pri), 0.0)
    return out


_USE_SCREEN = True       # routes: FTN is_screen_pass vs how screen-like the line is, where participation has no route
_SCREEN_Q = {1: 0.85, 0: 0.01}   # P(the route is a SCREEN | FTN's flag)
_SCREEN_W = 1.0


def _screen_cost(cf, pf):
    """-log likelihood ratio of each line's shape under each play's FTN screen flag, for plays without participation's
    route (the season in progress): the flag says SCREEN or not, the route model how screen-like the line is."""
    K = len(_RT.CLASSES)
    si = _RT.CLASSES.index("SCREEN")
    P = cf[[f"rt{i}" for i in range(K)]].to_numpy(float)
    m = _RT_MODEL if isinstance(_RT_MODEL, dict) else _RT._load()
    prior = m["prior"] if m is not None and "prior" in m else np.full(K, 1.0 / K)
    flag = pf["screen"].to_numpy(float)
    has_route = np.isfinite(pf["rt_class"].to_numpy(float)) if "rt_class" in pf else np.zeros(len(flag), bool)
    known = np.isfinite(flag) & ~has_route
    ratio = (1 - _RT_EPS) * P / prior[None, :] + _RT_EPS
    p_scr = ratio[:, si][:, None]
    p_oth = ((ratio * prior[None, :]).sum(1) - ratio[:, si] * prior[si])[:, None] / (1 - prior[si])
    q = np.where(flag == 1, _SCREEN_Q[1], _SCREEN_Q[0])[None, :]
    lr = -_SCREEN_W * np.log(q * p_scr + (1 - q) * p_oth)
    return np.where(known[None, :] & np.isfinite(P).all(1)[:, None], lr, 0.0)


def _route_type_cost(cf, pf):
    """-log likelihood ratio of each chart line's shape under each play's labelled route: how much more (or less)
    likely the model finds that route than it is a priori. 0 when either side has no information."""
    K = len(_RT.CLASSES)
    P = cf[[f"rt{i}" for i in range(K)]].to_numpy(float)
    k = pf["rt_class"].to_numpy(float)
    m = _RT_MODEL if isinstance(_RT_MODEL, dict) else _RT._load()
    prior = m["prior"] if m is not None and "prior" in m else np.full(K, 1.0 / K)
    known = np.isfinite(k)
    kk = np.where(known, k, 0).astype(int)
    p = (1 - _RT_EPS) * P[:, kk] + _RT_EPS / K
    lr = -_RT_W * np.log(p / prior[kk][None, :])
    return np.where(known[None, :] & np.isfinite(P).all(1)[:, None], lr, 0.0)


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


_CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pbp_calibration.json")


@functools.lru_cache(maxsize=1)
def _shipped_calibration():
    """{kind: {season: params}} fitted on every chart of each season (see params_to_json)."""
    try:
        with open(_CALIBRATION_FILE) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    return {kind: {int(s): params_from_json(p) for s, p in seasons.items()} for kind, seasons in raw.items()}


def params_to_json(params):
    """A params dict (match_to_pbp(..., return_params=True)) as plain JSON-able data."""
    out = {}
    for k, v in params.items():
        if k == "depth_tab":
            out[k] = {f"{src}|{bk}|{bd}": [float(a), float(b)] for (src, bk, bd), (a, b) in v.items()}
        elif isinstance(v, dict):
            out[k] = {lab: [float(a), float(b)] for lab, (a, b) in v.items()}
        elif isinstance(v, (tuple, list)):
            out[k] = [float(x) for x in v]
        else:
            out[k] = float(v)
    return out


def params_from_json(d):
    out = {}
    for k, v in d.items():
        if k == "depth_tab":
            out[k] = {(key.split("|")[0], key.split("|")[1], int(key.split("|")[2])): tuple(ab) for key, ab in v.items()}
        elif isinstance(v, dict):
            out[k] = {lab: tuple(ab) for lab, ab in v.items()}
        elif isinstance(v, list):
            out[k] = tuple(v)
        else:
            out[k] = v
    return out


def _league_params(kind, chart):
    """Shipped calibration for this chart data's season (or the nearest shipped season), else None."""
    shipped = _shipped_calibration().get(kind)
    if not shipped or "season" not in chart or not chart["season"].notna().any():
        return None
    season = int(chart["season"].mode().iloc[0])
    return shipped[min(shipped, key=lambda s: (abs(s - season), -s))]


_USE_CARRY_TYPE = True   # carries: the line's shape -> P(gap), P(shotgun) (carry_type.py) vs pbp run_gap / shotgun
_USE_GAP_TERM = False    # carries: the crossing point vs run_gap (the older, one-point version of the above)
_CT_MODEL = None         # carry_type model override (path or dict) - for held-out evaluation
_CT_W = (1.5, 1.5)       # weights of the gap / shotgun evidence
_CT_GUN_WITH_HASH = 0.0  # ... the shotgun evidence's weight on plays with a recorded hash (2022+)
_CT_EPS = 0.1
_USE_CHAIN = True        # carries: where the ball was snapped, chained from where the previous play ended
_CHAIN_W = (0.55, 0.45)  # share of chained spots that are right (penalties, chosen spots after a touchback... break it)
_CHAIN_NEAR = 0.5        # yds: a back lined up behind the QB takes the handoff this close to the ball's x
_CHAIN_GUN = (2.2, 0.8)  # yds: a back beside a shotgun QB, offset either side, and its spread
_CHAIN_GUN_SHARE = 0.4
_CHAIN_RANGE = 12.0      # yds a broken chain's spot is spread over
_HASH_X = 3.08           # yds from the middle of the field to a hash mark
_CARRY_OVER = ("qb_spike", "qb_kneel", "no_play")


def _play_ends(frames):
    """{(game_id, play_id): x where the play ended} from matched charts: carries and completed routes (end of the
    line), completed passes (the catch point - the run after it isn't drawn)."""
    ends = {}
    for d in frames or ():
        if d is None or not len(d):
            continue
        for row in d.itertuples(index=False):
            key = (row.game_id, row.play_id)
            px = getattr(row, "path_x", None)
            if px is not None and not (isinstance(px, float) and np.isnan(px)) and len(px) and \
                    getattr(row, "route_type", "COMPLETE") == "COMPLETE":
                ends[key] = float(px[-1])
            elif getattr(row, "pass_type", None) in ("COMPLETE", "TOUCHDOWN") and pd.notna(getattr(row, "x_coord", np.nan)):
                ends.setdefault(key, float(row.x_coord))
    return ends


def _snap_sequence(pbp_df, games):
    """Per game, the scrimmage snaps in order with what the chain needs: [(play_id, play_type, incomplete, sack,
    out_of_bounds, after_kick)]."""
    keep = ["game_id", "play_id", "play_type", "incomplete_pass", "sack", "out_of_bounds"]
    d = pbp_df[pbp_df["game_id"].isin(games)][[c for c in keep if c in pbp_df.columns]].sort_values(["game_id", "play_id"])
    seq = {}
    for g, grp in d.groupby("game_id"):
        rows, kick = [], False
        for r in grp.itertuples(index=False):
            if r.play_type not in ("run", "pass", "qb_kneel", "qb_spike", "no_play"):
                kick = True
                continue
            rows.append((r.play_id, r.play_type, getattr(r, "incomplete_pass", 0) == 1, getattr(r, "sack", 0) == 1,
                         getattr(r, "out_of_bounds", 0) == 1, kick))
            kick = False
        seq[g] = rows
    return seq


def _chain_spots(seq, ends):
    """{(game_id, play_id): ball x at the snap}, where it can be worked out: the previous play's end (on a hash if it
    ended outside them or out of bounds), or the same spot as the previous snap after an incompletion, spike, kneel,
    sack or no-play."""
    out = {}
    for g, rows in seq.items():
        bx = np.nan
        for i, (pid, ptype, inc, sack, oob, after_kick) in enumerate(rows):
            if i == 0 or after_kick:
                bx = np.nan                       # start of a drive: spot chosen by the offense, or unknown
            else:
                ppid, pptype, pinc, psack, poob, _ = rows[i - 1]
                if pptype in _CARRY_OVER or pinc or psack:
                    pass                          # ball goes back to where it was
                elif (g, ppid) in ends:
                    x = ends[(g, ppid)]
                    bx = np.sign(x) * _HASH_X if poob or abs(x) > _HASH_X else x
                else:
                    bx = np.nan
            out[(g, pid)] = bx
    return out


_USE_NEXT_HASH = True    # carries: where the line ends vs FTN's hash for the next snap (2022+) - a run that ends
                         # outside a hash puts the next snap on that hash, one that ends between them in the middle
_NEXT_HASH = (1.5, 0.5, 0.02)   # chart end x where the next snap's label turns from middle to a hash, logistic
                                # width, and the share of labels that don't follow (measured on 2023-25 carries)
_NEXT_W = 1.5
_USE_NEXT_HASH_ROUTE = True   # routes: where a completion's line ends vs FTN's hash for the next snap (2022+)
_NEXT_W_ROUTE = 1.0
_USE_NEXT_HASH_PASS = True    # passes: a completion's dot (the catch) vs the next snap's hash - the run after the
_NEXT_W_PASS = 0.5            # catch moves it, so weaker than a route line's end


def _next_snap_keys(pbp_df, games):
    """{(game_id, play_id): (play_id of the next scrimmage snap, same offense?)} for runs followed by a scrimmage
    snap with no kick or score in between."""
    cols = ["game_id", "play_id", "play_type", "posteam", "touchdown"]
    d = pbp_df[pbp_df["game_id"].isin(games)][[c for c in cols if c in pbp_df.columns]].sort_values(["game_id", "play_id"])
    out = {}
    for g, grp in d.groupby("game_id"):
        pending = None
        for r in grp.itertuples(index=False):
            if r.play_type in ("run", "pass", "qb_kneel", "qb_spike", "no_play"):
                if pending is not None:
                    out[(g, pending[0])] = (r.play_id, r.posteam == pending[1])
                pending = (r.play_id, r.posteam) if r.play_type == "run" and getattr(r, "touchdown", 0) != 1 else None
            elif isinstance(r.play_type, str):
                pending = None
    return out


def _next_snap_hash(pbp_df, games, kind="carry"):
    """{(game_id, play_id): FTN hash of the next snap, from this play's offense's view} for runs (kind="carry") or
    completed passes (kind="route") followed by a scrimmage snap with no kick or score in between."""
    if "starting_hash" not in pbp_df:
        return {}
    cols = ["game_id", "play_id", "play_type", "posteam", "starting_hash", "touchdown", "complete_pass", "interception"]
    d = pbp_df[pbp_df["game_id"].isin(games)][[c for c in cols if c in pbp_df.columns]].sort_values(["game_id", "play_id"])
    out = {}
    flip = {"L": "R", "R": "L", "M": "M"}
    for g, grp in d.groupby("game_id"):
        pending = None                     # a run still waiting for the next snap
        for r in grp.itertuples(index=False):
            if r.play_type in ("run", "pass", "qb_kneel", "qb_spike", "no_play"):
                if pending is not None:
                    h = r.starting_hash
                    if isinstance(h, str) and h in flip:
                        out[(g, pending[0])] = h if r.posteam == pending[1] else flip[h]
                ends_here = (r.play_type == "run") if kind == "carry" else \
                    (r.play_type == "pass" and getattr(r, "complete_pass", 0) == 1 and getattr(r, "interception", 0) != 1)
                pending = (r.play_id, r.posteam) if ends_here and getattr(r, "touchdown", 0) != 1 else None
            elif isinstance(r.play_type, str):
                pending = None             # a kick: the next snap is somewhere new
    return out


def _next_hash_cost(ex, labels):
    """(n_chart, n_pbp): -log likelihood ratio of each play's next-snap hash label given where each line ends."""
    c, w, eps = _NEXT_HASH
    ex = np.asarray(ex, float)[:, None]
    pl = eps + (1 - 3 * eps) / (1 + np.exp((ex + c) / w))
    pr = eps + (1 - 3 * eps) / (1 + np.exp(-(ex - c) / w))
    pm = 1 - pl - pr
    lab = np.asarray(labels, object)[None, :]
    p = np.where(lab == "L", pl, np.where(lab == "R", pr, pm))
    known = np.isin(lab, ["L", "M", "R"]) & np.isfinite(ex)
    ratio = p / np.maximum(np.where(known, p, np.nan).mean(axis=0, keepdims=True) if len(ex) else p, 1e-9)
    return np.where(known, -np.log(np.where(known, ratio, 1.0)), 0.0)


_USE_NGS_DIST = True     # carries: the line's length vs how far NGS says the ball carrier went (ngs_distance.py)
_NGS_DIST_W = 0.5
_NGS_DIST_IN_C = False   # with the untempered external terms, at half weight (it partly repeats the gain; measured best)
# log P(line length - NGS distance | the right run) / P(... | another run by the same back that game), measured on
# confident 2023-25 matches: most right pairs agree within a yard, but a line can run several yds short of NGS's
# distance (the back's steps before the handoff, a hidden line start) - so a measured ratio, not a Gaussian
_NGS_DIST_LR = ([-60, -15, -10, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7, 10, 15, 60],
                [-2.91, -2.13, -1.17, -0.35, 0.25, 0.55, 0.88, 2.14, 1.63, 0.12, -0.14, -0.60, -1.00, -1.60, -2.23, -2.85])
_USE_NGS_BELOW = False   # measured: hurts (2018-21-style -0.9 pt); kept switchable
# the same for completed routes (the receiver's whole line) vs another catch by the same receiver that game
_USE_NGS_DIST_ROUTE = True
_NGS_DIST_W_ROUTE = 0.5
_NGS_DIST_LR_ROUTE = ([-60, -15, -10, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7, 10, 15, 60],
                      [-2.22, -1.65, -0.94, -0.55, -0.10, 0.50, 1.36, 2.56, 1.89, 0.30, -0.10, -0.51, -0.75, -1.09, -1.47, -1.95])
_NGS_BELOW_SCALE = 1.2   # a play missing from a full list went less far than its shortest entry (84% of lines are)


def _ngs_dist_cost(L, dist, below, gain=None, table=None):
    """(n_chart, n_pbp) cost (-log likelihood ratio) of each line's length under each play's NGS ball-carrier
    distance - or, for a play the (full) list left out, under 'it went less far than the list's shortest'. 0 where
    NGS says nothing."""
    L = np.asarray(L, float)[:, None]
    dist, below = np.asarray(dist, float)[None, :], np.asarray(below, float)[None, :]
    edges, lr = (np.asarray(t, float) for t in (table or _NGS_DIST_LR))
    centres = np.clip((edges[:-1] + edges[1:]) / 2, -20, 20)
    r = L - np.nan_to_num(dist)
    cost = -np.interp(r, centres, lr)
    known = np.isfinite(dist) & np.isfinite(L)
    out = np.where(known, cost, 0.0)
    if _USE_NGS_BELOW:
        p_below = 1.0 / (1.0 + np.exp(-(np.nan_to_num(below) - L) / _NGS_BELOW_SCALE))
        bel = np.isfinite(below) & ~np.isfinite(dist) & np.isfinite(L)
        # P(line this long | went less far than the bound) vs the same for a typical other run (~half are)
        out = np.where(bel, -np.log((0.05 + 0.95 * p_below) / 0.5), out)
    return out


_USE_SPOT = True         # carries: one term for where the back started across the field (replaces hash + chain terms)
# where the back takes the handoff relative to the ball's x, by formation (measured on 2022-25 carries whose FTN hash
# put the ball on a hash): (weight, centre, Laplace scale) + a flat share over _SPOT_FLAT yds
_SPOT_UC = [(0.58, 0.0, 0.25), (0.30, 0.0, 1.6)]
_SPOT_GUN = [(0.18, 0.0, 0.3), (0.27, -2.2, 0.4), (0.27, 2.2, 0.4), (0.16, 0.0, 2.0)]
_SPOT_FLAT, _SPOT_FLAT_W = 16.0, 0.12
_SPOT_HASH_P = 0.455     # with no FTN hash: share of snaps on each hash (the rest between them)
_SPOT_G_W = 0.6
_SPOT_IN_G = True        # spot evidence enters untempered (like the chain) instead of with the pbp terms
_SPOT_FTN_W = 0.97       # how far FTN's hash is trusted


def _spot_kernel(d, gun):
    comps = _SPOT_GUN if gun else _SPOT_UC
    k = sum(w * np.exp(-np.abs(d - c) / b) / (2 * b) for w, c, b in comps)
    return k + _SPOT_FLAT_W / _SPOT_FLAT


_SPOT_GRID = np.linspace(-_HASH_X, _HASH_X, 25)


def _spot_cost(x0, hashes, bx, gun):
    """(n_chart, n_pbp) -log density of each carry's start x under each run: the ball at the snap (on FTN's hash, at
    the chained spot, or anywhere between the hashes), plus where a back lines up for that formation."""
    x0 = np.asarray(x0, float)[:, None]
    n = len(hashes)
    out = np.zeros((len(x0), n))
    for j in range(n):
        g = gun[j] == 1
        h = hashes[j]
        between = np.mean([_spot_kernel(x0[:, 0] - b, g) for b in _SPOT_GRID], axis=0)
        if h in ("L", "R"):
            on_hash = _spot_kernel(x0[:, 0] - (-_HASH_X if h == "L" else _HASH_X), g)
            p = _SPOT_FTN_W * on_hash + (1 - _SPOT_FTN_W) * between
        else:
            prior = between if h == "M" else (_SPOT_HASH_P * (_spot_kernel(x0[:, 0] + _HASH_X, g) + _spot_kernel(x0[:, 0] - _HASH_X, g))
                                              + (1 - 2 * _SPOT_HASH_P) * between)
            b = bx[j]
            p = _CHAIN_W[0] * _spot_kernel(x0[:, 0] - b, g) + _CHAIN_W[1] * prior if np.isfinite(b) else prior
        out[:, j] = -np.log(p)
    return out


def _chain_cost(x0, bx):
    """(n_chart, n_pbp) cost of each carry's start x given each play's chained snap spot (0-information where unknown)."""
    x0 = np.asarray(x0, float)[:, None]
    bx = np.asarray(bx, float)[None, :]
    d = x0 - np.nan_to_num(bx)
    lap = lambda z, b: np.exp(-np.abs(z) / b) / (2 * b)
    o, b = _CHAIN_GUN
    near = (1 - _CHAIN_GUN_SHARE) * lap(d, _CHAIN_NEAR) + _CHAIN_GUN_SHARE * 0.5 * (lap(d - o, b) + lap(d + o, b))
    dens = _CHAIN_W[0] * near + _CHAIN_W[1] / _CHAIN_RANGE
    cost = -np.log(dens) - np.log(_CHAIN_RANGE)          # 0 = no better than not knowing
    return np.where(np.isfinite(bx), cost, 0.0)


_USE_TRACK_END = True    # carries: tracking's end point of the run (tracking= - e.g. a tracked tackler), where known
_TRACK_END = (0.0, 1.2, 0.8, 0.9, 0.85)   # dx offset, dy offset (the tackler stands in front of the runner), x / y
                                          # Laplace scales, share that fit (the rest: tackler away from the ball)


# where a tracked player stands when a run ends, relative to where the runner (the chart line) ended: x offset, y
# offset, Laplace scales, share that fit (the rest: anywhere on the field) - fitted on confidently matched 2023-25
# carries. The tackler is at the runner; the others are a few yards off, defenders mostly in front of him.
_TRACK_KERNELS = {"tackler": (0.0, -1.15, 0.68, 0.82, 0.9), "def_DB": (0.0, 0.19, 4.27, 2.73, 0.73),
                  "def_DL": (0.0, -2.53, 4.39, 3.14, 0.85), "def_LB": (0.0, -2.1, 4.12, 2.72, 0.84),
                  "off_OL": (0.0, -2.34, 4.57, 3.14, 0.78), "other": (0.0, -1.5, 4.5, 3.0, 0.5)}
_TRACK_AREA = 53.3 * 40.0
_TRACK_OTHER_W = 1.0     # weight of the players who didn't make the tackle (they're correlated: pursuit converges)


_IMPLAUSIBLE_PROB = 0.08   # how often a carry flagged implausible is still in the right spot (2023-25 tracking)


def _implausible_carry(cf, pf, params, bucket):
    """A carry whose line can't be this play: its end is 4+ yds from the play's gain, or 2+ yds off and in the
    wrong colour - the assignment had to place it, but no play fits (usually a misread line: a short run hidden
    under a longer one, read as a branch of it). Right only ~8% of the time, against ~79% for the rest."""
    if bucket == "TOUCHDOWN" or not np.isfinite(pf["gain"]):
        return False
    off = params["gain"][0] + (params.get("gain_oob", (0.0, 0.0))[0] if pf.get("oob") == 1 else 0.0)
    res = abs(cf["gain"] - off - pf["gain"])
    alt = pf.get("gain_alt")
    if alt is not None and np.isfinite(alt):
        res = min(res, abs(cf["gain"] - off - alt))
    colour_bad = cf.get("gclass") is not None and pf.get("gclass") is not None and cf["gclass"] != pf["gclass"]
    return bool(res > 4 or (colour_bad and res > 2))


def _track_players_cost(ex, ey, players):
    """(n_chart, n_pbp): -log likelihood ratio (vs anywhere on the field) of each line's end under where every tracked
    player stood when each play ended. players[j]: (kernel keys, x, y) arrays, or None."""
    ex, ey = np.asarray(ex, float)[:, None], np.asarray(ey, float)[:, None]
    out = np.zeros((len(ex), len(players)))
    for j, pl in enumerate(players):
        if pl is None:
            continue
        for key, px, py in zip(*pl):
            ox, oy, bx, by, w = _TRACK_KERNELS.get(key, _TRACK_KERNELS["other"])
            dens = w * np.exp(-np.abs(px - ex - ox) / bx - np.abs(py - ey - oy) / by) / (4 * bx * by) + (1 - w) / _TRACK_AREA
            c = -np.log(dens * _TRACK_AREA)
            out[:, j:j + 1] += np.where(np.isfinite(c), c, 0.0) * (1.0 if key == "tackler" else _TRACK_OTHER_W)
    return out


_USE_TRACK_BALL = True   # tracked offensive linemen: where the ball was at the next snap, so where a run ended
_TRACK_BALL_B = 0.6      # Laplace scale (yds) of a line's end vs the next snap's ball x (a lineman's usual spacing away)
_TRACK_BALL_W = 0.9      # share that fit (penalties, re-spots, a lineman lined up somewhere new)


def _lineman_ball_x(tracking, pbp_df):
    """{(old_game_id, play_id): ball x at the snap} from tracked offensive linemen, each placed by his own usual
    spacing from the ball (learned on that season's snaps from a recorded hash)."""
    if "starting_hash" not in pbp_df:
        return {}
    sn = tracking[(tracking["moment"] == "snap") & tracking["offense"].astype(bool) & (tracking["group"] == "OL")]
    if not len(sn):
        return {}
    h = pbp_df[["old_game_id", "play_id", "starting_hash"]].assign(old_game_id=lambda d: d["old_game_id"].astype(str),
                                                                   play_id=lambda d: d["play_id"].astype(float))
    sn = sn.assign(old_game_id=sn["old_game_id"].astype(str), play_id=sn["play_id"].astype(float)).merge(h, on=["old_game_id", "play_id"], how="left")
    known = sn["starting_hash"].map({"L": -_HASH_X, "R": _HASH_X})
    off = (sn["x"] - known).groupby(sn["nfl_id"]).agg(["median", "count"])
    off = off[off["count"] >= 6]["median"]
    sn = sn[sn["nfl_id"].isin(off.index)]
    est = sn["x"] - sn["nfl_id"].map(off)
    est = est.groupby([sn["old_game_id"], sn["play_id"]]).median()
    # the ball is always between the hashes: anything else is a lineman lined up somewhere unusual (an extra tight end)
    return est[est.abs() <= _HASH_X + 0.3].to_dict()


def _next_ball_cost(ex, ball):
    """(n_chart, n_pbp): -log likelihood ratio of where each line ends (clamped to the hashes: where the ball is
    spotted after the run) given tracking's ball x at each play's next snap; 0 where unknown."""
    x = np.clip(np.asarray(ex, float), -_HASH_X, _HASH_X)
    ball = np.asarray(ball, float)
    d = x[:, None] - np.nan_to_num(ball)[None, :]
    dens = _TRACK_BALL_W * np.exp(-np.abs(d) / _TRACK_BALL_B) / (2 * _TRACK_BALL_B) + (1 - _TRACK_BALL_W) / (2 * _HASH_X + 4)
    c = -np.log(dens)
    known = np.isfinite(ball)[None, :] & np.isfinite(x)[:, None]
    c = np.where(known, c, np.nan)
    with np.errstate(all="ignore"):
        c = c - np.nanmean(c, axis=0, keepdims=True)
    return np.where(known, c, 0.0)


def _track_end_cost(ex, ey, tx, ty):
    """(n_chart, n_pbp) cost of each carry line's end under each run's tracked end point (0 where unknown)."""
    ox, oy, bx_, by_, w = _TRACK_END
    dx = np.asarray(ex, float)[:, None] - np.nan_to_num(np.asarray(tx, float))[None, :] - ox
    dy = np.asarray(ey, float)[:, None] - np.nan_to_num(np.asarray(ty, float))[None, :] - oy
    dens = w * np.exp(-np.abs(dx) / bx_ - np.abs(dy) / by_) / (4 * bx_ * by_) + (1 - w) / (_RANGE * _RANGE / 4)
    cost = -np.log(dens) - np.log(_RANGE * _RANGE / 4)
    return np.where(np.isfinite(np.asarray(tx, float))[None, :], cost, 0.0)


def match_to_pbp(chart_df, pbp_df, kind, route_matches=None, return_params=False, calibration="auto", context=None,
                 tracking=None, pass_matches=None):
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

    calibration: how each piece of evidence's offset and spread are set.
      - "auto" (default): the league-wide calibration shipped with the package for the chart data's season
        (the nearest season if that one isn't shipped). Matching one player gives exactly what matching the
        whole league at once gives for that player.
      - "self": fit it on this data (the plays that can only match one way, then the confident matches).
        Fine for a whole season of charts; for one player it has few plays to go on and comes out less sure.
      - a dict from match_to_pbp(..., return_params=True): use those.

    tracking (carries, optional): a DataFrame of old_game_id, play_id, end_x, end_y - where tracking puts the end of
        the run, in chart coordinates (see next_gen_scrapy.bdb for the Big Data Bowl's tracked tacklers).

    context (carries, optional): match_to_pbp output for the same games' other charts - the QB's passes, receivers'
    routes, other backs' carries. A carry starts where the ball was snapped, which is where the previous play ended
    (or the same spot, after an incompletion): those charts say where that was.

    route_matches (pass only, optional): the output of match_to_pbp(..., 'route') for the same
    games. A targeted receiver's route chart shows the same throw a second time (its catch point),
    so for plays whose target has a route chart that point is used as extra evidence of where the
    ball went. Match routes first, then pass them in here.

    pass_matches (route only, optional): the output of match_to_pbp(..., 'pass') for the same games - the other way
    round: the QB's chart shows each completion as a dot, a second measurement of the route's catch point. Match the
    passes without route_matches for this, so the two charts stay independent.

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
        the model being graded doesn't use (_TEMP): passes and routes against each other's chart,
        carries against FTN's starting hash and participation's formation.
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
    if kind == "carry" and tracking is not None and len(tracking) and _USE_TRACK_END and "moment" in tracking:
        # every tracked player at the end of each play (next_gen_scrapy.bdb.load_play_tracking)
        e = tracking[tracking["moment"] == "end"]
        key = np.where(e["tackler"].astype(bool), "tackler",
                       np.where(e["offense"].astype(bool), "off_", "def_") + e["group"].astype(str))
        e = pd.DataFrame({"g": e["old_game_id"].astype(str).to_numpy(), "p": e["play_id"].astype(float).to_numpy(),
                          "k": key, "x": e["x"].to_numpy(float), "y": e["y"].to_numpy(float)})
        by_play = {k: (g["k"].to_numpy(), g["x"].to_numpy(), g["y"].to_numpy()) for k, g in e.groupby(["g", "p"])}
        keys = list(zip(pbp_typed["old_game_id"].astype(str), pbp_typed["play_id"].astype(float)))
        pfeat_all["tk_players"] = [by_play.get(k) for k in keys]
        if _USE_TRACK_BALL:
            ball = _lineman_ball_x(tracking, pbp_df)
            nxt = _next_snap_keys(pbp_df, set(chart_reduced["game_id_nflverse"].dropna()))
            og = dict(zip(zip(pbp_df["game_id"], pbp_df["play_id"]), pbp_df["old_game_id"].astype(str)))
            nb = []
            for g, q in zip(pbp_typed["game_id"], pbp_typed["play_id"]):
                n = nxt.get((g, q))
                v = ball.get((og.get((g, n[0]), ""), float(n[0])), np.nan) if n else np.nan
                nb.append(v if n is None or n[1] else -v)
            pfeat_all["tk_next_ball"] = nb
    elif kind == "carry" and tracking is not None and len(tracking) and _USE_TRACK_END:
        tk = tracking.assign(old_game_id=tracking["old_game_id"].astype(str), play_id=tracking["play_id"].astype(float))
        tk = tk.drop_duplicates(["old_game_id", "play_id"]).set_index(["old_game_id", "play_id"])
        key = list(zip(pbp_typed["old_game_id"].astype(str), pbp_typed["play_id"].astype(float)))
        pfeat_all["tk_x"] = [tk["end_x"].get(k, np.nan) for k in key]
        pfeat_all["tk_y"] = [tk["end_y"].get(k, np.nan) for k in key]
    if (kind == "carry" and _USE_NEXT_HASH) or (kind == "route" and _USE_NEXT_HASH_ROUTE) or (kind == "pass" and _USE_NEXT_HASH_PASS):
        nh = _next_snap_hash(pbp_df, set(chart_reduced["game_id_nflverse"].dropna()), kind)
        pfeat_all["next_hash"] = [nh.get((g, q)) for g, q in zip(pbp_typed["game_id"], pbp_typed["play_id"])]
    if kind == "route" and pass_matches is not None and len(pass_matches):
        pm = pass_matches[pass_matches["pass_type"].isin(["COMPLETE", "TOUCHDOWN"])]
        link = pd.DataFrame({"game_id": pm["game_id"].values, "play_id": pm["play_id"].values,
                             "link_x": pm["x_coord"].values, "link_y": pm["y_coord"].values}).drop_duplicates(["game_id", "play_id"])
        keyed = pbp_typed[["game_id", "play_id"]].reset_index().merge(link, on=["game_id", "play_id"], how="left")
        pfeat_all["link_x"] = keyed.set_index("index")["link_x"]
        pfeat_all["link_y"] = keyed.set_index("index")["link_y"]
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
                    pairs.append((cfeat_all.loc[c_sub.index[0]], pfeat_all.loc[p_sub.index[0]], bucket))
    if isinstance(calibration, dict):
        fixed = calibration
    elif calibration == "auto":
        fixed = _league_params(kind, chart_reduced)
    elif calibration == "self":
        fixed = None
    else:
        raise ValueError(f"calibration must be 'auto', 'self' or a params dict, not {calibration!r}")
    params = fixed if fixed is not None else _calibrate(kind, pairs)

    def assign(params):
        """One matching pass: per group, the optimal assignment plus each pair's location confidence."""
        out = []
        for game_id, bucket, p_sub, c_sub in groups:
            cf, pf = cfeat_all.loc[c_sub.index], pfeat_all.loc[p_sub.index]
            C0, CL, CG = _cost_matrix(kind, cf, pf, params, bucket, split_link=3)
            C = C0 + CL + CG
            T = _TEMP_BUCKET.get((kind, bucket), _TEMP[kind])
            if kind == "carry" and pf["hash"].notna().any():
                T = _TEMP_CARRY_WITH_HASH
            if kind == "pass" and T == _TEMP["pass"] and not pf["ngs"].notna().any():
                T = _TEMP_PASS_NO_NGS
            # the temperature corrects the pbp-evidence terms only; the route-chart link is a measured
            # geometric likelihood and enters the probabilities as it is
            rows, cols = linear_sum_assignment(C)
            M = _assignment_marginals(C0 / T + CL + _GC_PROB_SCALE * CG)
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
    if _TWO_PASS and fixed is None:
        # The forced 1:1 pairs are mostly touchdowns (goal-line runs, end-zone catches), which skews
        # the calibration for ordinary plays. Re-fit every term on the first pass's confident matches,
        # across all outcomes, and match again.
        confident = [(cfeat_all.loc[ci], pfeat_all.loc[pi], bk) for _g, bk, _n, ci, pi, _c, _m, _s, lp, _mp in assigned
                     if lp >= 0.95]
        if len(confident) >= _MIN_CALIB:
            params = _calibrate(kind, pairs + confident)
            assigned = assign(params)

    use_gc = _USE_GC and kind == "pass" and "gc_key" in pfeat_all and pfeat_all["gc_key"].notna().any()
    use_zones = _USE_ZONES and kind == "pass" and "zones" in pfeat_all and pfeat_all["zones"].notna().any()
    if use_gc or use_zones:
        # NGS Game Center totals: a receiver's targets must add up to his air yards, a zone's completions to its
        # yards. Each round every pass gets what the rest of its receiver's targets / its zone leave for it, and
        # the assignment is redone - until nothing changes
        prev = None
        for _ in range(_GC_ITERS):
            if use_gc:
                _set_implied_depths(pfeat_all, cfeat_all, assigned)
            if use_zones:
                _set_zone_yards(pfeat_all, cfeat_all, chart_reduced, assigned)
            assigned = assign(params)
            now = [(a[3], a[4]) for a in assigned]
            if now == prev:
                break
            prev = now

    if _USE_CHAIN and kind == "carry" and len(assigned):
        # each round: where every play ended (other charts + this assignment), so where every snap was, then redo
        seq = _snap_sequence(pbp_df, set(chart_reduced["game_id_nflverse"].dropna()))
        base_ends = _play_ends(context if isinstance(context, (list, tuple)) else [context])
        x_end = {i: px[-1] for i, px in zip(chart_reduced.index, chart_reduced["path_x"])}
        prev = None
        for _ in range(_GC_ITERS):
            ends = dict(base_ends)
            for g, _bk, _n, ci, pi, *_ in assigned:
                ends[(g, pbp_df.at[pi, "play_id"])] = float(x_end[ci])
            spots = _chain_spots(seq, ends)
            pfeat_all["bx"] = [spots.get((g, q), np.nan) for g, q in zip(pbp_typed["game_id"], pbp_typed["play_id"])]
            assigned = assign(params)
            now = [(a[3], a[4]) for a in assigned]
            if now == prev:
                break
            prev = now

    # ---- build the output rows ----
    if kind in ("pass", "route"):
        ngs_games = set(pbp_typed.loc[pfeat_all.index[pfeat_all["ngs"].notna()], "game_id"])
    if kind == "carry":
        hash_games = set(pbp_typed.loc[pfeat_all.index[pd.notna(pfeat_all["hash"])], "game_id"]) \
            if pfeat_all["hash"].notna().any() else set()
    matches = []
    yard_col = "yards_gained" if kind == "carry" else "air_yards"
    chart_yard = cfeat_all["gain"] if kind == "carry" else cfeat_all["depth"]
    for game_id, bucket, n_cand, chart_idx, pbp_idx, cost, margin, swap, loc_prob, match_prob in assigned:
        if kind == "pass" and n_cand > 1:          # the only candidate play is certain
            knots = np.array(_LOC_CALIBRATION[game_id in ngs_games])
            loc_prob = float(np.interp(loc_prob, knots[:, 0], knots[:, 1]))
        if kind == "route" and n_cand > 1 and game_id not in ngs_games:
            knots = np.array(_ROUTE_LOC_CALIBRATION)
            loc_prob = float(np.interp(loc_prob, knots[:, 0], knots[:, 1]))
        if kind == "carry" and n_cand > 1:
            knots = np.array(_CARRY_LOC_CALIBRATION[game_id in hash_games])
            loc_prob = float(np.interp(loc_prob, knots[:, 0], knots[:, 1]))
        chart_row = chart_reduced.loc[chart_idx].to_dict()
        pv, cv = pbp_df.loc[pbp_idx, yard_col], chart_yard.loc[chart_idx]
        implausible = False
        if kind == "carry":
            implausible = _implausible_carry(cfeat_all.loc[chart_idx], pfeat_all.loc[pbp_idx], params, bucket)
            if implausible:                        # the assignment had to put this line somewhere; it fits nothing
                loc_prob, match_prob = min(loc_prob, _IMPLAUSIBLE_PROB), min(match_prob, _IMPLAUSIBLE_PROB)
        matches.append(dict(
            kind=kind, bucket=bucket, game_id=game_id, play_id=pbp_df.loc[pbp_idx, "play_id"],
            n_candidates=n_cand, match_cost=round(cost, 3),
            match_margin=round(margin, 3) if np.isfinite(margin) else None,
            swap_yds=round(swap, 2), location_prob=round(loc_prob, 3), match_prob=round(match_prob, 3),
            location_confident=bool(loc_prob >= 0.95), **({"implausible": implausible} if kind == "carry" else {}),
            residual=round(abs(float(pv) - float(cv)), 2) if pd.notna(pv) and pd.notna(cv) else None,
            **{f"chart_{k}": v for k, v in chart_row.items() if k not in ("game_id_nflverse", "gsis_id")}
        ))

    matched = pd.DataFrame(matches)
    if len(matched):
        matched = matched.rename(columns={f"chart_{c}": c for c in _PROMOTE_COLS if f"chart_{c}" in matched.columns})
        matched = _attach_pbp_columns(matched, pbp_df)
    return (matched, params) if return_params else matched


def _set_implied_depths(pf_all, cf_all, assigned):
    """
    For every pass to a receiver NGS Game Center lists: the NGS air yards it must have, given his NGS total and what
    the current assignment says about his other targets (their dots, less _GC_OFF; pbp air yards for targets with no
    dot here). A target whose dot is uncertain counts with a wider spread. Sets pf_all gc_imp (in chart-depth units)
    and gc_sd.
    """
    dot = {pi: (ci, lp, bk) for _g, bk, _n, ci, pi, _c, _m, _s, lp, _mp in assigned}
    imp = pd.Series(np.nan, index=pf_all.index)
    sd = pd.Series(np.nan, index=pf_all.index)
    for _key, idx in pf_all.groupby("gc_key").groups.items():
        g = pf_all.loc[idx]
        n = g["gc_n"].iloc[0]
        if len(idx) != n:                 # pbp and NGS disagree on his targets: don't trust the sum
            continue
        est, var = np.empty(len(idx)), np.empty(len(idx))
        for t, pi in enumerate(idx):
            if pi in dot:
                ci, lp, bk = dot[pi]
                est[t] = cf_all.at[ci, "depth"] - _GC_OFF
                var[t] = 2 * _GC_B ** 2 + (1 - lp) * 9.0 + (9.0 if bk == "INTERCEPTION" else 0.0)
            else:
                a = g.at[pi, "air"]
                est[t] = a if np.isfinite(a) else g["gc_avg"].iloc[0]
                var[t] = (_GC_SD_AIR[0] if g.at[pi, "complete"] == 1 else _GC_SD_AIR[1]) ** 2 + (0 if np.isfinite(a) else 25)
        total = n * g["gc_avg"].iloc[0]
        imp.loc[idx] = total - (est.sum() - est) + _GC_OFF
        sd.loc[idx] = np.sqrt(np.maximum(var.sum() - var, 0.0))
    pf_all["gc_imp"], pf_all["gc_sd"] = imp, sd


def _zone_of(depth, x):
    """NGS Game Center zone (depth band, third) of a pass dot."""
    a = np.asarray(depth, float) - _GC_OFF
    return np.select([a < 0, a < 10, a < 20], [0, 1, 2], 3), np.select([np.asarray(x) < -_ZONE_THIRD, np.asarray(x) <= _ZONE_THIRD], [0, 1], 2)


_DEPTH_CUTS = (0.0, 10.0, 20.0)


def _repair_zones(zone, dep, xs, zt, max_move=2.0):
    """
    Where one zone has a dot more than NGS counts and a neighbouring zone one fewer, the extra dot is one NGS placed
    across the boundary: move the dot closest to that boundary (if within max_move yards). Edits `zone` in place.
    """
    for _ in range(len(zone)):
        count = {}
        for zk in zone.values():
            count[zk] = count.get(zk, 0) + 1
        keys = set(count) | set(zt)
        surplus = [k for k in keys if count.get(k, 0) > zt.get(k, (0, 0))[0]]
        deficit = {k for k in keys if count.get(k, 0) < zt.get(k, (0, 0))[0]}
        best = None
        for sk in surplus:
            for dk in deficit:
                if abs(sk[0] - dk[0]) + abs(sk[1] - dk[1]) != 1:
                    continue
                for ci, zk in zone.items():
                    if zk != sk:
                        continue
                    if sk[0] != dk[0]:      # across a depth cut
                        dist = abs(dep[ci] - _DEPTH_CUTS[min(sk[0], dk[0])])
                    else:                   # across a third's edge
                        dist = abs(abs(xs[ci]) - _ZONE_THIRD)
                    if dist <= max_move and (best is None or dist < best[0]):
                        best = (dist, ci, dk)
        if best is None:
            return
        zone[best[1]] = best[2]


def _set_zone_yards(pf_all, cf_all, chart, assigned):
    """
    For every completion dot in an NGS Game Center zone whose drawn completions match NGS's count: the yards gained
    its play must have - the zone's yards less those of the plays the current assignment gives its other dots (an
    uncertain one counts with a wider spread). Sets cf_all zy_imp / zy_sd. Zones whose count doesn't match (a dot
    near a boundary NGS put on the other side) are left out.
    """
    play = {ci: (pi, mp) for _g, _bk, _n, ci, pi, _c, _m, _s, _lp, mp in assigned}
    zones = {}
    for key, z in pf_all.dropna(subset=["zones"]).groupby("qb_key")["zones"].first().items():
        zones[key] = {(int(d), int(q)): (int(c), int(y)) for d, q, c, y in (t.split(",") for t in z.split(";"))}
    imp = pd.Series(np.nan, index=cf_all.index)
    sd = pd.Series(np.nan, index=cf_all.index)
    dots = chart[chart["pass_type"].isin(["COMPLETE", "TOUCHDOWN"])]
    for (g, qb), grp in dots.groupby(["game_id_nflverse", "gsis_id"]):
        zt = zones.get(f"{g}|{qb}")
        if not zt:
            continue
        dep = cf_all.loc[grp.index, "depth"].to_numpy(float) - _GC_OFF
        xs = cf_all.loc[grp.index, "x"].to_numpy(float)
        dz, sz = _zone_of(dep + _GC_OFF, xs)
        zone = {ci: (int(d), int(q)) for ci, d, q in zip(grp.index, dz, sz)}
        _repair_zones(zone, dict(zip(grp.index, dep)), dict(zip(grp.index, xs)), zt)
        members = {}
        for ci, zk in zone.items():
            members.setdefault(zk, []).append(ci)
        for zkey, ks in members.items():
            comp, total = zt.get(zkey, (0, 0))
            if comp != len(ks) or any(k not in play for k in ks):
                continue
            yd = np.array([pf_all.at[play[k][0], "yds"] for k in ks], float)
            if not np.isfinite(yd).all():
                continue
            var = np.array([(1 - play[k][1]) * 64.0 for k in ks])
            imp.loc[ks] = total - (yd.sum() - yd)
            sd.loc[ks] = np.sqrt(np.maximum(var.sum() - var, 0.0))
    cf_all["zy_imp"], cf_all["zy_sd"] = imp, sd


_UNMATCHED_NATS = 12.0   # a chart line with no play left to pair with (a reading with the wrong split)


def add_snap_context(pbp_df):
    """pbp with the next snap's FTN hash per run (`next_hash_carry`) and per completion (`next_hash_route`), so
    code that sees only one player's plays (a chart reader choosing between readings) can use it like match_to_pbp."""
    out = pbp_df.copy()
    games = set(out["game_id"].dropna())
    for kind, col in (("carry", "next_hash_carry"), ("route", "next_hash_route")):
        nh = _next_snap_hash(out, games, kind)
        out[col] = [nh.get((g, q)) for g, q in zip(out["game_id"], out["play_id"])]
    return out


def line_fit(chart_rows, pbp_rows, kind, params, link=None):
    """{line id: cost (nats) of that chart line under the play it fits best} - every line scored on its own, no
    one-to-one assignment. For a reader choosing between ways to trace a tangled chart: a piece that fits none of
    the player's plays is a piece traced wrong."""
    red = to_paths(chart_rows, kind)
    idc = {"route": "route_id", "carry": "carry_id"}[kind]
    cfeat, pfeat = _chart_features(red, kind), _pbp_features(pbp_rows, kind)
    if link is not None and len(link):
        k = pbp_rows[["game_id", "play_id"]].reset_index().merge(link, on=["game_id", "play_id"], how="left").set_index("index")
        pfeat["link_x"], pfeat["link_y"] = k["link_x"], k["link_y"]
    out = {}
    for bucket, (p_sub, c_sub) in _buckets(kind, pbp_rows, red).items():
        if len(c_sub) and len(p_sub):
            C = _cost_matrix(kind, cfeat.loc[c_sub.index], pfeat.loc[p_sub.index], params, bucket)
            for i, v in zip(red.loc[c_sub.index, idc], C.min(axis=1)):
                out[int(i)] = float(v)
    return out


def score_chart(chart_rows, pbp_rows, kind, params, link=None):
    """
    How well one player-game chart reading fits that player's plays in pbp_rows (already restricted to
    the same game and player): the total negative log-likelihood of the best one-to-one assignment,
    plus a flat penalty for chart lines left without a play. Lower = better. Used to choose between
    alternative readings of a tangled chart (see next_gen_scrapy.retrace). link (routes, optional): DataFrame
    game_id, play_id, link_x, link_y - the QB's pass-chart dot for each completion, scored as in match_to_pbp.
    """
    red = to_paths(chart_rows, kind)
    if kind == "pass":
        red = red[red["x_coord"].notna() & red["y_coord"].notna()]
    cfeat, pfeat = _chart_features(red, kind), _pbp_features(pbp_rows, kind)
    if link is not None and len(link):
        k = pbp_rows[["game_id", "play_id"]].reset_index().merge(link, on=["game_id", "play_id"], how="left").set_index("index")
        pfeat["link_x"], pfeat["link_y"] = k["link_x"], k["link_y"]
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
