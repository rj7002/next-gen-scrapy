"""The curated set of filters the explorer exposes (and the natural-language parser may emit).

The database keeps every pbp/FTN column for export, but most of them aren't worth filtering on
(per-play lateral/tackler ids, running team EPA totals, ...). Each entry here is one filter a person
would actually reach for, with a label, a group, and the control it renders as:

  choice - a few fixed values, multi-select        -> ?col__in=a,b
  select - many values, single pick                -> ?col=a
  bool   - yes / no                                -> ?col=true|false
  range  - numeric min / max                       -> ?col__gte=x&col__lte=y
"""

from __future__ import annotations

ALL = ("passes", "routes", "carries")
PASSING = ("passes", "routes")

GROUPS = ["Situation", "Play call", "Passing", "Running", "Defense", "Result", "Player & game", "Match quality"]

# FTN charting only exists from 2022 on - flagged so the UI can say so.
FTN = "2022+"


def F(col, label, group, kind, tables=ALL, choices=None, since=None, step=None, help=None, panel=True):
    """panel=False: an Ask-bar-only filter - not in the filter panel, and its column isn't in the play rows the app
    loads; the app fetches it (/api/columns) when a question uses it."""
    return {"col": col, "label": label, "group": group, "kind": kind, "tables": tables,
            "choices": choices, "since": since, "step": step, "help": help, "panel": panel}


FILTERS = [
    # Situation
    F("down", "Down", "Situation", "choice", choices=[[1, "1st"], [2, "2nd"], [3, "3rd"], [4, "4th"]]),
    F("ydstogo", "Yards to go", "Situation", "range"),
    F("qtr", "Quarter", "Situation", "choice", choices=[[1, "Q1"], [2, "Q2"], [3, "Q3"], [4, "Q4"], [5, "OT"]]),
    F("yardline_100", "Yards from end zone", "Situation", "range", help="20 or less = red zone"),
    F("goal_to_go", "Goal to go", "Situation", "bool"),
    F("starting_hash", "Ball on hash", "Situation", "choice", choices=[["L", "Left"], ["M", "Middle"], ["R", "Right"]]),
    F("score_differential", "Score margin (offense)", "Situation", "range", help="negative = trailing"),
    F("half_seconds_remaining", "Seconds left in half", "Situation", "range"),
    F("wp", "Win probability", "Situation", "range", step=0.01),

    # Play call
    F("shotgun", "Shotgun", "Play call", "bool"),
    F("no_huddle", "No huddle", "Play call", "bool"),
    F("is_play_action", "Play action", "Play call", "bool", since=FTN),
    F("is_motion", "Pre-snap motion", "Play call", "bool", since=FTN),
    F("is_rpo", "RPO", "Play call", "bool", since=FTN),
    F("is_screen_pass", "Screen", "Play call", "bool", tables=PASSING, since=FTN),
    F("is_trick_play", "Trick play", "Play call", "bool", since=FTN),
    F("qb_location", "QB alignment", "Play call", "choice", since=FTN,
      choices=[["S", "Shotgun"], ["U", "Under center"], ["P", "Pistol"]]),
    F("n_offense_backfield", "Players in backfield", "Play call", "range", since=FTN),
    F("offense_formation", "Formation", "Play call", "choice"),
    F("offense_personnel", "Personnel", "Play call", "select"),

    # Passing
    F("air_yards", "Air yards", "Passing", "range", tables=PASSING, help="20+ = deep"),
    F("receiver_player_name", "Thrown to", "Passing", "select", tables=("passes",)),
    F("tree_route", "Route (tree)", "Passing", "choice", tables=PASSING,
      choices=[[r, r.title()] for r in ("GO", "POST", "CORNER", "OUT", "IN", "SLANT", "HITCH", "CROSS", "FLAT", "SCREEN",
                                        "WHEEL", "ANGLE")],
      help="the route tree's route: NGS's label for the target's route where there is one, else read from the line"),
    F("receiver_position", "Target's position", "Passing", "choice", tables=("passes",),
      choices=[["WR", "WR"], ["TE", "TE"], ["RB", "RB"], ["FB", "FB"]]),
    F("pass_length", "Pass length", "Passing", "choice", tables=PASSING,
      choices=[["short", "Short"], ["deep", "Deep"]]),
    F("pass_location", "Pass direction", "Passing", "choice", tables=PASSING,
      choices=[["left", "Left"], ["middle", "Middle"], ["right", "Right"]]),
    F("is_qb_out_of_pocket", "QB out of pocket", "Passing", "bool", tables=PASSING, since=FTN),
    F("qb_hit", "QB hit", "Passing", "bool", tables=PASSING),
    F("is_throw_away", "Throwaway", "Passing", "bool", tables=("passes",), since=FTN),
    F("target_route", "Target's route (NGS)", "Passing", "choice", tables=PASSING),
    F("time_to_throw", "Time to throw (s)", "Passing", "range", tables=PASSING, step=0.1),
    F("cp", "Completion probability", "Passing", "range", tables=PASSING, step=0.05,
      help="nflverse's chance this throw is completed, from where and how it was thrown"),

    # Running
    F("run_location", "Run direction", "Running", "choice", tables=("carries",),
      choices=[["left", "Left"], ["middle", "Middle"], ["right", "Right"]]),
    F("run_gap", "Run gap", "Running", "choice", tables=("carries",),
      choices=[["guard", "Guard"], ["tackle", "Tackle"], ["end", "End"]]),
    F("qb_scramble", "QB scramble", "Running", "bool", tables=("carries",)),
    F("is_qb_sneak", "QB sneak", "Running", "bool", tables=("carries",), since=FTN),

    # Defense
    F("defteam", "Opponent", "Defense", "select"),
    F("defense_coverage_type", "Coverage", "Defense", "choice", tables=PASSING),
    F("defense_man_zone_type", "Man / zone", "Defense", "choice", tables=PASSING),
    F("was_pressure", "QB pressured", "Defense", "bool", tables=PASSING),
    F("n_defense_box", "Defenders in box", "Defense", "range", since=FTN),
    F("n_pass_rushers", "Pass rushers", "Defense", "range", tables=PASSING, since=FTN),
    F("n_blitzers", "Blitzers", "Defense", "range", tables=PASSING, since=FTN),
    F("defense_look", "Defensive look", "Defense", "choice", since=FTN,
      choices=[["BASE", "Base (4 DBs)"], ["NICKEL", "Nickel (5 DBs)"], ["DIME", "Dime (6+ DBs)"]]),

    # Result
    F("bucket", "Outcome", "Result", "choice", choices=[
        ["COMPLETE", "Complete"], ["INCOMPLETE", "Incomplete"], ["TOUCHDOWN", "Touchdown"],
        ["INTERCEPTION", "Interception"], ["OTHER", "Non-TD"]]),
    F("gain_class", "Gain", "Result", "choice", tables=("carries",),
      choices=[["LONG", "5+ yds"], ["SHORT", "0-5 yds"], ["LOSS", "Loss"]]),
    F("yards_gained", "Yards gained", "Result", "range"),
    F("yards_after_catch", "Yards after catch", "Result", "range", tables=PASSING, help="completions only"),
    F("epa", "EPA", "Result", "range", step=0.1),
    F("success", "Successful play", "Result", "bool", help="positive EPA"),
    F("first_down", "Gained first down", "Result", "bool"),
    F("penalty", "Penalty on play", "Result", "bool"),
    F("fixed_drive_result", "Drive result", "Result", "choice", choices=[[v, v] for v in (
        "Touchdown", "Field goal", "Punt", "Turnover", "Turnover on downs", "Missed field goal", "End of half",
        "Opp touchdown", "Safety")]),
    F("fumble_lost", "Fumble lost", "Result", "bool", tables=("carries",)),
    F("is_catchable_ball", "Catchable", "Result", "bool", tables=PASSING, since=FTN),
    F("is_contested_ball", "Contested", "Result", "bool", tables=PASSING, since=FTN),
    F("is_drop", "Drop", "Result", "bool", tables=PASSING, since=FTN),
    F("is_created_reception", "Created reception", "Result", "bool", tables=PASSING, since=FTN),
    F("is_interception_worthy", "Interception-worthy", "Result", "bool", tables=("passes",), since=FTN),

    # Player & game
    F("position", "Position", "Player & game", "choice"),
    F("season_type", "Season type", "Player & game", "choice",
      choices=[["REG", "Regular season"], ["POST", "Playoffs"]]),
    F("posteam_type", "Offense is", "Player & game", "choice", choices=[["home", "Home"], ["away", "Away"]]),
    F("div_game", "Division game", "Player & game", "bool"),
    F("roof", "Roof", "Player & game", "choice"),
    F("temp", "Temperature (°F)", "Player & game", "range"),
    F("wind", "Wind (mph)", "Player & game", "range"),

    # Match quality
    F("location_prob", "Location probability", "Match quality", "range", step=0.05,
      help="chance the drawn location is within 3 yds of where this play happened"),
    F("n_candidates", "Candidate plays", "Match quality", "range",
      help="1 = the only play with this player + outcome in the game, so the match is certain"),
    F("start_ok", "Route start traced cleanly", "Match quality", "bool", tables=("routes",)),
    F("handoff_ok", "Handoff traced cleanly", "Match quality", "bool", tables=("carries",)),
    F("gap", "Includes an undrawn jump into a TD ring", "Match quality", "bool", tables=("routes", "carries"),
      help="NGS drew the TD ring but no line into it; the line was joined to it in a straight jump"),
]

PENALTY_TYPES = [
    'Offensive Holding',
    'False Start',
    'Defensive Pass Interference',
    'Defensive Holding',
    'Delay of Game',
    'Unnecessary Roughness',
    'Defensive Offside',
    'Illegal Formation',
    'Face Mask',
    'Illegal Contact',
    'Roughing the Passer',
    'Neutral Zone Infraction',
    'Offensive Pass Interference',
    'Illegal Use of Hands',
    'Intentional Grounding',
    'Illegal Block Above the Waist',
    'Illegal Shift',
    'Ineligible Downfield Pass',
    'Encroachment',
    'Unsportsmanlike Conduct',
    'Defensive Too Many Men on Field',
    'Illegal Blindside Block',
    'Taunting',
    'Offensive Offside',
    'Tripping',
    'Low Block',
    'Illegal Motion',
    'Offensive Too Many Men on Field',
    'Illegal Forward Pass',
    'Disqualification',
    'Chop Block',
    'Leverage',
    'Illegal Touch Pass',
    'Defensive Delay of Game',
    'Horse Collar Tackle',
    'Illegal Double-Team Block',
    'Hip Drop Tackle',
    'Illegal Bat',
    'Personal Foul',
    'Clipping',
]


# Everything else worth asking about: the Ask bar can filter on these, the panel doesn't list them.
def A(col, label, group, kind, tables=ALL, choices=None, step=None, help=None):
    return F(col, label, group, kind, tables, choices, step=step, help=help, panel=False)


FILTERS += [
    # situation
    A("posteam_timeouts_remaining", "Offense timeouts left", "Situation", "range"),
    A("defteam_timeouts_remaining", "Defense timeouts left", "Situation", "range"),
    A("posteam_score", "Offense's score", "Situation", "range"),
    A("defteam_score", "Defense's score", "Situation", "range"),
    A("game_seconds_remaining", "Seconds left in the game", "Situation", "range"),
    A("quarter_seconds_remaining", "Seconds left in the quarter", "Situation", "range"),
    A("drive", "Drive number (game)", "Situation", "range"),
    A("drive_play_count", "Plays in the drive", "Situation", "range"),
    A("drive_first_downs", "First downs in the drive", "Situation", "range"),
    A("drive_start_transition", "Drive started after", "Situation", "choice", choices=[[v, v.replace("_", " ").title()] for v in (
        "KICKOFF", "PUNT", "INTERCEPTION", "FUMBLE", "DOWNS", "MISSED_FG", "MUFFED_PUNT", "ONSIDE_KICK", "BLOCKED_FG",
        "BLOCKED_PUNT")]),
    A("drive_ended_with_score", "Scoring drive", "Situation", "bool"),
    A("drive_inside20", "Drive reached the red zone", "Situation", "bool"),
    A("series_result", "Series result", "Situation", "choice", choices=[[v, v] for v in (
        "First down", "Touchdown", "Field goal", "Punt", "Turnover", "Turnover on downs", "Missed field goal",
        "End of half", "QB kneel", "Opp touchdown", "Safety")]),
    A("ep", "Expected points before the snap", "Situation", "range", step=0.1),
    A("xpass", "Pass probability", "Situation", "range", step=0.05, help="nflverse's chance the offense passes here"),
    A("team_favored", "Team was favored", "Situation", "bool", help="by the betting line"),
    A("total_line", "Over / under", "Situation", "range"),
    A("surface", "Playing surface", "Player & game", "choice", choices=[[v, v] for v in (
        "grass", "fieldturf", "matrixturf", "sportturf", "astroturf", "a_turf")]),
    # result
    A("third_down_converted", "Third down converted", "Result", "bool"),
    A("third_down_failed", "Third down failed", "Result", "bool"),
    A("fourth_down_converted", "Fourth down converted", "Result", "bool"),
    A("fourth_down_failed", "Fourth down failed", "Result", "bool"),
    A("first_down_penalty", "First down by penalty", "Result", "bool"),
    A("penalty_type", "Penalty", "Result", "select", choices=[[v, v] for v in PENALTY_TYPES]),
    A("penalty_on", "Penalty on", "Result", "choice", choices=[["offense", "Offense"], ["defense", "Defense"]]),
    A("penalty_yards", "Penalty yards", "Result", "range"),
    A("penalty_player_name", "Penalized player", "Result", "select"),
    A("tackled_for_loss", "Tackled for loss", "Result", "bool"),
    A("out_of_bounds", "Out of bounds", "Result", "bool"),
    A("fumble", "Fumbled", "Result", "bool"),
    A("fumble_forced", "Forced fumble", "Result", "bool"),
    A("return_touchdown", "Returned for a touchdown", "Result", "bool", tables=PASSING),
    A("lateral_reception", "Lateral", "Result", "bool", tables=PASSING),
    A("replay_or_challenge", "Reviewed / challenged", "Result", "bool"),
    A("sp", "Scoring play", "Result", "bool"),
    A("wpa", "Win probability added", "Result", "range", step=0.01),
    A("qb_epa", "QB EPA", "Result", "range", step=0.1),
    A("air_epa", "Air EPA", "Passing", "range", tables=PASSING, step=0.1),
    A("yac_epa", "YAC EPA", "Passing", "range", tables=PASSING, step=0.1),
    A("cpoe", "CPOE", "Passing", "range", tables=PASSING, help="completion % over expected, per throw"),
    A("xyac_mean_yardage", "Expected YAC", "Passing", "range", tables=PASSING, step=0.5),
    A("yac_over_expected", "YAC over expected", "Passing", "range", tables=PASSING, step=0.5),
    A("read_thrown", "Read thrown", "Passing", "choice", tables=PASSING, choices=[
        ["1", "1st read"], ["2", "2nd read"], ["CHK", "Checkdown"], ["SD", "Scramble drill"], ["DES", "Designed throw"]]),
    # players involved
    A("tackler", "Tackled by", "Player & game", "select"),
    A("interception_player_name", "Intercepted by", "Player & game", "select", tables=PASSING),
    A("pass_defender", "Pass defended by", "Player & game", "select", tables=PASSING),
    A("qb_hitter", "QB hit by", "Player & game", "select", tables=PASSING),
    A("tfl_by", "Tackled for loss by", "Player & game", "select"),
    A("forced_fumble_by", "Fumble forced by", "Player & game", "select"),
    A("passer_player_name", "Thrown by", "Player & game", "select", tables=("routes",)),
    # NGS tracking
    A("ngs_top_speed", "Ball carrier top speed (mph)", "Result", "range", step=0.5),
    A("ngs_carrier_dist", "Distance the ball carrier ran (yds)", "Result", "range"),
]

BY_COL = {f["col"]: f for f in FILTERS}


def for_table(table: str) -> list[dict]:
    return [f for f in FILTERS if table in f["tables"]]
