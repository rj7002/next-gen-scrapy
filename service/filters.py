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


def F(col, label, group, kind, tables=ALL, choices=None, since=None, step=None, help=None):
    return {"col": col, "label": label, "group": group, "kind": kind, "tables": tables,
            "choices": choices, "since": since, "step": step, "help": help}


FILTERS = [
    # Situation
    F("down", "Down", "Situation", "choice", choices=[[1, "1st"], [2, "2nd"], [3, "3rd"], [4, "4th"]]),
    F("ydstogo", "Yards to go", "Situation", "range"),
    F("qtr", "Quarter", "Situation", "choice", choices=[[1, "Q1"], [2, "Q2"], [3, "Q3"], [4, "Q4"], [5, "OT"]]),
    F("yardline_100", "Yards from end zone", "Situation", "range", help="20 or less = red zone"),
    F("goal_to_go", "Goal to go", "Situation", "bool"),
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
    F("pass_length", "Pass length", "Passing", "choice", tables=PASSING,
      choices=[["short", "Short"], ["deep", "Deep"]]),
    F("pass_location", "Pass direction", "Passing", "choice", tables=PASSING,
      choices=[["left", "Left"], ["middle", "Middle"], ["right", "Right"]]),
    F("is_qb_out_of_pocket", "QB out of pocket", "Passing", "bool", tables=PASSING, since=FTN),
    F("qb_hit", "QB hit", "Passing", "bool", tables=PASSING),
    F("is_throw_away", "Throwaway", "Passing", "bool", tables=("passes",), since=FTN),
    F("target_route", "Target's route (NGS)", "Passing", "choice", tables=PASSING),
    F("time_to_throw", "Time to throw (s)", "Passing", "range", tables=PASSING, step=0.1),

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

    # Result
    F("bucket", "Outcome", "Result", "choice", choices=[
        ["COMPLETE", "Complete"], ["INCOMPLETE", "Incomplete"], ["TOUCHDOWN", "Touchdown"],
        ["INTERCEPTION", "Interception"], ["OTHER", "Non-TD"]]),
    F("gain_class", "Gain", "Result", "choice", tables=("carries",),
      choices=[["LONG", "5+ yds"], ["SHORT", "0-5 yds"], ["LOSS", "Loss"]]),
    F("yards_gained", "Yards gained", "Result", "range"),
    F("epa", "EPA", "Result", "range", step=0.1),
    F("success", "Successful play", "Result", "bool", help="positive EPA"),
    F("first_down", "Gained first down", "Result", "bool"),
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
    F("location_confident", "Location confident", "Match quality", "bool",
      help="95%+ likely the drawn location belongs to this play (look-alike plays swapped on top of each other don't count)"),
    F("location_prob", "Location probability", "Match quality", "range", step=0.05,
      help="chance the drawn location is within 3 yds of where this play happened"),
    F("n_candidates", "Candidate plays", "Match quality", "range",
      help="1 = the only play with this player + outcome in the game, so the match is certain"),
    F("start_ok", "Route start traced cleanly", "Match quality", "bool", tables=("routes",)),
    F("handoff_ok", "Handoff traced cleanly", "Match quality", "bool", tables=("carries",)),
]

BY_COL = {f["col"]: f for f in FILTERS}


def for_table(table: str) -> list[dict]:
    return [f for f in FILTERS if table in f["tables"]]
