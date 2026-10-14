"""The Ask bar's rules for the filters beyond the panel's (filters.py's panel=False ones, plus a few panel ones that
only had model coverage): how people say them, matched on whatever the main rules left unconsumed.

catalog_rules(rp) runs early in RuleParser.run - these phrases are specific enough not to steal from the main
rules, and some contain words the main rules would otherwise take ("third down conversions", "tackled for a loss").
"""
from __future__ import annotations

import re

from .filters import PENALTY_TYPES

WORDNUM = {"zero": 0, "no": 0, "one": 1, "a": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
           "forty": 40, "fifty": 50}
N = r"(\d{1,3}(?:\.\d+)?|" + "|".join(sorted(WORDNUM, key=len, reverse=True)) + r")"


def num(s: str) -> float:
    return float(WORDNUM[s]) if s in WORDNUM else float(s)


GE = r"(?:at least|over|more than|above|greater than|>=?)"
PLAY = r"(?:plays?|throws?|passes|runs?|carries|catches|targets?|completions|receptions|snaps|moments)"
LE = r"(?:under|less than|fewer than|below|at most|<=?)"

# yes / no columns: (phrases meaning yes, phrases meaning no)
BOOL = {
    "penalty": (r"penalt(?:y|ies)|flags?|flagged|called back|drew a flag",
                r"no (?:penalty|penalties|flags?)|without (?:a )?(?:penalty|flag)|clean plays?|unflagged"),
    "third_down_converted": (r"(?:3rd|third)[ -]down conversions?|converted (?:on )?(?:3rd|third)(?: downs?)?|"
                             r"(?:3rd|third) downs? (?:he|they) converted|moved the chains on (?:3rd|third)", None),
    "third_down_failed": (r"failed (?:3rd|third)[ -]downs?|(?:3rd|third)[ -]down (?:stops|failures)|"
                          r"(?:came up )?short on (?:3rd|third)(?: down)?|didn'?t convert on (?:3rd|third)", None),
    "fourth_down_converted": (r"(?:4th|fourth)[ -]down conversions?|converted (?:on )?(?:4th|fourth)(?: downs?)?", None),
    "fourth_down_failed": (r"failed (?:4th|fourth)[ -]downs?|turnovers? on downs|(?:4th|fourth)[ -]down stops", None),
    "first_down_penalty": (r"first downs? (?:by|from|via|on) (?:a )?penalt(?:y|ies)|penalty first downs?", None),
    "tackled_for_loss": (r"tackled for (?:a )?loss|tackles? for (?:a )?loss|tfls?", None),
    "out_of_bounds": (r"pushed out|knocked out of bounds|(?:ran|went|stepped|got) out of bounds|stepped out|out of bounds",
                      r"(?:stayed )?in bounds|didn'?t go out"),
    "fumble_lost": (r"fumbled it away|coughed it up|coughed up the ball|lost (?:a |the )?fumbles?|fumbles? lost|turned it over on a fumble", None),
    "fumble": (r"(?<!lost )(?<!forced )fumbl(?:e|es|ed|ing)(?!\s+(?:lost|forced|it away|away))|put the ball on the (?:ground|turf)", None),
    "fumble_forced": (r"forced fumbles?|fumbles? forced|(?:ball )?(?:punched|stripped) (?:out|loose)|strip(?:ped)?", None),
    "return_touchdown": (r"pick[ -]?six(?:es)?|returned for (?:a )?(?:touchdown|td|score)|taken back for (?:a )?(?:touchdown|td|score)", None),
    "lateral_reception": (r"laterals?|hook and ladders?|hook and laterals?", None),
    "replay_or_challenge": (r"challenged|reviewed|under review|replay reviews?|(?:coach'?s )?challenges?", None),
    "drive_ended_with_score": (r"scoring drives?|drives? that (?:ended in|led to|produced) points|drives? that scored",
                               r"drives? that (?:came up empty|didn'?t score|ended without points)|empty drives?"),
    "drive_inside20": (r"drives? that (?:reached|got to|made it to) the red zone|red zone trips?", None),
    "team_favored": (r"as (?:the )?favou?rites?|when (?:they were )?favou?red|as (?:the )?favou?red team",
                     r"as (?:an? |the )?underdogs?|when (?:they were )?(?:the )?underdogs?|as (?:a )?dog"),
}

# columns with fixed values: {value: phrases}
CHOICE = {
    "defense_look": {
        "NICKEL": r"(?:against|vs\.?|versus|facing|out of|in) (?:a |the )?nickel(?: (?:defense|package|personnel|look))?|nickel (?:defense|package|looks?)",
        "DIME": r"(?:against|vs\.?|versus|facing|out of|in) (?:a |the )?dime(?: (?:defense|package|personnel|look))?|dime (?:defense|package|looks?)",
        "BASE": r"(?:against|vs\.?|versus|facing) (?:a |the )?base(?: (?:defense|personnel|package|look))?|base (?:defense|personnel)",
    },
    "starting_hash": {
        "L": r"(?:from|on|off) the left hash|left hash(?:mark)?", "R": r"(?:from|on|off) the right hash|right hash(?:mark)?",
        "M": r"(?:from|on) the middle of the field(?: hash)?|middle hash|ball in the middle",
    },
    "surface": {
        "grass": r"on (?:natural |real )?grass|on a grass field",
        "fieldturf,matrixturf,sportturf,astroturf,a_turf": r"on (?:artificial |fake )?turf|on a turf field|on fake grass|on astro ?turf",
    },
    "read_thrown": {
        "1": r"(?:to )?(?:his|the) (?:1st|first|primary) read|(?:1st|first)[ -]read throws?|primary reads?",
        "2": r"(?:to )?(?:his|the) (?:2nd|second) read|(?:2nd|second)[ -]read throws?|(?:off|past) (?:his|the) first read|went through (?:his )?progressions?",
        "SD": r"(?:in|out of) (?:the )?scramble drill|scramble drill throws?",
        "DES": r"designed (?:throws|passes|plays)|schemed[ -]up throws",
    },
    "penalty_on": {
        "defense": r"defensive penalt(?:y|ies)|penalt(?:y|ies) on the defen[cs]e|flags? on the defen[cs]e",
        "offense": r"offensive penalt(?:y|ies)|penalt(?:y|ies) on the offen[cs]e|flags? on the offen[cs]e",
    },
    "fixed_drive_result": {
        "Touchdown": r"(?:on |during )?(?:touchdown|td) drives?|drives? that (?:ended in|finished with) (?:a )?(?:touchdown|td)",
        "Field goal": r"(?:on |during )?field goal drives?|drives? that (?:ended in|finished with) (?:a )?field goal",
        "Punt": r"drives? that (?:ended in|finished with) (?:a )?punt|punt drives?",
        "Turnover": r"drives? that (?:ended in|finished with) (?:a )?turnover",
        "Turnover on downs": r"drives? that (?:ended in|finished with) (?:a )?turnover on downs",
    },
    "drive_start_transition": {
        "INTERCEPTION,FUMBLE": r"(?:on )?(?:drives? )?(?:after|off|following) (?:a |an )?(?:turnover|takeaway)s?|on short fields? after (?:a )?turnovers?",
        "INTERCEPTION": r"(?:on )?(?:drives? )?(?:after|off|following) (?:an )?interceptions?|(?:after|off) (?:a )?picks?",
        "FUMBLE": r"(?:on )?(?:drives? )?(?:after|off|following) (?:a )?fumbles?(?: recovery)?",
        "PUNT": r"(?:on )?(?:drives? )?(?:after|off|following) (?:a )?punts?",
        "KICKOFF": r"(?:on )?(?:drives? )?(?:after|off|following) (?:a )?kickoffs?",
        "DOWNS": r"(?:on )?(?:drives? )?(?:after|off|following) (?:a )?(?:stop on downs|turnover on downs)",
        "MISSED_FG": r"(?:on )?(?:drives? )?(?:after|off|following) (?:a )?missed (?:field goal|fg)",
    },
}

# numbers: (noun phrase after a number / before a comparison, column, scale)
RANGE = [
    (r"(?:mph|miles? (?:an|per) hour)", "ngs_top_speed", 1),
    (r"(?:play|plays)[ -](?:long )?drives?|plays? drives?", "drive_play_count", 1),
    (r"(?:penalty yards|yards? (?:penalty|penalties|flags?))", "penalty_yards", 1),
    (r"(?:expected points|ep)", "ep", 1),
    (r"(?:% |percent )?completion probability|cp", "cp", 0.01),
    (r"(?:% |percent )?(?:pass|passing) probability|xpass", "xpass", 0.01),
    (r"(?:% |percent )?(?:wpa|win probability added)", "wpa", 0.01),
    (r"(?:% |percent )?cpoe|completion percentage over expected", "cpoe", 1),
    (r"(?:expected yac|xyac|expected yards after (?:the )?catch)", "xyac_mean_yardage", 1),
    (r"(?:yac over expected|yards after catch over expected|yacoe)", "yac_over_expected", 1),
    (r"(?:air epa)", "air_epa", 1),
    (r"(?:yac epa)", "yac_epa", 1),
    (r"(?:qb epa)", "qb_epa", 1),
    (r"(?:point total|over/?under|total line)", "total_line", 1),
    (r"first downs? (?:on|in) the drive|first[ -]down drives?", "drive_first_downs", 1),
    (r"yards? (?:of distance|covered|run|traveled|travelled)", "ngs_carrier_dist", 1),
]

# words that mean a number-free extreme of a range column: (phrase, column, op, value)
RANGE_WORDS = [
    (r"(?:tough|difficult|hard|low[ -]percentage|low[ -]probability) (?:throws|passes|completions|catches|targets)", "cp", "lte", 0.4),
    (r"(?:easy|high[ -]percentage|high[ -]probability|layup|gimme) (?:throws|passes|completions|catches|targets)", "cp", "gte", 0.75),
    (r"(?:obvious |clear )?passing situations|obvious passing downs|(?:when|where) (?:everyone|the defense) knew (?:it was|they'?d) pass", "xpass", "gte", 0.7),
    (r"(?:obvious |clear )?(?:running|run) situations|obvious (?:running|run) downs", "xpass", "lte", 0.35),
    (rf"(?:game[ -](?:changing|swinging|winning)|high[ -]leverage|swing|big[ -]wpa|clutch[ -]wpa) {PLAY}", "wpa", "gte", 0.05),
    (r"(?:beat (?:the )?expected yac|more yac than expected|yac over expected|extra yards? after (?:the )?catch)", "yac_over_expected", "gt", 0),
    (r"(?:long|sustained|marathon) drives?", "drive_play_count", "gte", 10),
    (r"(?:high[ -](?:scoring|total) games?|expected shootouts?|shootouts? (?:on paper|by the line))", "total_line", "gte", 48),
    (r"(?:low[ -](?:scoring|total) games?|expected (?:slugfests?|defensive battles?))", "total_line", "lte", 41),
    (r"(?:full speed|top speed|fastest) plays?|(?:hit|reached|topped) top speed", "ngs_top_speed", "gte", 18),
]

# players involved: {column: phrases that come right before the player's name}
PLAYERS = {
    "tackler": r"tackled by|brought down by|taken down by|stopped by|wrapped up by|tackles? (?:made )?by",
    "interception_player_name": r"picked off by|intercepted by|picks? by|interceptions? (?:by|from)|(?:pick|interception) thrown to",
    "pass_defender": r"broken up by|defended by|deflected by|batted (?:down )?by|pbus? by|pass breakups? by|knocked away by|defensed by",
    "qb_hitter": r"(?:he was |when he was |getting )?hit by|hits? (?:from|by)|drilled by|smacked by|crushed by",
    "tfl_by": r"tackled for (?:a )?loss by|tfls? by|stuffed by|blown up by",
    "forced_fumble_by": r"(?:fumble |ball )?(?:forced|punched out|stripped|knocked loose) by|strips? by",
    "penalty_player_name": r"penalt(?:y|ies) on|flags? on|flagged on|called on",
    "passer_player_name": r"thrown by|passes? from|throws? from|targets? from|catches? from|with (?=[a-z]+ (?:at|under) (?:qb|quarterback))",
}

# the clock: "under 2 minutes left in the game", "final 5 minutes of the quarter"
CLOCK = rf"(?:{LE}|inside|within|in the (?:final|last)|(?:final|last))\s+{N}\s+(minutes?|mins?|seconds?|secs?)\s+(?:left\s+|remaining\s+)?(?:in|of)\s+the\s+(game|quarter|qtr|4th|fourth)"


def catalog_rules(rp) -> None:
    """Run the catalog's rules on parser rp (a nl.RuleParser): players first (the most specific), then names of
    penalties, the clock, timeouts, fixed values, yes / no columns and numbers."""
    dyn = rp.dyn or {}
    choices = dyn.get("choices") or {}

    def setter(col, op, value):
        return lambda m: rp.set(col, op, value)

    # players involved: "<verb phrase> <name>", the name matched against the names on this player's plays
    def by_name(names, words):
        last, first = words[-1], (words[0] if len(words) == 2 else "")
        return [n for n in names if "." in n and re.sub(r"[^a-z]", "", n.split(".", 1)[1].lower()) == last
                and (not first or n.lower().startswith(first[0]))]
    for col, verbs in PLAYERS.items():
        names = sorted(choices.get(col, ()), key=len, reverse=True)
        if not names:
            continue
        for two in (True, False):
            pat = r"([a-z][a-z.'\-]*)\s+([a-z][a-z.'\-]*)" if two else r"([a-z][a-z.'\-]*)"

            def hit(m, names=names, col=col):
                found = by_name(names, [re.sub(r"[^a-z]", "", g) for g in m.groups()])
                return rp.set(col, "in", found) if found else False
            rp.rule(rf"\b(?:{verbs})\s+(?:the\s+)?{pat}\b", hit)

    # penalty yards: "15 yard penalties", "5-yard flags"
    rp.rule(rf"\b{N}[ -]?(?:yard|yd)s? (?:penalt(?:y|ies)|flags?|infractions?)\b",
            lambda m: rp.sets(("penalty_yards", "gte", num(m.group(1))), ("penalty_yards", "lte", num(m.group(1)))))

    # penalty types by name (every type the league calls, + common shorthand)
    types = sorted(set(PENALTY_TYPES) | set(choices.get("penalty_type", ())), key=len, reverse=True)
    short = {"dpi": "Defensive Pass Interference", "opi": "Offensive Pass Interference", "pi": "Pass Interference",
             "pass interference": "Pass Interference", "holding": "Holding", "roughing": "Roughing",
             "false starts?": "False Start", "offsides?": "Offside", "face ?masks?": "Face Mask"}
    for t in types:
        rp.rule(rf"\b{re.escape(t.lower())}s?(?: penalt(?:y|ies)| flags?| calls?)?\b", setter("penalty_type", "in", [t]))
    for word, part in short.items():
        hit = [t for t in types if part.lower() in t.lower()]
        if hit:
            rp.rule(rf"\b{word}(?: penalt(?:y|ies)| flags?| calls?)?\b", setter("penalty_type", "in", hit))

    def clock(m):
        n = num(m.group(1)) * (60 if m.group(2).startswith("min") else 1)
        col = "quarter_seconds_remaining" if m.group(3).startswith(("quarter", "qtr")) else "game_seconds_remaining"
        return rp.set(col, "lte", n)
    rp.rule(rf"\b{CLOCK}\b", clock)

    # timeouts: "with no timeouts left", "out of timeouts", "with 2 timeouts"; the defense's when it says so
    rp.rule(r"\b(?:the )?(?:defen[cs]e|opponents?|they) (?:had|with) no timeouts(?: left)?\b|\b(?:defen[cs]e|opponents?) (?:was |were )?out of timeouts\b",
            setter("defteam_timeouts_remaining", "lte", 0))
    rp.rule(r"\b(?:with )?no timeouts(?: left| remaining)?\b|\bout of timeouts\b", setter("posteam_timeouts_remaining", "lte", 0))
    rp.rule(rf"\bwith (?:all )?{N} timeouts?(?: left| remaining)?\b",
            lambda m: rp.sets(("posteam_timeouts_remaining", "gte", num(m.group(1))), ("posteam_timeouts_remaining", "lte", num(m.group(1)))))

    for col, values in CHOICE.items():
        # the longest phrases first, so "turnover on downs" isn't read as "turnover"
        for value, phrases in sorted(values.items(), key=lambda kv: -len(kv[1])):
            rp.rule(rf"\b(?:{phrases})\b", setter(col, "in", value.split(",")))

    for col, (yes, no) in BOOL.items():
        if no:
            rp.rule(rf"\b(?:{no})\b", setter(col, "eq", "false"))
        rp.rule(rf"\b(?:(?:no|non|not|without)[ -])(?:{yes})\b", setter(col, "eq", "false"))
        rp.rule(rf"\b(?:{yes})\b", setter(col, "eq", "true"))

    # numbers on the catalog's range columns: "N+ <noun>", "<cmp> N <noun>", "<noun> <cmp> N"
    rp.rule(rf"\b(?:hit|reached|topped|got (?:up )?to|clocked(?: in at)?|ran)\s+{N}\s*(?:mph|miles? (?:an|per) hour)\b",
            lambda m: rp.set("ngs_top_speed", "gte", num(m.group(1))))
    for noun, col, scale in RANGE:
        def make(op, col=col, scale=scale):
            def h(m):
                v = num(next(g for g in m.groups() if g))
                return rp.set(col, op, round(v * scale, 3) if scale != 1 and v > 1 else v)
            return h
        rp.rule(rf"\b{N}\s*(?:\+|or more|plus)\s*(?:{noun})\b", make("gte"))
        rp.rule(rf"(?<![a-z]){GE}\s*{N}\s*(?:{noun})\b|\b(?:{noun})\s*(?:of\s+)?{GE}\s*{N}\b", make("gte"))
        rp.rule(rf"(?<![a-z]){LE}\s*{N}\s*(?:{noun})\b|\b(?:{noun})\s*(?:of\s+)?{LE}\s*{N}\b", make("lte"))
    for phrase, col, op, value in RANGE_WORDS:
        rp.rule(rf"\b(?:{phrase})\b", setter(col, op, value))
