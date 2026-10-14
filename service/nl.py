"""Natural-language question -> explorer filters.

Rules first: a deterministic pass over the question that recognises football phrasing ("3rd and
short", "red zone", "play action", "vs the Cowboys", "weeks 1-5", player names, ...) and turns each
phrase into a curated filter (service/filters.py). Every character it uses is marked consumed; any
meaningful words left over mean the rules weren't sure, and only then is Gemini asked (see
service/gemini.py) - with the rules' draft as context, and its answer validated against the same
curated filter list, so it can refine the parse but never invent a column or value.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .filters import BY_COL, for_table

DATASETS = ("passes", "routes", "carries")
OP_SUFFIX = {"eq": "", "in": "__in", "gte": "__gte", "lte": "__lte", "gt": "__gt", "lt": "__lt", "ne": "__ne"}
OP_SYMBOL = {"eq": "=", "in": "=", "gte": "≥", "lte": "≤", "gt": ">", "lt": "<", "ne": "≠"}

# (chart `team` code, pbp `posteam`/`defteam` code, aliases)
TEAMS = [
    ("AZ", "ARI", ["cardinals", "arizona"]), ("ATL", "ATL", ["falcons", "atlanta"]),
    ("BAL", "BAL", ["ravens", "baltimore"]), ("BUF", "BUF", ["bills", "buffalo"]),
    ("CAR", "CAR", ["panthers", "carolina"]), ("CHI", "CHI", ["bears", "chicago"]),
    ("CIN", "CIN", ["bengals", "cincinnati"]), ("CLE", "CLE", ["browns", "cleveland"]),
    ("DAL", "DAL", ["cowboys", "dallas"]), ("DEN", "DEN", ["broncos", "denver"]),
    ("DET", "DET", ["lions", "detroit"]), ("GB", "GB", ["packers", "green bay"]),
    ("HOU", "HOU", ["texans", "houston"]), ("IND", "IND", ["colts", "indianapolis"]),
    ("JAX", "JAX", ["jaguars", "jags", "jacksonville"]), ("KC", "KC", ["chiefs", "kansas city"]),
    ("LAC", "LAC", ["chargers"]), ("LAR", "LA", ["rams"]),
    ("LV", "LV", ["raiders", "las vegas", "oakland"]), ("MIA", "MIA", ["dolphins", "miami"]),
    ("MIN", "MIN", ["vikings", "minnesota"]), ("NE", "NE", ["patriots", "pats", "new england"]),
    ("NO", "NO", ["saints", "new orleans"]), ("NYG", "NYG", ["giants"]), ("NYJ", "NYJ", ["jets"]),
    ("PHI", "PHI", ["eagles", "philadelphia", "philly"]), ("PIT", "PIT", ["steelers", "pittsburgh"]),
    ("SEA", "SEA", ["seahawks", "seattle"]), ("SF", "SF", ["49ers", "niners", "san francisco"]),
    ("TB", "TB", ["buccaneers", "bucs", "tampa bay", "tampa"]), ("TEN", "TEN", ["titans", "tennessee"]),
    ("WAS", "WAS", ["commanders", "washington", "redskins", "football team"]),
]
CHART_TO_PBP = {c: p for c, p, _ in TEAMS}
PBP_TO_CHART = {p: c for c, p, _ in TEAMS}
ALIAS_TO_CHART = {a: c for c, _, aliases in TEAMS for a in aliases}
ALL_CODES = sorted(set(CHART_TO_PBP) | set(PBP_TO_CHART), key=len, reverse=True)

ORD = {"1st": 1, "first": 1, "2nd": 2, "second": 2, "3rd": 3, "third": 3, "4th": 4, "fourth": 4}
ORD_RE = r"(?:1st|first|2nd|second|3rd|third|4th|fourth)"
NUM = r"(\d{1,3}(?:\.\d+)?)"

POSITIONS = {
    "qb": "QB", "qbs": "QB", "quarterback": "QB", "quarterbacks": "QB",
    "rb": "RB", "rbs": "RB", "running back": "RB", "running backs": "RB", "hb": "HB", "halfback": "HB",
    "halfbacks": "HB", "fb": "FB", "fullback": "FB", "fullbacks": "FB",
    "wr": "WR", "wrs": "WR", "wide receiver": "WR", "wide receivers": "WR", "receiver": "WR", "receivers": "WR",
    "te": "TE", "tes": "TE", "tight end": "TE", "tight ends": "TE",
}

STOPWORDS = set("""
a an the of in on for with by and or to at from his her their its that which who whom when where while
what how did does do was were is are be been being plays play games game season seasons year years i
me my we our you your show shows showing give get find list draw plot chart charts map look lets let us
please can could would some any all every only just times time during stats data nfl player players
team teams like these those this it they them there here yards yard yds attempts attempt one ones
him he she has have had into over under than more less most least made make making
against vs versus facing back backs runner runners ball carrier carriers guy guys where
""".split())


@dataclass
class Parse:
    dataset: str | None = None
    seasons: list[int] = field(default_factory=list)   # empty = not mentioned (keep the current pick)
    all_seasons: bool = False                            # "career", "all seasons" - clear the pick
    weeks: list[int] = field(default_factory=list)
    team: str | None = None
    name: str | None = None
    params: dict[str, str] = field(default_factory=dict)
    understood: list[dict] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    engine: str = "rules"
    refused: str | None = None         # the question asks for something this chart can't show (another player's chart)
    llm_prompt: str | None = None      # llm="client": what the visitor's in-browser model should answer
    llm_schema: dict | None = None     # ...and the JSON shape its answer is held to
    llm_messages: list | None = None   # ...or as chat messages: system prompt, worked examples, the question
    llm_ft: list | None = None         # ...and in the fine-tuned model's own compact format

    def as_dict(self):
        return self.__dict__.copy()


def _norm_name(s: str) -> str:
    s = re.sub(r"[.'’\-]", "", s.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def describe(col: str, op: str, value) -> str:
    spec = BY_COL.get(col)
    label = spec["label"] if spec else col
    if spec and spec["kind"] == "bool":
        return f"{label}: {'yes' if str(value) in ('true', '1') else 'no'}"
    labels = {str(v): lab for v, lab in (spec or {}).get("choices") or []}
    vals = str(value).split(",") if op == "in" else [str(value)]
    shown = ", ".join(labels.get(v, v) for v in vals)
    if op in ("eq", "in"):
        return f"{label}: {shown}"
    return f"{label} {OP_SYMBOL[op]} {shown}"


class Players:
    """Name lookup over every player in the DB: full name, unique last name, or unique first name."""

    def __init__(self, rows):
        # rows: (name, team, season, table, count)
        self.by_norm: dict[str, dict] = {}
        for name, team, season, table, n in rows:
            if not name:
                continue
            p = self.by_norm.setdefault(_norm_name(name), {
                "name": name, "count": 0, "teams": set(), "seasons": set(), "tables": {}})
            p["count"] += n
            p["teams"].add(team)
            p["seasons"].add(season)
            p["tables"][table] = p["tables"].get(table, 0) + n
        self.full = sorted(self.by_norm, key=len, reverse=True)
        self.full_re = [(n, re.compile(r"\b" + re.escape(n) + r"(?:s)?\b")) for n in self.full]
        self.by_last, self.by_first = {}, {}
        for n in self.by_norm:
            parts = n.split()
            if len(parts) >= 2:
                last = parts[-1] if parts[-1] not in ("jr", "sr", "ii", "iii", "iv") else parts[-2]
                self.by_last.setdefault(last, []).append(n)
                self.by_first.setdefault(parts[0], []).append(n)
        self.seasons = sorted({s for p in self.by_norm.values() for s in p["seasons"] if s})

    def rank(self, cands, dataset=None, seasons=None, team=None):
        def score(n):
            p = self.by_norm[n]
            return ((team in p["teams"]) if team else 0, bool(set(seasons or []) & p["seasons"]),
                    p["tables"].get(dataset, 0) > 0 if dataset else 0, p["count"])
        return sorted(cands, key=score, reverse=True)


class RuleParser:
    def __init__(self, query: str, players: Players, current_dataset: str, dyn: dict):
        q = query.replace("’", "'").replace("–", "-").replace("—", "-")
        self.orig = q
        self.t = q.lower()
        self.used = [False] * len(self.t)
        self.players = players
        self.current = current_dataset
        self.dyn = dyn
        self.p = Parse()
        self.deferred = []   # (kind, value, text) resolved once the dataset is known

    # ---------- plumbing ----------
    def free(self, s, e):
        return not any(self.used[s:e])

    def consume(self, s, e):
        for i in range(s, e):
            self.used[i] = True

    def rule(self, pattern, handler, flags=0, text=None):
        # already-consumed characters are masked out so a later rule can match around them
        # ("screen passes" after "passes" was taken as the dataset) but never through them
        src = self.t if text is None else text
        src = "".join("\x00" if u else ch for ch, u in zip(src, self.used))
        for m in re.finditer(pattern, src, flags):
            if m.end() > m.start():
                meaning = handler(m)
                if meaning is not False:
                    self.consume(m.start(), m.end())
                    if meaning:
                        self.p.understood.append({"text": self.orig[m.start():m.end()].strip(), "meaning": meaning})

    def set(self, col, op, value):
        if op == "eq":
            self.p.params.pop(col + "__in", None)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if op == "in":
            self.p.params.pop(col, None)
            existing = self.p.params.get(col + "__in")
            vals = (existing.split(",") if existing else []) + [str(v) for v in (value if isinstance(value, list) else [value])]
            value = ",".join(dict.fromkeys(vals))
        self.p.params[col + OP_SUFFIX[op]] = str(value)
        return describe(col, op, value)

    def sets(self, *triples):
        return "; ".join(self.set(*t) for t in triples)

    def want(self, dataset):
        if not self.p.dataset:
            self.p.dataset = dataset

    # ---------- the rules ----------
    def run(self) -> Parse:
        self.receivers()
        self.players_full()
        self.teams()
        self.dataset_words()
        self.seasons_weeks()
        self.clock()
        self.downs()
        self.field_position()
        self.score()
        self.outcomes()
        self.yardage()
        self.passing()
        self.play_calls()
        self.defense()
        self.game()
        self.positions()
        self.epa()
        self.players_partial()
        self.resolve_deferred()
        self.validate()
        self.leftovers()
        return self.p

    def receivers(self):
        """On a quarterback's pass chart a name is who the ball went to - a filter on his throws, not another
        player's chart. Names are the pbp's ("G.Kittle"), matched by last name, and by first name when one is given."""
        chart = self.dyn.get("chart")
        if not chart or chart["table"] != "passes":
            return
        for abbr in sorted(self.dyn.get("choices", {}).get("receiver_player_name", ()), key=len, reverse=True):
            first, _, last = abbr.partition(".")
            words = [w for w in re.findall(r"[a-z]+", last.lower()) if w not in ("jr", "sr", "ii", "iii", "iv")]
            if not words or len("".join(words)) < 3 or "".join(words) in STOPWORDS:
                continue
            first = re.sub(r"[^a-z]", "", first.lower())
            last_rx = r"[.\s'\-]*".join(re.escape(w) for w in words) + r"(?:'s|s)?"

            def full(m, abbr=abbr, first=first):
                if not re.sub(r"[^a-z]", "", m.group(1)).startswith(first):
                    return False
                return self.set("receiver_player_name", "eq", abbr)

            def only_last(m, abbr=abbr, first=first):
                before = re.findall(r"[a-z'.\-]+", self.t[:m.start()])
                given = re.sub(r"[^a-z]", "", before[-1]) if before else ""
                if given in self.players.by_first and not given.startswith(first):
                    return False          # "Brian Robinson" isn't "D.Robinson"
                return self.set("receiver_player_name", "eq", abbr)

            self.rule(rf"\b([a-z][a-z'.\-]+)\s+{last_rx}\b", full)
            self.rule(rf"\b{last_rx}\b", only_last)

    def players_full(self):
        alpha = re.sub(r"[.'\-]", "", self.t)
        # map alpha-string offsets back to self.t offsets
        idx = [i for i, ch in enumerate(self.t) if ch not in ".'-"]
        for n, rx in self.players.full_re:
            if self.p.name:
                break
            m = rx.search(alpha)
            if m:
                s, e = idx[m.start()], idx[m.end() - 1] + 1
                if self.free(s, e):
                    self.p.name = self.players.by_norm[n]["name"]
                    self.consume(s, e)
                    self.p.understood.append({"text": self.orig[s:e], "meaning": f"Player: {self.p.name}"})

    def teams(self):
        aliases = sorted(ALIAS_TO_CHART, key=len, reverse=True)
        opp = r"(?:against|vs\.?|versus|facing|at)\s+(?:the\s+)?"

        def team_hit(chart_code, m, opponent, text):
            if opponent:
                if "at " in text.lower()[:3]:
                    self.set("posteam_type", "eq", "away")
                return self.set("defteam", "eq", CHART_TO_PBP[chart_code]) + (
                    "; Offense is: Away" if text.lower().startswith("at ") else "")
            if self.p.team and self.p.team != chart_code:
                return False
            self.p.team = chart_code
            return f"Team: {chart_code}"

        alias_re = "|".join(re.escape(a) for a in aliases)
        self.rule(rf"\b{opp}({alias_re})(?:'s|')?\b",
                  lambda m: team_hit(ALIAS_TO_CHART[m.group(1)], m, True, m.group(0)))
        code_re = "|".join(ALL_CODES)
        self.rule(rf"\b(?i:{opp})({code_re})\b",
                  lambda m: team_hit(PBP_TO_CHART.get(m.group(1), m.group(1)), m, True, m.group(0)),
                  text=self.orig)
        self.rule(rf"\b({alias_re})(?:'s|')?\b", lambda m: team_hit(ALIAS_TO_CHART[m.group(1)], m, False, ""))
        self.rule(rf"\b({code_re})\b",
                  lambda m: team_hit(PBP_TO_CHART.get(m.group(1), m.group(1)), m, False, ""), text=self.orig)

    def dataset_words(self):
        def ds(name):
            def h(m):
                self.want(name)
                return f"Dataset: {name}" if self.p.dataset == name else ""
            return h
        self.rule(r"\b(?:qb |quarterback )?scrambl(?:e|es|ed|ing)\b",
                  lambda m: (self.want("carries"), self.set("qb_scramble", "eq", "true"))[1])
        self.rule(r"\bdesigned (?:qb |quarterback )?runs?\b",
                  lambda m: (self.want("carries"), self.set("qb_scramble", "eq", "false"))[1])
        # "throws on the run" is a passer on the move, not a carry
        self.rule(r"\bon the run\b", lambda m: self.set("is_qb_out_of_pocket", "eq", "true"))
        # "runs for a loss" / "5-man rush" belong to later rules - leave those words for them
        self.rule(r"\b(?:carries|carry|runs(?! for (?:a )?loss)|run(?![ -](?:pass|option|gap|direction|location|game|for))|"
                  r"run plays?|(?<!man )(?<!man-)(?:rushes|rushing(?: attempts| plays)?|rush attempts?))\b",
                  ds("carries"))
        self.rule(r"\b(?:routes?|targets?|targeted|receiving)\b", ds("routes"))
        self.rule(r"\b(?:passes|passing(?: plays)?|throws|throwing|pass attempts?|dropbacks?|pass plays?)\b",
                  ds("passes"))

    def seasons_weeks(self):
        have = self.players.seasons
        latest = have[-1] if have else None

        def seasons(ys):
            ys = [y for y in ys if y in have]
            if not ys:
                return False
            self.p.seasons = sorted(set(self.p.seasons) | set(ys))
            return ("Seasons: " if len(self.p.seasons) > 1 else "Season: ") + ", ".join(map(str, self.p.seasons))

        def year(s):
            y = int(s)
            return y + 2000 if y < 100 else y

        def span(a, b):
            a, b = year(a), year(b)
            return range(min(a, b), max(a, b) + 1)

        def all_seasons(m):
            self.p.all_seasons = True
            return "Seasons: all"

        # "2024-25" is one season (the one starting in 2024); "2022-24" / "2022 to 2024" is a range
        self.rule(r"\b(20[12]\d)-(\d{2})\b(?:\s+season)?",
                  lambda m: seasons([int(m.group(1))]) if year(m.group(2)) == int(m.group(1)) + 1 else seasons(span(m.group(1), m.group(2))))
        self.rule(r"\b(?:from\s+)?(20[12]\d)\s*(?:-|to|through|thru|until)\s*(20[12]\d)\b(?:\s+seasons?)?",
                  lambda m: seasons(span(m.group(1), m.group(2))))
        if latest:
            self.rule(r"\b(?:since|from|starting(?: in)?)\s+(20[12]\d)\b(?:\s+on(?:ward)?)?|\b(20[12]\d)\s+(?:and )?(?:on|onward|onwards|and later|or later)\b",
                      lambda m: seasons(range(int(m.group(1) or m.group(2)), latest + 1)))
            self.rule(r"\b(?:after)\s+(20[12]\d)\b", lambda m: seasons(range(int(m.group(1)) + 1, latest + 1)))
            self.rule(r"\b(?:last|past|previous|recent)\s+(\d|two|three|four|five)\s+(?:seasons|years)\b",
                      lambda m: seasons(range(latest - {"two": 2, "three": 3, "four": 4, "five": 5}.get(m.group(1), int(m.group(1)) if m.group(1).isdigit() else 1) + 1, latest + 1)))
            self.rule(r"\b(?:this|current) (?:season|year)\b", lambda m: seasons([latest]))
            self.rule(r"\b(?:last|previous) (?:season|year)\b", lambda m: seasons([latest - 1]))
        self.rule(r"\b(20[12]\d)(?:\s+season)?\b", lambda m: seasons([int(m.group(1))]))
        self.rule(r"\b(?:all|every|any) (?:seasons|years)\b|\bcareer\b|\ball[ -]time\b", all_seasons)

        def weeks(ws, text):
            ws = sorted({w for w in ws if 1 <= w <= 23})
            if not ws:
                return False
            self.p.weeks = sorted(set(self.p.weeks) | set(ws))
            return "Weeks: " + ", ".join(map(str, self.p.weeks))

        self.rule(r"\bweeks?\s+(\d{1,2})\s*(?:-|to|through|thru)\s*(?:week\s+)?(\d{1,2})\b",
                  lambda m: weeks(range(int(m.group(1)), int(m.group(2)) + 1), m.group(0)))
        self.rule(r"\bweeks?\s+(\d{1,2}(?:\s*(?:,|and|or|&)\s*(?:week\s+)?\d{1,2})+)\b",
                  lambda m: weeks([int(x) for x in re.findall(r"\d{1,2}", m.group(1))], m.group(0)))
        self.rule(r"\b(?:first|last) (\d{1,2}) weeks\b",
                  lambda m: weeks(range(1, int(m.group(1)) + 1), m.group(0)) if "first" in m.group(0) else False)
        self.rule(r"\bweek\s+(\d{1,2})\b", lambda m: weeks([int(m.group(1))], m.group(0)))
        self.rule(r"\b(?:playoffs?|postseason|post-season|wild ?card(?: round)?|divisional round|"
                  r"conference championships?|championship games?|super bowls?|(?:afc|nfc) (?:title|championship)(?: games?)?|"
                  r"title games?|conference finals?)\b",
                  lambda m: self.set("season_type", "eq", "POST"))
        self.rule(r"\bregular season\b", lambda m: self.set("season_type", "eq", "REG"))

    def clock(self):
        def qtrs(text):
            return [ORD[o] for o in re.findall(ORD_RE, text)]
        self.rule(rf"\b({ORD_RE}(?:\s*(?:,|or|and|&|/)\s*{ORD_RE})*)\s+(?:quarters?|qtrs?)\b",
                  lambda m: self.set("qtr", "in", qtrs(m.group(1))))
        self.rule(r"\bq([1-4])\b", lambda m: self.set("qtr", "in", [int(m.group(1))]))
        self.rule(r"\b(?:overtime|ot)\b", lambda m: self.set("qtr", "in", [5]))
        self.rule(r"\b(?:1st|first) half\b", lambda m: self.set("qtr", "in", [1, 2]))
        self.rule(r"\b(?:2nd|second) half\b", lambda m: self.set("qtr", "in", [3, 4]))
        self.rule(r"\b(?:two|2)[ -]minute(?: drill| warning)?s?\b|\bend of (?:the )?half\b|"
                  r"\b(?:final|last) (?:two|2) minutes(?: of (?:the )?half)?\b",
                  lambda m: self.set("half_seconds_remaining", "lte", 120))
        self.rule(r"\b(?:end of (?:the )?game|late in (?:the )?game|crunch time|clutch)\b",
                  lambda m: self.sets(("qtr", "in", [4, 5]), ("half_seconds_remaining", "lte", 300)))

    def downs(self):
        def dist(tok):
            tok = tok.strip()
            if tok == "long":
                return [("ydstogo", "gte", 7)]
            if tok == "medium":
                return [("ydstogo", "gte", 3), ("ydstogo", "lte", 6)]
            if tok == "short":
                return [("ydstogo", "lte", 2)]
            if tok == "goal":
                return [("goal_to_go", "eq", "true")]
            n = int(re.match(r"\d+", tok).group(0))
            if re.search(r"\+|or more|plus", tok):
                return [("ydstogo", "gte", n)]
            if re.search(r"or less|or fewer", tok):
                return [("ydstogo", "lte", n)]
            return [("ydstogo", "eq", n)]

        self.rule(rf"\b({ORD_RE})[\s-]*(?:downs?[\s-]*)?(?:and|&|-)[\s-]*(long|short|medium|goal|\d{{1,2}}(?:\s*\+|\s*or more|\s*plus|\s*or less|\s*or fewer)?)",
                  lambda m: self.sets(("down", "in", [ORD[m.group(1)]]), *dist(m.group(2))))
        # "for a first down" / "moved the chains" are a result, not the down - handle before plain downs
        self.rule(r"\b(?:for|picked up|pick up|got|get|gained|gaining|gain|resulting in|results? in|went for|"
                  r"moving the chains|moved the chains|move the chains)\s*(?:a\s+)?(?:(?:1st|first) downs?)?",
                  lambda m: self.set("first_down", "eq", "true") if re.search(r"first|1st|chains", m.group(0)) else False)
        self.rule(r"\b(?:conversions?|converted|converting)\b", lambda m: self.set("first_down", "eq", "true"))
        self.rule(rf"\b({ORD_RE}(?:\s*(?:,|or|and|&|/)\s*{ORD_RE})*)\s+downs?\b",
                  lambda m: self.set("down", "in", [ORD[o] for o in re.findall(ORD_RE, m.group(1))]))
        self.rule(r"\blate downs?\b", lambda m: self.set("down", "in", [3, 4]))
        self.rule(r"\bearly downs?\b", lambda m: self.set("down", "in", [1, 2]))
        self.rule(r"\bshort[ -]yardage\b", lambda m: self.set("ydstogo", "lte", 2))
        self.rule(r"\blong[ -]yardage\b|\band long\b", lambda m: self.set("ydstogo", "gte", 7))
        self.rule(r"\band short\b", lambda m: self.set("ydstogo", "lte", 2))
        self.rule(rf"\b{NUM}\s*(?:\+|or more)?\s*(?:yards?|yds)?\s+to go\b",
                  lambda m: self.set("ydstogo", "gte" if re.search(r"\+|more", m.group(0)) else "eq", int(float(m.group(1)))))
        self.rule(r"\bgoal[ -]to[ -]go\b", lambda m: self.set("goal_to_go", "eq", "true"))

    def field_position(self):
        self.rule(r"\bred ?zone\b", lambda m: self.set("yardline_100", "lte", 20))
        self.rule(r"\b(?:goal ?line|goal-line)\b", lambda m: self.set("yardline_100", "lte", 5))
        self.rule(r"\binside (?:the |their |its |his |opponent'?s? )?(\d{1,2})\b",
                  lambda m: self.set("yardline_100", "lte", int(m.group(1))))
        self.rule(r"\b(?:backed up|own end ?zone)\b", lambda m: self.set("yardline_100", "gte", 90))
        self.rule(r"\b(?:own territory|own side(?: of the field)?|own half)\b",
                  lambda m: self.set("yardline_100", "gte", 50))
        self.rule(r"\b(?:opponent'?s? territory|plus territory|opp territory|enemy territory|their territory)\b",
                  lambda m: self.set("yardline_100", "lte", 49))

    def score(self):
        self.rule(r"\b(?:down|trailing|behind) by (\d{1,2})(?:\s*(?:\+|or more))?\b",
                  lambda m: self.set("score_differential", "lte", -int(m.group(1))))
        self.rule(r"\b(?:up|leading|ahead) by (\d{1,2})(?:\s*(?:\+|or more))?\b",
                  lambda m: self.set("score_differential", "gte", int(m.group(1))))
        self.rule(r"\b(?:when |while )?(?:trailing|losing|playing from behind)\b",
                  lambda m: self.set("score_differential", "lt", 0))
        self.rule(r"\b(?:when |while )?(?:leading|winning|with the lead|ahead)\b",
                  lambda m: self.set("score_differential", "gt", 0))
        self.rule(r"\b(?:tied|tie game|all square)\b", lambda m: self.set("score_differential", "eq", 0))
        self.rule(r"\b(?:one[ -]score games?|close games?|one possession)\b",
                  lambda m: self.sets(("score_differential", "gte", -8), ("score_differential", "lte", 8)))
        self.rule(r"\b(?:blowouts?|garbage time)\b",
                  lambda m: self.sets(("wp", "lte", 0.1)) if "garbage" in m.group(0) else False)

    def outcomes(self):
        neg = r"(?:(no|non|without|not|w/o)[\s-]+)?"
        # before "picks"/"ints" below would read these as actual interceptions
        self.rule(r"\b(?:interception|int|turnover)[ -]worthy(?: throws| passes| plays)?\b|"
                  r"\b(?:near|almost|should-have-been|should have been|dropped)[ -](?:picks?|interceptions?|ints?)\b",
                  lambda m: (self.want("passes"), self.set("is_interception_worthy", "eq", "true"))[1])

        def td(m):
            if m.group(1):
                return self.set("bucket", "ne", "TOUCHDOWN")
            return self.set("bucket", "eq", "TOUCHDOWN")
        self.rule(r"\b(?:qb |quarterback )?sneaks?\b|\btush[ -]push(?:es)?\b|\bbrotherly shoves?\b",
                  lambda m: (self.want("carries"), self.set("is_qb_sneak", "eq", "true"))[1])
        self.rule(rf"\b{neg}(?:touchdowns?|tds?|scores|scored|scoring(?: plays?| runs?| catches| passes| throws)?|"
                  rf"found the end ?zone|reached the end ?zone|into the end ?zone|to the house|house calls?)\b", td)

        self.rule(r"\b(?:interceptions?|picks|picked off|ints?|pick[ -]sixe?s?)\b",
                  lambda m: (self.want("passes"), self.set("bucket", "eq", "INTERCEPTION"))[1])
        self.rule(r"\b(?:incompletions?|incompletes?|incomplete (?:passes|throws|targets))\b",
                  lambda m: (self.want("passes"), self.set("bucket", "eq", "INCOMPLETE"))[1])
        self.rule(r"\b(?:catches|receptions?|caught)\b",
                  lambda m: (self.want("routes"), self.set("bucket", "in", ["COMPLETE", "TOUCHDOWN"]))[1])
        self.rule(r"\b(?:completions?|completed(?: passes)?|complete(?: passes)?)\b",
                  lambda m: (self.want("passes"), self.set("bucket", "in", ["COMPLETE", "TOUCHDOWN"]))[1])
        self.rule(r"\b(?:(?:runs? |carries |rushes )?for (?:a )?loss|negative (?:runs|plays|carries)|tackles? for (?:a )?loss|tfls?|"
                  r"stuffed|stuffs|lost yard(?:s|age)|losses)\b",
                  lambda m: (self.want("carries"), self.set("gain_class", "eq", "LOSS"))[1])
        self.rule(r"\bfumbles?(?: lost)?\b", lambda m: self.set("fumble_lost", "eq", "true"))
        self.rule(r"\b(?:unsuccessful|failed|failures?)\b", lambda m: self.set("success", "eq", "false"))
        self.rule(r"\b(?:successful|success)\b", lambda m: self.set("success", "eq", "true"))

        def explosive(m):
            self.deferred.append(("explosive", None))
            return "Explosive play"
        self.rule(r"\b(?:explosive|big|chunk|huge)(?: plays?| runs?| gains?| catches| passes| completions)?\b", explosive)

    def yardage(self):
        cmp = [(r"(?:at least|over|more than|>=?|above)", "gte"), (r"(?:under|less than|fewer than|<=?|below)", "lte")]
        for word, op in cmp:
            self.rule(rf"(?<![a-z]){word}\s*{NUM}\s*air yards?\b", lambda m, op=op: self.set("air_yards", op, float(m.group(1))))
            self.rule(rf"(?<![a-z]){word}\s*{NUM}\s*(?:yards?|yds)\b", lambda m, op=op: self.set("yards_gained", op, float(m.group(1))))
        self.rule(rf"\b{NUM}\s*\+\s*air yards?\b|\b{NUM}\s+or more air yards?\b",
                  lambda m: self.set("air_yards", "gte", float(m.group(1) or m.group(2))))
        self.rule(rf"\b{NUM}\s*(?:\+|or more)\s*(?:-?\s*)?(?:yards?|yds)\b|\b{NUM}\s*\+\s*(?:yard|yd)\b|\b{NUM}\s*\+(?=\s*(?:runs|gains|catches|plays|carries|passes|completions))",
                  lambda m: self.set("yards_gained", "gte", float(next(g for g in m.groups() if g))))
        self.rule(rf"\bno gain\b|\bzero[ -]yard", lambda m: self.set("yards_gained", "eq", 0))

    def passing(self):
        def defer(kind, value, label):
            def h(m):
                self.deferred.append((kind, value))
                return label
            return h
        self.rule(r"\b(?:behind the (?:line(?: of scrimmage)?|los))\b", defer("depth", "behind", "Behind the line"))
        self.rule(r"\b(?:deep(?: passes| shots?| balls?| throws?| targets?| routes?)?|bombs?|downfield|go balls?)\b",
                  defer("depth", "deep", "Deep"))
        self.rule(r"\bintermediate(?: passes| throws| targets| routes)?\b", defer("depth", "intermediate", "Intermediate depth"))
        self.rule(r"\bshort(?: passes| throws| targets| routes| balls)\b|\bquick (?:passes|throws|game)\b",
                  defer("depth", "short", "Short"))
        self.rule(r"\b(?:over the middle|middle of the field|up the middle|between the tackles|inside runs?)\b",
                  defer("dir", ["middle"], "Middle"))
        self.rule(r"\b(?:outside(?! (?:the |of the )?pocket)(?: runs?| zone)?|to the outside|on the edge|to the edges?|perimeter)\b",
                  defer("dir", ["left", "right"], "Outside"))
        self.rule(r"\b(?:to the |toward the |on the )?(left|right)(?: side)?\b", lambda m: self.deferred.append(("dir", [m.group(1)])) or f"Direction: {m.group(1)}")
        self.rule(r"\b(?:off )?(guard|tackle)(?: runs?| gap)?\b|\b(?:around the |off the )?(?:end|edge)(?: runs?)?\b|\bsweeps?\b",
                  lambda m: (self.want("carries"), self.set("run_gap", "in", [m.group(1) or "end"]))[1])

    def play_calls(self):
        neg = r"(?:(no|non|without|not|w/o|never)[\s-]+)?"
        bools = [
            ("no_huddle", r"no[ -]huddle|hurry[ -]up|up[ -]tempo", True),
            ("is_play_action", r"play[ -]?action(?: pass(?:es)?)?|pa|fakes?", False),
            ("shotgun", r"shotgun|(?:out of|from) the gun", False),
            ("is_motion", r"(?:pre[ -]?snap )?motion", False),
            ("is_rpo", r"rpos?|run[ -]pass options?", False),
            ("is_screen_pass", r"screens?(?: pass(?:es)?)?", False),
            ("is_trick_play", r"trick(?: plays?)?|gadget(?: plays?)?", False),
            ("is_qb_out_of_pocket", r"(?:out of|outside) (?:the )?pocket|on the run|rollouts?|roll outs?|boot(?:leg)?s?", False),
            ("qb_hit", r"qb hits?|hit (?:as|while) (?:he )?thr(?:ew|owing)", False),
            ("is_throw_away", r"throw ?aways?|thrown away", False),
            ("is_catchable_ball", r"catchable(?: balls?| passes| targets)?", False),
            ("is_contested_ball", r"contested(?: catches| balls?| targets?)?", False),
            ("is_drop", r"drops|dropped(?: passes| balls)?|drop", False),
            ("is_created_reception", r"created receptions?|circus catch(?:es)?|highlight catch(?:es)?", False),
            ("div_game", r"divisi(?:on|onal) (?:games?|rivals?|matchups?)", False),
        ]
        self.rule(r"\buncatchable\b", lambda m: self.set("is_catchable_ball", "eq", "false"))
        self.rule(r"\bunder center\b", lambda m: self.sets(("shotgun", "eq", "false")))
        self.rule(r"\bin (?:the )?pocket\b", lambda m: self.set("is_qb_out_of_pocket", "eq", "false"))
        forms = [(r"pistol", "PISTOL"), (r"empty(?: set| backfield| formation)?", "EMPTY"),
                 (r"i[- ]form(?:ation)?", "I_FORM"), (r"single[ -]?back", "SINGLEBACK"),
                 (r"jumbo|heavy (?:set|package|formation)", "JUMBO"), (r"wildcat", "WILDCAT")]
        for phrase, val in forms:
            self.rule(rf"\b(?:{phrase})\b", lambda m, val=val: self.set("offense_formation", "in", [val]))
        for col, phrase, literal in bools:
            if literal:
                self.rule(rf"\b(?:{phrase})\b", lambda m, col=col: self.set(col, "eq", "true"))
            else:
                self.rule(rf"\b{neg}(?:{phrase})\b",
                          lambda m, col=col: self.set(col, "eq", "false" if m.group(1) else "true"))

    def defense(self):
        word_n = {"five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "four": 4, "three": 3}
        n = r"(\d|five|six|seven|eight|nine|four|three)"
        val = lambda s: int(s) if s.isdigit() else word_n[s]
        cmp = r"(?:(more than|over|at least|fewer than|less than|under)\s+)?"

        def count_op(m, gi_cmp, gi_plus):
            c, plus = m.group(gi_cmp), m.group(gi_plus)
            if plus or c == "at least":
                return "gte"
            return {"more than": "gt", "over": "gt", "fewer than": "lt", "less than": "lt", "under": "lt"}.get(c, "eq")

        self.rule(r"\b(?:stacked|heavy|loaded|crowded) box(?:es)?\b", lambda m: self.set("n_defense_box", "gte", 8))
        self.rule(r"\b(?:light|empty|soft) box(?:es)?\b", lambda m: self.set("n_defense_box", "lte", 6))
        self.rule(rf"\b{cmp}{n}\s*(\+|or more)?\s*(?:-?\s*man|defenders|men|players)\s*(?:in the )?box(?:es)?\b",
                  lambda m: self.set("n_defense_box", count_op(m, 1, 3), val(m.group(2))))
        self.rule(rf"\b{cmp}{n}\s*(\+|or more)?\s*(?:-?\s*man rush(?:es)?|(?:pass )?rushers)\b",
                  lambda m: (self.want("passes"), self.set("n_pass_rushers", count_op(m, 1, 3), val(m.group(2))))[1])
        self.rule(r"\b(?:no|without|not|w/o)[\s-]+(?:a\s+)?blitz(?:es|ed|ing)?\b",
                  lambda m: self.set("n_blitzers", "eq", 0))
        self.rule(r"\b(?:blitz(?:es|ed|ing)?|extra rushers?|sends? pressure)\b",
                  lambda m: self.set("n_blitzers", "gte", 1))
        # NGS coverage / pressure (participation data)
        self.rule(r"\b(?:cover[ -]?2[ -]?man|2[ -]man(?: under)?)\b",
                  lambda m: self.set("defense_coverage_type", "in", ["2_MAN"]))
        self.rule(r"\b(?:tampa[ -]?2)\b", lambda m: self.set("defense_coverage_type", "in", ["COVER_2"]))
        self.rule(r"\bcover[ -]?(zero|one|two|three|four|six|[0-6])\b",
                  lambda m: self.set("defense_coverage_type", "in", [
                      "COVER_" + {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "six": "6"}.get(m.group(1), m.group(1))]))
        self.rule(r"\bquarters(?: coverage)?\b", lambda m: self.set("defense_coverage_type", "in", ["COVER_4"]))
        self.rule(r"\bprevent(?: defense)?\b", lambda m: self.set("defense_coverage_type", "in", ["PREVENT"]))
        self.rule(r"\b(?:man[ -]to[ -]man|man)(?: coverage| defense)?\b",
                  lambda m: self.set("defense_man_zone_type", "in", ["MAN_COVERAGE"]))
        self.rule(r"\bzone(?: coverage| defense)?\b", lambda m: self.set("defense_man_zone_type", "in", ["ZONE_COVERAGE"]))
        self.rule(r"\b(?:clean pocket|kept clean|no pressure|without pressure|not pressured|unpressured)\b",
                  lambda m: self.set("was_pressure", "eq", "false"))
        self.rule(r"\b(?:under pressure|pressured|pressure)\b", lambda m: self.set("was_pressure", "eq", "true"))

    def game(self):
        self.rule(r"\b(?:domes?|indoors?|indoor games?|under a roof)\b", lambda m: self.set("roof", "in", ["dome", "closed"]))
        self.rule(r"\b(?:outdoors?|outdoor games?|open air)\b", lambda m: self.set("roof", "in", ["outdoors", "open"]))
        self.rule(r"\b(?:cold(?: weather)?(?: games?)?|freezing|below freezing|snow(?:y)?)\b",
                  lambda m: self.set("temp", "lte", 32 if re.search("freez|snow", m.group(0)) else 40))
        self.rule(r"\b(?:windy|high wind|wind(?:y)? games?)\b", lambda m: self.set("wind", "gte", 15))
        self.rule(r"\b(?:at home|home games?|home)\b", lambda m: self.set("posteam_type", "eq", "home"))
        self.rule(r"\b(?:on the road|road games?|away games?|away)\b", lambda m: self.set("posteam_type", "eq", "away"))

    def positions(self):
        keys = sorted(POSITIONS, key=len, reverse=True)
        pos_re = "|".join(re.escape(k) for k in keys)
        valid = self.dyn.get("positions") or set(POSITIONS.values())

        def h(m):
            pos = POSITIONS[m.group(1)]
            if pos not in valid:
                return False
            if pos in ("RB",):
                return self.set("position", "in", ["RB", "HB"])
            return self.set("position", "in", [pos])
        self.rule(rf"\b({pos_re})\b", h)

    def epa(self):
        self.rule(r"\bpositive epa\b", lambda m: self.set("epa", "gt", 0))
        self.rule(r"\bnegative epa\b", lambda m: self.set("epa", "lt", 0))
        self.rule(r"\bepa\s*(?:of\s*)?(?:over|above|greater than|>=?|at least|\+)\s*(-?\d+(?:\.\d+)?)",
                  lambda m: self.set("epa", "gte", float(m.group(1))))
        self.rule(r"\bepa\s*(?:under|below|less than|<=?)\s*(-?\d+(?:\.\d+)?)",
                  lambda m: self.set("epa", "lte", float(m.group(1))))

    def players_partial(self):
        """Last-name or unique first-name matches on whatever is still unconsumed."""
        if self.p.name:
            return
        for m in re.finditer(r"[a-z][a-z.'\-]+", self.t):
            if not self.free(m.start(), m.end()):
                continue
            tok = re.sub(r"[.'\-]", "", re.sub(r"'s$", "", m.group(0)))
            tok = tok[:-1] if tok.endswith("s") and tok[:-1] in self.players.by_last and tok not in self.players.by_last else tok
            if tok in STOPWORDS or len(tok) < 3:
                continue
            cands = self.players.by_last.get(tok) or self.players.by_first.get(tok)
            if not cands:
                continue
            ranked = self.players.rank(cands, self.p.dataset or self.current, self.p.seasons, self.p.team)
            best = self.players.by_norm[ranked[0]]["name"]
            self.p.name = best
            self.consume(m.start(), m.end())
            self.p.understood.append({"text": self.orig[m.start():m.end()], "meaning": f"Player: {best}"})
            if len(ranked) > 1:
                others = ", ".join(self.players.by_norm[n]["name"] for n in ranked[1:4])
                self.p.notes.append(f"'{m.group(0)}' matched several players - picked {best}. Others: {others}. "
                                    f"Use the full name to switch.")
            return

    def resolve_deferred(self):
        ds = self.p.dataset or self.current
        for kind, value in self.deferred:
            if kind == "explosive":
                self.set("yards_gained", "gte", 10 if ds == "carries" else 20)
            elif kind == "depth":
                if ds == "carries":
                    self.p.notes.append("Pass depth doesn't apply to carries - ignored.")
                    continue
                if value == "deep":
                    self.set("air_yards", "gte", 20)
                elif value == "short":
                    self.set("air_yards", "lt", 10)
                elif value == "intermediate":
                    self.sets(("air_yards", "gte", 10), ("air_yards", "lt", 20))
                elif value == "behind":
                    self.set("air_yards", "lt", 0)
            elif kind == "dir":
                self.set("run_location" if ds == "carries" else "pass_location", "in", value)

    def validate(self):
        _validate(self.p, self.current, self.dyn)

    def leftovers(self):
        words = []
        for m in re.finditer(r"[a-z0-9][a-z0-9.'\-+%]*", self.t):
            if self.free(m.start(), m.end()):
                w = m.group(0).strip(".'")
                if w and w not in STOPWORDS and not re.fullmatch(r"'?s", w):
                    words.append(self.orig[m.start():m.end()])
        self.p.ignored = words


BUCKETS = {"passes": {"COMPLETE", "INCOMPLETE", "TOUCHDOWN", "INTERCEPTION"},
           "routes": {"COMPLETE", "INCOMPLETE", "TOUCHDOWN"},
           "carries": {"TOUCHDOWN", "OTHER"}}


def _validate(p: Parse, current: str, dyn: dict | None = None):
    """Drop anything the chosen dataset can't filter on, with a note instead of a silent drop."""
    chart = (dyn or {}).get("chart")
    if chart:
        # with a player's chart open, the question is about that chart: no other player's, no other kind
        if p.name and _norm_name(p.name) != _norm_name(chart["name"]):
            p.refused = (f"{p.name} isn't one of {chart['name']}'s receivers." if chart["table"] == "passes"
                         else f"This is {chart['name']}'s chart.") + " Use search to switch players."
            p.name = None
        p.dataset = chart["table"]
        pos = p.params.pop("position__in", None) or p.params.pop("position", None)
        if pos and chart["table"] == "passes":
            # pass charts are one quarterback's throws: a position means who they went to
            vals = list(dict.fromkeys({"HB": "RB"}.get(v, v) for v in pos.split(",")))
            vals = [v for v in vals if v in ("WR", "TE", "RB", "FB")]
            if vals:
                p.params["receiver_position__in"] = ",".join(vals)
    ds = p.dataset = p.dataset or current
    pos = p.params.get("position__in", p.params.get("position", ""))
    if ds == "passes" and pos and "QB" not in pos.split(","):
        # pass charts are per quarterback; the receivers' side of the same throws is the routes table
        p.dataset = ds = "routes"
        p.notes.append("Pass charts are per quarterback, so this shows the receivers' routes instead.")
    allowed = {f["col"] for f in for_table(ds)}
    # choice filters are always multi-select in the UI: normalise `col=x` to `col__in=x`
    for key in list(p.params):
        if "__" not in key and BY_COL.get(key, {}).get("kind") == "choice":
            p.params[key + "__in"] = p.params.pop(key)
    for key in list(p.params):
        col = key.split("__")[0]
        if col not in allowed:
            del p.params[key]
            label = BY_COL[col]["label"] if col in BY_COL else col
            p.notes.append(f"'{label}' doesn't apply to {ds} - ignored.")
    for key in [k for k in p.params if k.startswith("bucket")]:
        vals = [v for v in p.params[key].split(",") if v in BUCKETS[ds]]
        if not vals:
            del p.params[key]
            p.notes.append(f"That outcome doesn't exist for {ds} - ignored.")
        else:
            p.params[key] = ",".join(vals)


def parse_rules(query: str, players: Players, current_dataset: str, dyn: dict) -> Parse:
    return RuleParser(query, players, current_dataset, dyn).run()


# ---------- Gemini fallback ----------

def _filters_for_prompt() -> str:
    lines = []
    for f in BY_COL.values():
        vals = ""
        if f["choices"]:
            vals = ", ".join(f"{v}={lab}" for v, lab in f["choices"])
        elif f["kind"] == "bool":
            vals = "true/false"
        elif f["kind"] == "range":
            vals = "number"
        tables = "all" if len(f["tables"]) == 3 else "/".join(f["tables"])
        extra = f" ({f['help']})" if f.get("help") else ""
        lines.append(f"- {f['col']} | {f['label']}{extra} | {f['kind']} | {vals} | {tables}")
    return "\n".join(lines)


def _prompt(query: str, draft: Parse, seasons: list[int], dyn: dict) -> str:
    draft_json = json.dumps({
        "dataset": draft.dataset, "seasons": draft.seasons, "all_seasons": draft.all_seasons,
        "weeks": draft.weeks, "team": draft.team,
        "player": draft.name, "filters": [
            {"column": k.split("__")[0], "op": (k.split("__")[1] if "__" in k else "eq"), "value": v}
            for k, v in draft.params.items()],
    })
    return f"""You translate questions about NFL plays into filters for a play-by-play database. Reply with ONLY a JSON object, no prose.

Datasets:
- passes: one row per quarterback pass (the player is the passer)
- routes: one row per targeted route (the player is the receiver)
- carries: one row per run (the player is the ball carrier)
Seasons available: {seasons[0]}-{seasons[-1]}. Weeks 1-18 are regular season, higher weeks are playoffs.
Offense team codes (for "team"): {", ".join(c for c, _, _ in TEAMS)}
Allowed values for choice filters without listed values: {json.dumps({k: sorted(map(str, v)) for k, v in (dyn.get("choices") or {}).items()})}.

Filters you may use (column | label | type | allowed values | datasets):
{_filters_for_prompt()}
- defteam | Opponent (defense) | select | pbp team codes: {", ".join(p for _, p, _ in TEAMS)} | all

Question: {json.dumps(query)}
A rule-based parser already produced this draft: {draft_json}
It could not interpret these words: {json.dumps(draft.ignored)}

Return this JSON shape:
{{"dataset": "passes|routes|carries", "seasons": [int] (empty if not mentioned), "all_seasons": true if the question
  asks for every season / a career, "weeks": [int], "team": code or null,
  "opponent": code or null, "player": "full name" or null,
  "filters": [{{"column": "...", "op": "eq|in|gte|lte|gt|lt|ne", "value": ...}}],
  "unsupported": "the part of the question these filters cannot express, or empty string"}}
Return the COMPLETE corrected filter list: keep every draft filter that is right, fix any the uninterpreted
words change the meaning of (e.g. "near interceptions" is not an interception), and add filters for those words.
Never return two filters on the same column that contradict each other.
Use only the listed columns. For "in", value is a list. Booleans are true/false. Numbers are numbers.
Playoff week numbers differ by season - for playoff rounds use season_type POST, never guess week numbers.
Map football slang sensibly (e.g. "garbage time"/"blowout" -> 4th quarter with a large score margin; "late" -> 4th quarter)."""


def _coerce(f: dict, op: str, value, dyn: dict):
    """Validate one Gemini-proposed filter; returns the API param value or raises ValueError."""
    kind, col = f["kind"], f["col"]
    if kind == "bool":
        if op != "eq":
            raise ValueError("bool filters only take '='")
        s = str(value).lower()
        if s in ("true", "1", "yes"):
            return "true"
        if s in ("false", "0", "no"):
            return "false"
        raise ValueError(f"not a boolean: {value}")
    if kind == "range":
        if op not in ("gte", "lte", "gt", "lt", "eq", "ne"):
            raise ValueError(f"bad op for a range: {op}")
        v = float(value[0] if isinstance(value, list) else value)
        return str(int(v) if v.is_integer() else v)
    # choice / select
    if op not in ("eq", "in", "ne"):
        raise ValueError(f"bad op for a choice: {op}")
    if f["choices"]:
        valid = {str(v) for v, _ in f["choices"]}
    elif col == "defteam":
        valid = set(PBP_TO_CHART)
    else:   # values that actually occur in the database (positions, roof, coverage, formation, ...)
        valid = {str(v) for v in (dyn.get("choices") or {}).get(col, [])} or None
    vals = value if isinstance(value, list) else [value]
    out = []
    for v in vals:
        v = str(v)
        if col == "defteam":
            v = CHART_TO_PBP.get(v.upper(), v.upper())
        if valid is not None and v not in valid:
            raise ValueError(f"'{v}' isn't a valid {f['label']}")
        out.append(v)
    if not out:
        raise ValueError("empty value")
    return ",".join(out)


def _apply_llm(p: Parse, out: dict, players: Players, dyn: dict):
    changed = []
    ds = out.get("dataset")
    if ds in DATASETS and ds != p.dataset:
        p.dataset = ds
        changed.append(f"Dataset: {ds}")
    ss = sorted({s for s in (out.get("seasons") or []) if isinstance(s, int) and s in players.seasons})
    if ss and ss != p.seasons:
        p.seasons = ss
        changed.append(("Seasons: " if len(ss) > 1 else "Season: ") + ", ".join(map(str, ss)))
    if out.get("all_seasons") is True and not p.seasons and not p.all_seasons:
        p.all_seasons = True
        changed.append("Seasons: all")
    ws = [w for w in (out.get("weeks") or []) if isinstance(w, int) and 1 <= w <= 23]
    if ws and sorted(ws) != p.weeks:
        p.weeks = sorted(set(ws))
        changed.append("Weeks: " + ", ".join(map(str, p.weeks)))
    team = str(out.get("team") or "").upper()
    team = PBP_TO_CHART.get(team, team)
    if team in CHART_TO_PBP and team != p.team:
        p.team = team
        changed.append(f"Team: {team}")
    opp = str(out.get("opponent") or "").upper()
    opp = CHART_TO_PBP.get(opp, opp)
    if opp in PBP_TO_CHART and p.params.get("defteam") != opp:
        p.params["defteam"] = opp
        changed.append(describe("defteam", "eq", opp))
    who = out.get("player")
    if who and not p.name:
        n = _norm_name(who)
        if n in players.by_norm:
            p.name = players.by_norm[n]["name"]
        else:
            parts = n.split()
            cands = players.by_last.get(parts[-1] if parts else "", [])
            if cands:
                p.name = players.by_norm[players.rank(cands, p.dataset, p.seasons, p.team)[0]]["name"]
        if p.name:
            changed.append(f"Player: {p.name}")
        else:
            p.notes.append(f"Couldn't find a player named '{who}'.")

    # Gemini returns the complete corrected filter list (it saw the draft), so it replaces the draft's
    # filters - unless it came back empty, which is more likely laziness than "no filters".
    proposed = {}
    for item in out.get("filters") or []:
        col, op, value = item.get("column"), item.get("op", "eq"), item.get("value")
        f = BY_COL.get(col)
        if not f or op not in OP_SUFFIX:
            p.notes.append(f"Ignored an unknown filter suggestion ({col}).")
            continue
        try:
            v = _coerce(f, op, value, dyn)
        except (ValueError, TypeError) as e:
            p.notes.append(f"Ignored a filter suggestion: {e}.")
            continue
        if f["kind"] == "choice" and op == "eq":
            op = "in"
        elif op == "in" and f["kind"] != "choice":
            op = "eq"
        if op in ("eq", "in"):   # an exact value supersedes any range on the same column
            for k in [k for k in proposed if k.split("__")[0] == col]:
                del proposed[k]
        elif col in proposed:    # a range supersedes an exact value on the same column
            del proposed[col]
        proposed[col + OP_SUFFIX[op]] = v
    if proposed:
        if "defteam" in p.params and "defteam" not in proposed:
            proposed["defteam"] = p.params["defteam"]
        for k, v in proposed.items():
            if p.params.get(k) != v:
                col, op = (k.split("__") + ["eq"])[:2]
                changed.append(describe(col, op, v))
        dropped = [k for k in p.params if k not in proposed]
        for k in dropped:
            col, op = (k.split("__") + ["eq"])[:2]
            changed.append("removed " + describe(col, op, p.params[k]))
        p.params = proposed

    if out.get("unsupported"):
        p.notes.append(f"Couldn't express: {out['unsupported']}")
    for c in changed:
        p.understood.append({"text": "Gemini", "meaning": c})


# a small model can't choose from all ~70 filters: it gets the ones the leftover words name, plus these, which most
# football phrasing ("late", "up big", "third and forever", "garbage time") comes down to
SMALL_FULL_MENU = True
_SITUATION = ["down", "ydstogo", "qtr", "score_differential", "yardline_100", "half_seconds_remaining", "wp"]


def _small_candidates(p: Parse, dataset: str) -> list[dict]:
    allowed = [f for f in for_table(dataset)]
    words = {w.lower().strip(".,'") for w in p.ignored} | set(re.findall(r"[a-z]+", " ".join(p.ignored).lower()))
    words = {w for w in words if len(w) > 2}

    def score(f):
        text = " ".join([f["col"].replace("_", " "), f["label"], f.get("help") or "",
                         " ".join(str(lab) for _, lab in (f["choices"] or []))]).lower()
        return sum(w in text or (len(w) > 4 and w[:5] in text) for w in words)
    if SMALL_FULL_MENU:                       # a 7B model handles the whole menu (~40 lines) better than a guess at it
        return [f for f in allowed if f["col"] not in ("location_prob", "n_candidates", "start_ok", "handoff_ok", "gap")]
    hits = sorted((f for f in allowed if score(f) > 0), key=lambda f: -score(f))[:8]
    keep = {f["col"] for f in hits} | set(_SITUATION) | {k.split("__")[0] for k in p.params}
    return [f for f in allowed if f["col"] in keep]


def small_prompt(query: str, p: Parse) -> tuple[str, dict]:
    """A short prompt and schema for a small in-browser model: only the leftover words, the draft, and a handful of
    candidate filters - it adds or removes filters, nothing else."""
    ds = p.dataset or "passes"
    cands = _small_candidates(p, ds)
    lines = []
    for f in cands:
        if f["choices"]:
            vals = "one of " + ", ".join(f"{v} ({lab})" for v, lab in f["choices"])
        elif f["kind"] == "bool":
            vals = "true/false"
        elif f["kind"] == "range":
            vals = "a number" + (f" - {f['help']}" if f.get("help") else "")
        else:
            vals = "text"
        lines.append(f"- {f['col']}: {f['label']} ({vals})")
    draft = [describe(k.split("__")[0], (k.split("__") + ["eq"])[1], v) for k, v in p.params.items()] or ["(none)"]
    prompt = f"""You turn football words into database filters for {ds} (NFL plays). A parser already understood part of the question.

Question: {json.dumps(query)}
Already applied: {"; ".join(draft)}
Words it did NOT understand: {json.dumps(p.ignored)}

Filters you can use:
{chr(10).join(lines)}

Return JSON: "add" = filters that express the words it did not understand (column, op, value; op is eq, in, gte, lte, gt or lt),
"remove" = columns of applied filters those words show are wrong, "unsupported" = words no filter can express.
Football meanings: late / late in the game = qtr 4; close game = score_differential between -8 and 8; up big / blowout lead = score_differential >= 17; trailing = score_differential < 0; stacked box = n_defense_box >= 8; light box = n_defense_box <= 6; third and forever / long = ydstogo >= 10; deep = air_yards >= 20; short = air_yards <= 9; behind the line = air_yards < 0.
Only add filters for the words it did not understand. If none fit, return empty "add"."""
    cols = sorted({f["col"] for f in cands})
    value = {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"},
                       {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "number"}]}}]}
    schema = {"type": "object", "properties": {
        "add": {"type": "array", "items": {"type": "object", "properties": {
            "column": {"type": "string", "enum": cols}, "op": {"type": "string", "enum": ["eq", "in", "gte", "lte", "gt", "lt"]},
            "value": value}, "required": ["column", "op", "value"]}},
        "remove": {"type": "array", "items": {"type": "string", "enum": sorted({k.split("__")[0] for k in p.params}) or ["none"]}},
        "unsupported": {"type": "string"}}, "required": ["add"]}
    return prompt, schema


SMALL_SYSTEM = """You map leftover words in a football question to database filters. A rule parser already applied some
filters; you only handle the words it did NOT understand. Reply with one JSON object: {"add": [...], "remove": [...],
"unsupported": "..."}. "add": filters for the leftover words, each {"column", "op", "value"} using ONLY the listed
columns. "remove": columns of applied filters that the leftover words show are wrong. "unsupported": leftover words no
listed filter can express. Filler words (out, of, when, the, they, were) need no filter. Keep every applied filter that is
right. Never invent seasons, weeks, players or teams."""

# worked examples (none of them in the evaluation set): question, applied filters, leftover words, candidates, answer
SMALL_EXAMPLES = [
    ("runs when the box is loaded", [], ["box", "loaded"], ["n_defense_box", "down", "qtr", "score_differential"],
     {"add": [{"column": "n_defense_box", "op": "gte", "value": 8}], "remove": [], "unsupported": ""}),
    ("targets late in a tie game", [], ["late", "tie"], ["qtr", "score_differential", "half_seconds_remaining"],
     {"add": [{"column": "qtr", "op": "in", "value": [4]}, {"column": "score_differential", "op": "eq", "value": 0}], "remove": [], "unsupported": ""}),
    ("carries when they were down big", ["Yards gained ≥ 10"], ["down"], ["yards_gained", "score_differential", "down", "qtr"],
     {"add": [{"column": "score_differential", "op": "lte", "value": -17}], "remove": ["yards_gained"], "unsupported": ""}),
    ("throws when the pocket collapsed", [], ["pocket", "collapsed"], ["was_pressure", "is_qb_out_of_pocket", "n_pass_rushers"],
     {"add": [{"column": "was_pressure", "op": "eq", "value": True}], "remove": [], "unsupported": ""}),
    ("deep balls into double coverage", ["Air yards ≥ 20"], ["double"], ["is_contested_ball", "defense_coverage_type", "air_yards"],
     {"add": [{"column": "is_contested_ball", "op": "eq", "value": True}], "remove": [], "unsupported": ""}),
    ("passes on the opening drive", [], ["opening", "drive"], ["qtr", "half_seconds_remaining", "down"],
     {"add": [], "remove": [], "unsupported": "opening drive"}),
    ("runs out of shotgun on second and long", ["Shotgun: yes", "Down: 2nd", "Yards to go ≥ 7"], ["out"], ["shotgun", "down", "ydstogo"],
     {"add": [], "remove": [], "unsupported": ""}),
    ("short throws in the second half", ["Quarter: Q3, Q4"], ["short"], ["air_yards", "qtr", "pass_length"],
     {"add": [{"column": "air_yards", "op": "lte", "value": 9}], "remove": [], "unsupported": ""}),
]


def _small_user(query: str, draft: list[str], ignored: list[str], cand_lines: list[str]) -> str:
    return (f"Question: {json.dumps(query)}\nApplied: {'; '.join(draft) or '(none)'}\nLeftover words: {json.dumps(ignored)}\n"
            f"Columns:\n" + "\n".join(cand_lines))


def _cand_line(f: dict) -> str:
    if f["choices"]:
        vals = "one of " + ", ".join(f"{v} ({lab})" for v, lab in f["choices"])
    elif f["kind"] == "bool":
        vals = "true/false"
    elif f["kind"] == "range":
        vals = "number" + (f" - {f['help']}" if f.get("help") else "")
    else:
        vals = "text"
    return f"- {f['col']}: {f['label']} ({vals})"


def small_messages(query: str, p: Parse) -> list[dict]:
    """System prompt + worked examples + this question, as chat messages for a small model."""
    msgs = [{"role": "system", "content": SMALL_SYSTEM}]
    for q, draft, ign, cols, ans in SMALL_EXAMPLES:
        lines = [_cand_line(BY_COL[c]) for c in cols if c in BY_COL]
        msgs.append({"role": "user", "content": _small_user(q, draft, ign, lines)})
        msgs.append({"role": "assistant", "content": json.dumps(ans)})
    ds = p.dataset or "passes"
    draft = [describe(k.split("__")[0], (k.split("__") + ["eq"])[1], v) for k, v in p.params.items()]
    msgs.append({"role": "user", "content": _small_user(query, draft, p.ignored, [_cand_line(f) for f in _small_candidates(p, ds)])})
    return msgs


FT_SYSTEM = ("Map leftover words of an NFL play question to filters. Reply with JSON "
             '{"add": [{"column", "op", "value"}], "remove": [columns], "unsupported": "words no filter can express"}.')


def ft_messages(query: str, p: Parse) -> list[dict]:
    """The compact prompt the fine-tuned Ask-bar model was trained on (it knows the filters; no menu needed)."""
    return [{"role": "system", "content": FT_SYSTEM},
            {"role": "user", "content": f"dataset: {p.dataset or 'passes'}\nquestion: {query}\n"
                                        f"applied: {json.dumps(p.params, sort_keys=True)}\nleftover: {json.dumps(p.ignored)}"}]


NAMED = {"receiver_player_name"}   # set from names the rules matched exactly; a model neither adds nor removes them


def _apply_small(p: Parse, out: dict, dyn: dict):
    """A small model's {add, remove} on top of the rules' draft, every filter validated like Gemini's."""
    changed = []
    for col in out.get("remove") or []:
        if col in NAMED:
            continue
        for k in [k for k in p.params if k.split("__")[0] == col]:
            changed.append("removed " + describe(col, (k.split("__") + ["eq"])[1], p.params.pop(k)))
    for item in out.get("add") or []:
        col, op, value = item.get("column"), item.get("op", "eq"), item.get("value")
        f = BY_COL.get(col)
        if not f or op not in OP_SUFFIX or col in NAMED:
            continue
        try:
            v = _coerce(f, op, value, dyn)
        except (ValueError, TypeError):
            continue
        if f["kind"] == "choice" and op == "eq":
            op = "in"
        elif op == "in" and f["kind"] != "choice":
            op = "eq"
        if op in ("eq", "in"):
            for k in [k for k in p.params if k.split("__")[0] == col]:
                del p.params[k]
        p.params[col + OP_SUFFIX[op]] = v
        changed.append(describe(col, op, v))
    if out.get("unsupported"):
        p.notes.append(f"Couldn't express: {out['unsupported']}")
    for c in changed:
        p.understood.append({"text": "AI", "meaning": c})


def llm_schema() -> dict:
    """JSON schema for a model's answer to _prompt: the same shape, with column / op held to the curated lists so a
    small model's constrained decoding can't produce anything _apply_llm would have to throw away."""
    cols = sorted(set(BY_COL) | {"defteam"})
    value = {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"},
                       {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "number"}]}}]}
    return {
        "type": "object",
        "properties": {
            "dataset": {"type": "string", "enum": list(DATASETS)},
            "seasons": {"type": "array", "items": {"type": "integer"}},
            "all_seasons": {"type": "boolean"},
            "weeks": {"type": "array", "items": {"type": "integer"}},
            "team": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "opponent": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "player": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "filters": {"type": "array", "items": {
                "type": "object",
                "properties": {"column": {"type": "string", "enum": cols},
                               "op": {"type": "string", "enum": list(OP_SUFFIX)}, "value": value},
                "required": ["column", "op", "value"]}},
            "unsupported": {"type": "string"},
        },
        "required": ["dataset", "filters"],
    }


def refine(query: str, players: Players, current_dataset: str, dyn: dict, out: dict) -> Parse:
    """The rules' parse with an in-browser model's answer to its prompt applied - validated exactly like Gemini's."""
    p = parse_rules(query, players, current_dataset, dyn)
    if not p.ignored or p.refused:
        return p
    _apply_small(p, out if isinstance(out, dict) else {}, dyn)
    _validate(p, current_dataset, dyn)
    p.ignored = []
    p.engine = "rules + local model"
    return p


def parse(query: str, players: Players, current_dataset: str, dyn: dict, llm: str = "server") -> Parse:
    """Rules first; for whatever the rules left over, Gemini (llm="server") - or, with llm="client", the prompt and
    schema for the visitor's in-browser model, whose answer comes back through refine()."""
    from . import gemini

    p = parse_rules(query, players, current_dataset, dyn)
    if not p.ignored or p.refused:
        return p
    if llm == "client":
        p.llm_prompt, p.llm_schema = small_prompt(query, p)
        p.llm_messages = small_messages(query, p)
        p.llm_ft = ft_messages(query, p)
        p.engine = "rules (in-browser model to refine)"
        return p
    try:
        out = gemini.ask(_prompt(query, p, players.seasons, dyn))
    except gemini.GeminiUnavailable as e:
        p.engine = f"rules (Gemini unavailable: {e})"
        p.notes.append("Some words weren't understood: " + ", ".join(p.ignored))
        return p
    _apply_llm(p, out, players, dyn)
    _validate(p, current_dataset, dyn)
    p.ignored = []
    p.engine = "rules + Gemini"
    return p
