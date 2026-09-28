# next-gen-scrapy

Scrape NFL **Next Gen Stats** pass, route and carry charts, turn the drawn chart art back into field
coordinates, and match every dot and line to the real play it came from.

Next Gen Stats publishes per-player chart images - every pass a quarterback threw, every route a
receiver was targeted on, every carry a back took - but only as pictures. This package downloads those
images and recovers the underlying data: each pass becomes an `(x, y)` location in yards from the line
of scrimmage, each route or carry becomes an ordered path of `(x, y)` points along the field, and (with
the optional `pbp` extra) each one is matched to its nflverse play-by-play row, so you get down,
distance, EPA, coverage, formation and everything else about the play alongside the coordinates.

```
pip install next-gen-scrapy
```

```python
from next_gen_scrapy import get_passes

passes = get_passes("Josh Allen", 2025, weeks=[1, 2])      # downloads the charts into memory, returns a DataFrame
# columns: game_id, season, season_type, week, team, esb_id, name, position,
#          pass_type (COMPLETE / INCOMPLETE / INTERCEPTION / TOUCHDOWN), located, x_coord, y_coord
```

> **Status: revived for the current Next Gen Stats site.** The site is now a single-page app and the chart art
> was redesigned, so the original HTML-scraping and image-undistortion code no longer works. It is preserved in
> [`archive/`](https://github.com/rj7002/next-gen-scrapy/tree/main/archive) (2017-2018 layout); everything below replaces it. Checked against every season from
> 2018 to 2025.

---

## Contents

- [Installation](#installation)
- [Three ways to use it](#three-ways-to-use-it)
- [Coordinates](#coordinates)
- [Player API: `get_passes`, `get_routes`, `get_carries`](#player-api)
- [Command-line pipeline: `ngs-scrape` and `ngs-extract`](#command-line-pipeline)
- [Output data reference](#output-data-reference)
- [One row per route or carry: `to_paths`](#one-row-per-route-or-carry-to_paths)
- [Matching charts to play-by-play](#matching-charts-to-play-by-play)
  - [`load_pbp_with_ftn`](#load_pbp_with_ftn)
  - [`match_to_pbp`](#match_to_pbp)
  - [Match confidence columns](#match-confidence-columns)
  - [Re-reading tangled route charts: `retrace_routes`](#re-reading-tangled-route-charts-retrace_routes)
  - [How matching works](#how-matching-works)
- [One-image API](#one-image-api)
  - [`calibrate` and `Chart`](#calibrate-and-chart)
  - [`detect_passes`, `detect_blue_rings`](#detect_passes-detect_blue_rings)
  - [`detect_routes`, `route_alternatives`](#detect_routes-route_alternatives)
  - [`detect_carries`](#detect_carries)
- [How extraction works](#how-extraction-works)
- [Accuracy and known limitations](#accuracy-and-known-limitations)
- [What's new](#whats-new)
- [Credits](#credits)

---

## Installation

```
pip install next-gen-scrapy              # scraping + coordinate extraction
pip install "next-gen-scrapy[pbp]"       # + matching to nflverse play-by-play (adds nflreadpy, pyarrow)
```

Python 3.9+ (developed on 3.14). No R needed. Core dependencies: `requests`, `numpy`, `pandas`, `scipy`,
`scikit-learn`, `scikit-image`, `opencv-python-headless`.

From a clone of this repo: `pip install -e ".[pbp]"`.

---

## Three ways to use it

| You want... | Use | Saves images? |
|---|---|---|
| One player's charts as a DataFrame, quickly | [`get_passes` / `get_routes` / `get_carries`](#player-api) | No - downloaded into memory |
| Whole seasons / all teams, reproducibly, as CSVs | [`ngs-scrape` + `ngs-extract`](#command-line-pipeline) | Yes - `Pass_Charts/`, `Route_Charts/`, `Carry_Charts/` |
| To run the image processing yourself, one chart at a time | [`calibrate`, `detect_*`](#one-image-api) | Up to you |

Whichever you use, the resulting DataFrames/CSVs have the same columns, and any of them can be fed to
[`match_to_pbp`](#match_to_pbp) to attach the real play to every row.

---

## Coordinates

All coordinates are in **yards on the field**:

- `x_coord`: across the field. `0` is the middle; `-26.67` is the left sideline and `+26.67` the right
  sideline, as seen from behind the offense. Points slightly outside the sidelines are real (out-of-bounds
  throws, runs out of bounds).
- `y_coord`: downfield, in **yards past the line of scrimmage**. Negative = behind the line (backfield,
  screens). This is the same zero point as pbp's `air_yards` and `yards_gained`.

Routes and carries are resampled to one point every 0.5 yd along the path, ordered from where the route/carry
starts to where it ends.

---

## Player API

```python
from next_gen_scrapy import get_passes, get_routes, get_carries

get_passes("Josh Allen", 2025)                          # every charted pass, every week found
get_routes("Justin Jefferson", 2024, weeks=[1, 2, 3])   # just those weeks
get_carries("Bijan Robinson", 2025, weeks=["post-19"])  # a playoff game
```

### `get_passes(player, season, weeks=None, quiet=False)`
### `get_routes(player, season, weeks=None, quiet=False)`
### `get_carries(player, season, weeks=None, quiet=False)`

Page the Next Gen Stats chart API for one player's charts in one season, download each matching chart image
**into memory** (nothing is written to disk), extract it, and return one DataFrame.

| Argument | Meaning |
|---|---|
| `player` | The player's display name exactly as Next Gen Stats writes it, e.g. `"Ja'Marr Chase"`, `"Amon-Ra St. Brown"`. Case and extra spaces don't matter. A near-miss raises `ValueError` listing the closest names found. |
| `season` | Season year, e.g. `2025` (int or str). |
| `weeks` | Optional list of week labels: `1`-`18` for the regular season, `"post-19"`, `"post-20"`, ... for the playoffs. `None` = every week found. Weeks with no chart produce a warning, not an error. |
| `quiet` | `True` to suppress the progress messages. |

**Returns** a DataFrame with the same columns as the CSVs `ngs-extract` writes:
[`pass_locations.csv`](#pass_locationscsv) for `get_passes`, [`route_coords.csv`](#route_coordscsv) for
`get_routes` (one row per point), [`carry_coords.csv`](#carry_coordscsv) for `get_carries` (one row per point).
Use [`to_paths`](#one-row-per-route-or-carry-to_paths) for one row per route/carry.

**Raises** `ValueError` if no chart for that player exists in that season (with suggestions), or none for the
requested weeks. A chart that fails to process is skipped with a warning.

Charts are only published for players with enough volume in a game (NGS's choice), so not every player-game has one.

---

## Command-line pipeline

Two console scripts are installed with the package. Both read and write relative to the current directory by
default - `cd` to wherever you want the chart folders and CSVs to live.

```
ngs-scrape --type pass  -s 2025                 # download pass charts + metadata
ngs-scrape --type route -s 2025 -t MIN KC -w 1 2
ngs-scrape --type carry -s 2024 2025
ngs-extract -s 2025                              # extract everything scraped for 2025 into CSVs
```

### `ngs-scrape`

Downloads chart images and their metadata from the same JSON API the website uses
(`/api/content/microsite/chart`, which needs a `nextgenstats.nfl.com` Referer header - handled for you).

| Option | Meaning |
|---|---|
| `--type {pass,route,carry}` | Chart type (default `pass`). |
| `-s, --seasons` | One or more seasons (default: the current year). |
| `-t, --teams` | Team abbreviations, e.g. `MIN KC` (default: all). |
| `-w, --weeks` | Week labels, e.g. `1 2 post-19` (default: all). |
| `--size {small,medium,large,extraLarge}` | Image size to download (default `extraLarge` = 1200x1200; extraction is tuned for it). |
| `--delay` | Seconds between image downloads (default 0.2). |

It **resumes where it left off**: charts already on disk are skipped. For route charts it also fetches the
receiver's **target count** from the NGS receiving statboard and stores it in the chart metadata (the chart's own
metadata only has receptions; a route chart draws one line per target, so targets bound the incomplete routes).

Files land in:

```
{Pass,Route,Carry}_Charts/<TEAM>/<season>/<week>/images/<Last>_<First>_<POS>.jpeg
{Pass,Route,Carry}_Charts/<TEAM>/<season>/<week>/data/<Last>_<First>_<POS>.json
```

The JSON is the chart's metadata from the API: `gameId`, `esbId`, `playerName`, `position`, `week`,
`seasonType`, and the game totals the chart shows (completions/attempts/touchdowns/interceptions/passingYards
for passes; receptions/targets/receivingYards for routes; carries/touchdowns/rushingYards for carries).

### `ngs-extract`

Turns every scraped image into coordinates, in parallel, and writes the CSVs.

| Option | Meaning |
|---|---|
| `--type {pass,route,carry,all}` | What to extract (default `all`). |
| `--root` | Folder holding the `*_Charts` folders (default `.`). |
| `--out` | Folder to write the CSVs to (default `.`). |
| `-s, --seasons` / `-t, --teams` / `-w, --weeks` | Limit which charts are processed. |
| `-j, --jobs` | Worker processes (default: all CPU cores). |

Writes `pass_locations.csv`, `route_coords.csv`, `carry_coords.csv` (for the types processed) and
`chart_qc.csv`. **Each run overwrites these files with just the charts it processed** - to add a season to
existing CSVs, extract into a separate `--out` folder and concatenate.

---

## Output data reference

Every file starts with the chart's identity columns:

| Column | Meaning |
|---|---|
| `game_id` | NGS's numeric game id, e.g. `2025090700` (equals nflverse's `old_game_id`) |
| `season`, `season_type` (`REG`/`POST`), `week` | When |
| `team` | The player's team abbreviation |
| `esb_id` | NGS's player id (map to nflverse's `gsis_id` via `nflreadpy.load_players()`) |
| `name`, `position` | The player, as NGS displays them |

### `pass_locations.csv`

One row per pass.

| Column | Meaning |
|---|---|
| `pass_type` | `COMPLETE`, `INCOMPLETE`, `INTERCEPTION` or `TOUCHDOWN` (ring colour on the chart) |
| `located` | `False` if the ring couldn't be placed (drawn completely under another ring, or past the chart's verified depth) - then the coordinates are empty rather than guessed |
| `x_coord`, `y_coord` | Where the ball was caught / targeted |

Every completion, touchdown and interception in the box score gets a row, so the counts always tie out.
Incompletions are **not** padded: the chart genuinely doesn't draw every incompletion (throwaways, spikes...).

### `route_coords.csv`

One row per point along each route (0.5 yd spacing, ordered from the start of the route).

| Column | Meaning |
|---|---|
| `route_id` | 1, 2, ... within this player's chart (unique per `game_id` + `esb_id`, not per game) |
| `route_type` | `COMPLETE` (white line) or `INCOMPLETE` (grey line) |
| `touchdown` | Route ends in a touchdown ring |
| `start_ok` | `False` if the start couldn't be traced back to the line of scrimmage (starts > 1 yd ahead of it; ~6% of routes) |
| `point` | 0, 1, 2 ... along the route |
| `segment` | `route` or `after_catch` (the green yards-after-catch part) |
| `x_coord`, `y_coord` | The point |

The catch point is the last `route` point before the first `after_catch` point.

### `carry_coords.csv`

One row per point along each carry, from the backfield to where the runner was stopped.

| Column | Meaning |
|---|---|
| `carry_id` | 1, 2, ... within this player's chart (unique per `game_id` + `esb_id`) |
| `gain_class` | Line colour: `LOSS` (red, < 0), `SHORT` (yellow, 0-5 yds), `LONG` (green, 6+ yds or TD) |
| `touchdown`, `fumble_lost` | Ends in a touchdown / lost-fumble ring |
| `handoff_ok` | `False` if the line couldn't be followed back to a handoff behind the LOS (~0.4% of carries) |
| `point`, `x_coord`, `y_coord` | The point |

### `chart_qc.csv`

One row per chart, comparing what was detected with the chart's own metadata (the box score).

| Column | Meaning |
|---|---|
| `type` | `pass`, `route` or `carry` |
| `image` | Path of the chart image |
| `depth_yd` | Deepest yard line the chart's calibration verified (coordinates beyond it are dropped) |
| `expected_*` / `detected_*` | Completions, touchdowns, interceptions, incompletions (pass); receptions, targets, routes, TD rings (route); carries, TD rings (carry) |
| `n_deep_truncated`, `n_deep_dropped` | Lines cut / dropped at the verified depth |
| `counts_ok` | Every count that can be checked matches. **Filter on it** if you only want charts that fully tie out |
| `error` | Why a chart couldn't be processed (e.g. a blank image NGS published) |

---

## One row per route or carry: `to_paths`

```python
from next_gen_scrapy import get_carries, to_paths

carries = get_carries("Bijan Robinson", 2025, weeks=[1, 2, 3])
per_carry = to_paths(carries, "carry")     # one row per carry
```

### `to_paths(chart_df, kind)`

Reduces point-per-row data to one row per entity, folding the path into array columns.

| `kind` | Returns |
|---|---|
| `"pass"` | The input unchanged (a pass is already one row) |
| `"carry"` | One row per `(game_id, esb_id, carry_id)`: the scalar columns, `path_x` / `path_y` (numpy arrays, in order), and `y_coord` (the end point's depth) |
| `"route"` | One row per `(game_id, esb_id, route_id)`: the scalar columns, `path_x` / `path_y` / `path_segment`, and `catch_y` (the catch point's depth - the target's air yards) |

`y_coord` / `catch_y` are the point's own value, not a difference from the line's start: the chart is already
zeroed at the LOS, so a carry starting in the backfield isn't overstated.

---

## Matching charts to play-by-play

A chart is just dots and lines per player-game, with no play ids. The functions below pair each one with its
nflverse play-by-play row (optionally enriched with FTN charting and NGS participation data), so you get every
play-level fact alongside the coordinates. Needs the `pbp` extra: `pip install "next-gen-scrapy[pbp]"`.

```python
from next_gen_scrapy import get_routes, get_passes, load_pbp_with_ftn, match_to_pbp

pbp = load_pbp_with_ftn(2024)                         # pbp + FTN + NGS participation, one season

routes = match_to_pbp(get_routes("Justin Jefferson", 2024), pbp, "route")
passes = match_to_pbp(get_passes("Sam Darnold", 2024), pbp, "pass", route_matches=routes)

# e.g. where Darnold's 3rd-down throws landed, keeping only matches whose location is 95%+ certain
third = passes[(passes.down == 3) & passes.location_confident]
```

### `load_pbp_with_ftn`

#### `load_pbp_with_ftn(seasons, participation=True)`

Loads nflverse play-by-play for `seasons` (an int or a list) and left-joins:

- **FTN charting** (2022 onward) on `(game_id, play_id)`: play action, RPO, screen, motion, QB location,
  starting hash, box count, blitzers, pass rushers, catchable / contested / drop, interception-worthy, and more.
  Seasons before 2022 get the same columns, all null, so a multi-season call has one schema. Expect some nulls
  in later seasons too (FTN doesn't chart admin rows and lags a few days behind the newest games).
- **NGS participation** (`participation=True`), renamed where noted:

  | Column | Meaning |
  |---|---|
  | `ngs_air_yards` | NGS's *measured* air yards (2018-2022 only) - ~2.5x more precise than pbp's scorer-spotted `air_yards` |
  | `time_to_throw` | Seconds from snap to throw |
  | `was_pressure` | QB was pressured |
  | `target_route` | The **targeted receiver's** route type (`GO`, `OUT`, `SLANT`, `HITCH`, `CROSS`, `SCREEN`, ...) - from participation's `route` |
  | `defense_man_zone_type` | `MAN_COVERAGE` / `ZONE_COVERAGE` |
  | `defense_coverage_type` | `COVER_0` ... `COVER_6`, `2_MAN`, `PREVENT`, ... |
  | `offense_formation` | `SHOTGUN`, `SINGLEBACK`, `I_FORM`, `EMPTY`, `PISTOL`, `JUMBO`, `WILDCAT`, ... |
  | `offense_personnel`, `defense_personnel` | e.g. `1 RB, 1 TE, 3 WR` |

Returns a pandas DataFrame, one row per pbp play.

### `match_to_pbp`

#### `match_to_pbp(chart_df, pbp_df, kind, route_matches=None, return_params=False)`

| Argument | Meaning |
|---|---|
| `chart_df` | Chart data as returned by `get_*` / read from the CSVs (point-per-row for routes and carries is fine; it's reduced internally) |
| `pbp_df` | Plain nflverse pbp, or `load_pbp_with_ftn`'s output. Uses FTN `starting_hash`, participation `ngs_air_yards` / `target_route` / `offense_formation` when present; works without them, less precisely |
| `kind` | `"pass"`, `"route"` or `"carry"` |
| `route_matches` | *(passes only)* The output of `match_to_pbp(..., "route")` for the same games. A targeted receiver's route chart shows the same throw again (its catch point), which is used as extra evidence of where the ball went. **Match routes first, then pass them in.** |
| `return_params` | Also return the calibrated model parameters: `(matched, params)` |

Returns **one row per matched play**:

- every chart column, promoted to its natural name (`name`, `team`, `position`, `season`, `week`,
  `pass_type` / `route_type` / `gain_class`, `x_coord` / `y_coord`, `catch_y`, `path_x` / `path_y` /
  `path_segment`, `located` / `start_ok` / `handoff_ok`) - where a name collides with a pbp column the chart's
  value wins;
- `chart_game_id`, `chart_esb_id`, `chart_route_id` / `chart_carry_id`, `chart_touchdown`, `chart_fumble_lost`:
  the chart's own identifiers and ring flags (pbp's `touchdown` / `fumble_lost` are whole-play flags, kept separately);
- `game_id`, `play_id`: nflverse's play key, plus **every** pbp / FTN / participation column for that play;
- `kind`, `bucket` (the outcome group it was matched within), and the [confidence columns](#match-confidence-columns).

Plays with no chart entity (e.g. incompletions NGS didn't draw) are simply absent.

### Match confidence columns

| Column | Meaning |
|---|---|
| `location_prob` | Probability that the chart location attached to this play is within 3 yds of where the play really happened. Two look-alike plays whose dots sit on top of each other can't be told apart - but swapping them doesn't move anything, so that doesn't count against this. **Use this for location-based analysis** (heatmaps, charts by situation). |
| `location_confident` | `location_prob >= 0.95` |
| `match_prob` | Probability that this chart entity is *exactly* this play (not just in the same spot). Use it when the play's identity matters, e.g. looking up one specific play. |
| `n_candidates` | How many plays in this game had this player + outcome. `1` = the only candidate, so the match is certain. |
| `match_cost`, `match_margin` | The matched pair's negative log-likelihood, and how much worse the next-best play would have fit (nats) |
| `swap_yds` | The farthest a plausible rival chart entity sits from the chosen one (0 if none) |
| `residual` | `|pbp yardage - chart yardage|` (air yards for pass/route, yards gained for carries), for reference |

Both probabilities are checked against evidence the model doesn't use (see [How matching works](#how-matching-works)).

### Re-reading tangled route charts: `retrace_routes`

Where several routes cross, the tracer can join the wrong pieces and give one route another route's catch point
- then no matching can put that play in the right place. `retrace_routes` re-reads those charts: for every chart
where a catch's match is unsure, it generates up to 40 alternative readings (different ways of joining the
crossing pieces), scores each against that receiver's real targets in that game, and keeps the best fit.

```python
import pandas as pd
from next_gen_scrapy import load_pbp_with_ftn, match_to_pbp
from next_gen_scrapy.retrace import retrace_routes

pbp = load_pbp_with_ftn(2024)
routes = pd.read_csv("route_coords.csv").query("season == 2024")
qc = pd.read_csv("chart_qc.csv")

routes, report = retrace_routes(routes, pbp, qc, root=".")   # report: {'unsure_charts': 176, 'reread': 76}
matched = match_to_pbp(routes, pbp, "route")
```

#### `retrace_routes(route_df, pbp_df, chart_qc, root=".", geo_weight=0.05, unsure_below=0.6, max_alts=40)`

| Argument | Meaning |
|---|---|
| `route_df` | Point-per-row route data (`route_coords.csv` format) |
| `pbp_df` | Play-by-play, as for `match_to_pbp` |
| `chart_qc` | `chart_qc.csv` - used to find each chart's image |
| `root` | Folder the `image` paths in `chart_qc` are relative to (where you ran `ngs-scrape`) |
| `geo_weight` | How much a less natural join is penalised, in nats per pixel of join cost - geometry only breaks near-ties |
| `unsure_below` | Re-read charts with any catch whose `location_prob` is below this |
| `max_alts` | Alternative readings tried per chart |

Returns `(route_df with re-read charts swapped in, report)`. It needs the chart **images on disk**, so it works
with the `ngs-scrape` / `ngs-extract` workflow, not with `get_routes` (which never saves images).

Lower-level pieces, if you want them: `next_gen_scrapy.routes.route_alternatives(image, expected)` (the
alternative readings of one chart, see [below](#detect_routes-route_alternatives)) and
`next_gen_scrapy.pbp_match.score_chart(chart_rows, pbp_rows, kind, params)` (how well one reading fits a
player's plays; lower is better; `params` from `match_to_pbp(..., return_params=True)`).

### How matching works

**Exact first.** The game is joined exactly (the chart's `game_id` is nflverse's `old_game_id`), the player
exactly (`esb_id` -> `gsis_id`), and within each player-game the outcome exactly: touchdown / interception /
complete / incomplete for passes and routes, touchdown / other for carries.

**Then scored.** Within each of those groups, every chart entity is scored against every candidate play by a
negative log-likelihood built from everything the chart and pbp both describe, and the optimal one-to-one
assignment (Hungarian algorithm) wins:

| Kind | Evidence |
|---|---|
| Pass | Depth vs NGS air yards (2018-22) or pbp air yards · across-field spot vs pbp pass location **and** NGS route type of the target (an out or go lands near the sideline, a slant inside) · the receiver's route-chart catch point for the same throw, when there is one |
| Route | Catch depth and spot as for passes · after-catch run vs pbp yards after catch · ends at the sideline vs pbp out of bounds |
| Carry | End point vs yards gained · line colour (loss / 0-5 / 6+) vs yards gained · where the run crosses the LOS vs pbp run gap (guard ~2 yds from the handoff, tackle ~4-5, end ~10) · handoff depth vs formation (shotgun ~-4.4 yds, under center ~-6.4) · start vs FTN starting hash (2022+) · ends at the sideline vs out of bounds · fumble ring vs fumble lost |

**Self-calibrating.** Every term's offset and spread is measured on the plays that can only match one way (one
chart entity, one pbp play), then re-fit on the first pass's confident matches across all outcomes (the
one-way plays are mostly touchdowns, which would otherwise skew it), and the match is run again. Each term is
robust to outliers, and a missing pbp field counts as "no information", never as a match.

**Probabilities.** The assignment's uncertainty comes from Sinkhorn-balanced assignment probabilities
(respecting that each dot belongs to one play), checked against held-out evidence. For passes and routes: a
QB's pass dot and the receiver's catch point are the same throw drawn on two separate charts, and their
agreement rises in step with the reported confidence, up to the ceiling that check allows. For carries the raw
probabilities came out too cautious, so they are temperature-scaled (`_TEMP` in `pbp_match.py`) to match
FTN's starting hash, fitted with the hash left out of the model.

**The limit.** Plays that look the same to both the chart and pbp - same player, outcome, depth and side,
e.g. two 2-yard runs up the middle - can't be told apart by anyone. The matcher then picks one, and
`match_prob` says so; if their dots sit together, `location_prob` still says the location is right.

---

## One-image API

For running the image processing yourself on charts you already have (e.g. from `ngs-scrape`).

```python
from next_gen_scrapy import calibrate, detect_passes, detect_routes, detect_carries

img = "Pass_Charts/MIN/2025/1/images/McCarthy_Jonathan_QB.jpeg"
im, rings, chart = detect_passes(img, {"COMPLETE": 12, "TOUCHDOWN": 1, "INTERCEPTION": 0})
field_xy = chart.img_to_field([[r["cx"], r["cy"]] for r in rings])     # pixels -> yards
```

Every `image` argument accepts a file path or an already-decoded BGR array (e.g. from `cv2.imdecode`).
`expected` counts come from the chart's metadata JSON; they are optional but make results more reliable.

### `calibrate` and `Chart`

#### `calibrate(im) -> Chart`

Measures one chart image's geometry from its own pixels (see [How extraction works](#how-extraction-works)).
Raises **`CalibrationError`** (a `ValueError`) if the geometry can't be recovered - e.g. a blank chart.

#### `class Chart`

The calibration of one image: converts between image pixels and field yards.

| Member | Meaning |
|---|---|
| `img_to_field(pts)` | `Nx2` array of `(col, row)` pixels -> `Nx2` array of `(x, y)` yards |
| `field_to_img(pts)` | `Nx2` yards -> `Nx2` pixels |
| `row_to_yard(row)` / `yard_to_row(d)` | Image row <-> yards past the LOS (the perspective model `row(d) = (A d + B) / (C d + 1)`) |
| `scale(cx, row)` | Local pixels per yard along x and y at a pixel position |
| `max_reliable_yard` | Deepest yard line the calibration verified; coordinates beyond it are extrapolation |
| `B`, `V`, `C` | LOS row, horizon row, scale |
| `left`, `right` | Sideline lines `(slope, intercept)` in pixels |
| `quality` | `yard_rms`, `n_yard_lines`, `sideline_rms`, `sideline_inliers` |
| `los_first`, `los_last` | Image rows of the drawn LOS bar |
| `sideline_mask()`, `on_card_field()`, `text_mask()`, `los_bar_mask()` | Boolean pixel masks: playing surface, field area without overlay text, the overlay text/logo, the LOS bar |

### `detect_passes`, `detect_blue_rings`

#### `detect_passes(image, expected=None) -> (im, rings, chart)`

Finds the coloured pass rings. `expected` maps `pass_type -> count`; it decides how many rings a merged blob
holds and drops the least ring-like extras (incompletions aren't constrained). Returns the image, a list of
`dict(pass_type, cx, cy)` in **pixels**, and the `Chart` to convert them.

#### `detect_blue_rings(hsv, lay, want=None) -> [(cx, cy), ...]`

Pixel centres of blue touchdown rings on any chart type (`hsv` = the image in HSV, `lay` = its `Chart`,
`want` = touchdown count, which lets a ring mostly covered by a line still be recovered).

### `detect_routes`, `route_alternatives`

#### `detect_routes(image, expected=None) -> (routes, info)`

Traces every route. `expected = {"receptions", "touchdowns", "targets", "max_yards"}` (targets bounds the grey
routes; `max_yards` - the game's receiving yards - caps how deep one route can be). Returns a list of
`dict(route_type, td, pts, seg, start_ok)` (`pts` = `Nx2` yards, `seg` = `route` / `after_catch` per point) and
an `info` dict (`depth_yd`, piece counts, orphan green segments, TD rings, glyphs removed, deep truncations).

#### `next_gen_scrapy.routes.route_alternatives(image, expected, max_alts=40) -> [(geo_cost, routes, info), ...]`

Several complete readings of one chart, for when crossing routes can be joined more than one way. The first
is always `detect_routes`' own reading (`geo_cost` 0); the rest are ordered by how natural their joins are.
Used by [`retrace_routes`](#re-reading-tangled-route-charts-retrace_routes).

### `detect_carries`

#### `detect_carries(image, expected=None) -> (carries, info)`

Traces every carry. `expected = {"carries", "touchdowns", "max_yards"}`. Returns a list of
`dict(color, td, fumble, handoff_ok, pts)` (`color` = `LOSS` / `SHORT` / `LONG`, `pts` in yards, starting in the
backfield) and an `info` dict (`depth_yd`, `n_lines`, `n_td_rings`, `n_fumble_rings`, deep truncations).

---

## How extraction works

**Every chart is calibrated from its own pixels** (`calib.py`). NGS picks the zoom per chart so the deepest play
fits (in one season the view ranges from 54 to 110 yards deep), so there is no fixed layout to hard-code:

- the blue LOS bar gives the image row of `y = 0`;
- the two grey sideline bands give `x = +/-26.67 yd`, and where they meet gives the horizon;
- the **bold 5-yard lines** pin the last degree of freedom, the scale. The chart art also draws a faint 1-yard
  grid; the bold lines are found separately by how far they stand out, and the scale is the largest one that
  puts every bold line on the 5-yard grid, refined by least squares over all of them. (Older art, e.g. 2018-19,
  has a denser faint grid that could otherwise out-vote the bold lines.) If too few bold lines are visible, it
  falls back to a vote over every detected line.

That gives a homography per chart. It calibrates 99.6% of charts (the rest are blank images NGS occasionally
publishes), reproduces a hand-measured calibration to within 0.07 yd, and recovered yard lines sit on the 5-yard
grid to about 0.02 yd. Then:

- **Passes** (`passes.py`): the coloured rings are found, with ring size predicted from the chart's geometry.
  Overlapping rings are separated using the completion/touchdown/interception counts from the metadata - those
  counts are exact, so the job is to place that many rings rather than re-count them. Touchdown arcs and the LOS
  bar are excluded.
- **Routes and carries** (`paths.py` + `routes.py` / `carries.py`): the drawn lines are skeletonised, split at
  junctions and re-joined through crossings by the smoothest continuation, bridging gaps where one line passes
  under another. After-catch (green) segments are attached to their route; touchdown and lost-fumble rings to the
  line they end. Lines sharing a start point are separated, and the metadata counts merge stray fragments.

Counts come from the chart metadata plus, for route charts, the **target count** from the NGS receiving
statboard (merged in by `ngs-scrape`). The statboard omits low-volume players, so about 13% of route charts
have no target count and fall back to receptions only.

---

## Accuracy and known limitations

### Extraction

Over a full season of scraped charts (1,870 charts: 2025 plus early 2026), counting a chart as consistent only
when every count it can be checked against matches the box score (`counts_ok`):

| Chart | Charts fully consistent | What is checked |
|---|---|---|
| Pass  | 595 / 607 (98.0%) | completions, touchdowns, interceptions |
| Route | 597 / 642 (93.0%) | receptions, touchdown rings |
| Carry | 602 / 621 (96.9%) | carries, touchdown rings |

Of 17,668 extracted passes, 17,658 (99.94%) have coordinates; the 10 without are rings drawn under another ring.
Chart depth agrees with NGS's measured air yards to within 3-9% in every season 2018-2025 (a small, steady
offset that the matcher calibrates out).

- **Incomplete passes:** the chart doesn't draw every incompletion the box score counts, so detected incompletes
  are fewer than `attempts - completions - interceptions`. Not an error, and not padded.
- **Passes** are the most reliable: a ring centre is a well-defined point.
- **Routes and carries** are traced from pixels, so a matching *count* doesn't prove every line is right: where
  several lines overlap (carries leaving the same handoff, routes crossing near the LOS) a line can be mis-split
  or mis-joined. [`retrace_routes`](#re-reading-tangled-route-charts-retrace_routes) fixes some of these for
  routes using the play-by-play. `start_ok` / `handoff_ok` flag lines whose start couldn't be traced.
- **Blank charts** NGS sometimes publishes are reported as errors in `chart_qc.csv`, not silently skipped.
- **If the chart art changes**, a chart that can't be calibrated is reported rather than silently mis-scaled;
  because calibration is per chart, a new zoom level needs no code change.

### Matching (2018-2025, after `retrace_routes`)

Average `location_prob` (estimated share of plays whose attached location is within 3 yds of where the play
happened), and the share of plays that are `location_confident`:

| Kind | Est. location accuracy, 2018-22 | 2023-25 | `location_confident`, 2018-22 | 2023-25 |
|---|---|---|---|---|
| Passes | 92% | 78% | 68% | 31% |
| Routes | 90% | 87% | 62% | 55% |
| Carries | 68% | 72% | 31% | 36% |

- **Passes from 2023 on** are less precise because NGS stopped publishing its measured air yards after 2022,
  leaving pbp's scorer-spotted ones. No free per-play source replaces them.
- **Carries** have the most look-alikes (several short runs through the same gap in one game), which neither the
  chart nor pbp can tell apart.
- **No defender-in-coverage** is available from any free source; pbp only names defenders who made a play
  (tackler, pass defended, interception, QB hit, ...), and those columns come along with every match.

---

## What's new

0.3.1: `load_pbp_with_ftn` no longer fails for a season whose NGS participation or FTN data isn't
published yet (e.g. the season in progress) - it warns and leaves those columns empty.

0.3.0 (since 0.2.3):

- **Matching** (`match_to_pbp`) rewritten: scored on every piece of evidence both sides share (see
  [How matching works](#how-matching-works)) instead of yardage alone - on independent checks, pass locations went
  from ~39% to ~69% agreeing with the receiver's chart within 3 yds, at the ceiling of that check. New
  `route_matches=` and `return_params=` arguments, and new `location_prob`, `match_prob`, `location_confident`,
  `n_candidates`, `swap_yds`, `match_margin`, `match_cost` columns.
- **`load_pbp_with_ftn`** merges NGS participation data (coverage, pressure, formation, personnel, time to throw,
  target route, NGS air yards) and no longer errors on seasons before FTN coverage (2022).
- **`retrace_routes`** and **`route_alternatives`**: re-read tangled route charts against the play-by-play.
- **Calibration fix for 2018-2019 charts**: their denser faint grid made depths ~67% too deep; the scale now
  comes from the bold 5-yard lines. Re-extract 2018-19 if you have older CSVs.
- **`to_paths` fix**: routes/carries are grouped per player (`game_id`, `esb_id`, id); previously two players'
  "route #2" in the same game were merged into one garbled path.

---

## Credits

`next-gen-scrapy` was created by **Sarah Mallepalle** and her team at Carnegie Mellon University, with
**Sam Ventura**, **Kostas Pelechrinis** and **Ron Yurko**. Their original work extracted every pass location -
completions, incompletions, interceptions and touchdowns - from the 2017-2018 NGS pass charts, and is written up
in [*Extracting NFL Tracking Data from Images to Evaluate Quarterbacks and Pass
Defenses*](https://arxiv.org/abs/1906.03339). The idea that these chart images are recoverable data at all is
theirs; this package only carries it forward. Route and carry extensions came from Arrowhead Analytics.

The original 2017-2018 pipeline (Python 3.7 + R + nflscrapR) is preserved in [`archive/`](https://github.com/rj7002/next-gen-scrapy/tree/main/archive). The current
package is a rewrite for the redesigned NGS site: per-chart calibration instead of a fixed undistortion, line
tracing for routes and carries, box-score QC on every chart, and matching to play-by-play.
