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
- [Plotting: `plot` and `draw_field`](#plotting-plot-and-draw_field)
- [Matching charts to play-by-play](#matching-charts-to-play-by-play)
  - [`load_pbp_with_ftn`](#load_pbp_with_ftn)
  - [`match_to_pbp`](#match_to_pbp)
  - [Match confidence columns](#match-confidence-columns)
  - [Re-reading tangled route charts: `retrace_routes`](#re-reading-tangled-route-charts-retrace_routes)
  - [How matching works](#how-matching-works)
- [Explorer web app](#explorer-web-app)
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
pip install "next-gen-scrapy[plot]"      # + plotting on an NGS-style field (adds matplotlib)
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
| `located` | `False` if the ring couldn't be placed (drawn completely under another ring) - then the coordinates are empty rather than guessed |
| `extrapolated` | `True` for a ring past the chart's deepest gridline (the art stops at about +50): its depth is extrapolated from the calibration - checked against play-by-play air yards, as accurate as the rest (median 1.4 yds) |
| `x_coord`, `y_coord` | Where the ball was caught / targeted |

Every completion, touchdown and interception on the chart gets a row - the counts come from the chart's own stat box
(read off the image, so a chart made before the game ended, or one NGS's listing has no stats for, still ties out).
Rings the main pass can't separate are found by a second one: ones on the LOS bar, a sideline or under a yard label,
in the logo corner, tinted by an arc over them, two touchdowns drawn on top of each other, a touchdown at the end of
its arc, a 2018-19 touchdown disc at the foot of its arc. Of the 96,388 completion / touchdown / interception rings
drawn on 4,451 charts (2018-2026), 96,308 (99.92%) are located; the rest are covered completely by other rings and
arcs. 11 more touchdowns have their arc run off the top of the chart: their ring isn't drawn.
Incompletions are **not** padded: the chart genuinely doesn't draw every incompletion (throwaways, spikes...).
White rings are found the same ways - including one mostly under a completion / touchdown / interception ring, or
a tiny far one by a yard label - checked so that charts whose white rings already match their targeted
incompletions gain none.

### `route_coords.csv`

One row per point along each route (0.5 yd spacing, ordered from the start of the route).

| Column | Meaning |
|---|---|
| `route_id` | 1, 2, ... within this player's chart (unique per `game_id` + `esb_id`, not per game) |
| `route_type` | `COMPLETE` (white line) or `INCOMPLETE` (grey line) |
| `touchdown` | Route ends in a touchdown ring |
| `start_ok` | `False` if the start couldn't be traced back to the line of scrimmage (starts > 1 yd ahead of it; ~6% of routes) |
| `gap` | `True` if part of the route is a straight jump the chart doesn't draw: a TD ring NGS drew with no line leading into it (~0.7% of charts), joined to the route that most plausibly scored |
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
| `gap` | `True` if part of the carry is a straight jump the chart doesn't draw: a TD ring with no line reaching it (~1% of charts), joined to the green carry that most plausibly scored |
| `point`, `x_coord`, `y_coord` | The point |

### `chart_qc.csv`

One row per chart, comparing what was detected with the chart's own metadata (the box score).

| Column | Meaning |
|---|---|
| `type` | `pass`, `route` or `carry` |
| `image` | Path of the chart image |
| `depth_yd` | Deepest yard line the chart's calibration verified (route / carry coordinates beyond it are dropped; pass rings beyond it are kept, flagged `extrapolated`) |
| `expected_*` / `detected_*` | Completions, touchdowns, interceptions, incompletions (pass); receptions, targets, routes, TD rings (route); carries, TD rings (carry) |
| `n_deep_truncated`, `n_deep_dropped` | Lines cut / dropped at the verified depth |
| `n_extrapolated` | Pass rings placed past the verified depth (see `extrapolated`) |
| `stats_from` | Where the expected counts came from: `image` (the chart's stat box, read off it) or `listing` (NGS's chart listing) |
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

## Plotting: `plot` and `draw_field`

```python
from next_gen_scrapy import get_routes, plot

routes = get_routes("Justin Jefferson", 2024)
plot(routes, kind="route", title="Justin Jefferson 2024 routes")        # every route
plot(routes, kind="route", heatmap=True)                                # where his routes go
plot(routes, kind="route", heatmap=True, heat_of="end")                 # where he's targeted
```

<p>
  <img src="https://raw.githubusercontent.com/rj7002/next-gen-scrapy/main/docs/plot_routes.png" width="49%" alt="Justin Jefferson's 2024 routes on an NGS-style field">
  <img src="https://raw.githubusercontent.com/rj7002/next-gen-scrapy/main/docs/plot_heatmap.png" width="49%" alt="Heatmap of Josh Allen's 2024 pass locations">
</p>

Needs matplotlib (`pip install "next-gen-scrapy[plot]"`).

### `plot(data, kind="pass", heatmap=False, ax=None, title=None, legend=True, heat_of="path", smooth=1.5, cmap="inferno", colorbar=True, ymin=None, ymax=None, figsize=None)`

Draws passes, routes or carries on a Next Gen Stats-style field and returns the matplotlib `Axes`.

| Argument | Meaning |
|---|---|
| `data` | Anything this package produces for that kind: `get_*` output, the `ngs-extract` CSVs, [`to_paths`](#one-row-per-route-or-carry-to_paths) output, or [`match_to_pbp`](#match_to_pbp) output. Point-per-row routes/carries are grouped into paths for you. Filter it first to plot a situation (3rd downs, play action, one coverage, ...) |
| `kind` | `"pass"`, `"route"` or `"carry"` |
| `heatmap` | `False`: every play in the charts' own colours. `True`: a smoothed density of where they happened |
| `heat_of` | Route/carry heatmaps only. `"path"`: everywhere the route/carry went, each play counting once in total. `"end"`: the catch/target point (routes) or where the carry ended |
| `smooth` | Heatmap smoothing radius, in yards |
| `cmap` | Heatmap colour map (any matplotlib name) |
| `ax` | Draw into an existing axes (e.g. one panel of `plt.subplots`) instead of a new figure |
| `title`, `legend`, `colorbar` | As named. The legend and colour bar sit under the field |
| `ymin`, `ymax` | Depth range in yards past the LOS. Fits the data when `None` |
| `figsize` | Figure size when `ax` is `None`. Defaults to a width of 7 in and a height that keeps the field to scale |

What normal mode draws:

| `kind` | Marks |
|---|---|
| `"pass"` | Rings coloured by `pass_type`: green complete, white incomplete, red interception, blue touchdown |
| `"route"` | White (complete) or grey (incomplete) route up to the catch point, a dot at the catch, green yards after the catch, and a blue ring on touchdowns |
| `"carry"` | Paths coloured by `gain_class`: green 5+ yds, yellow 0-5, red loss. Blue ring on touchdowns, red ring on lost fumbles |

Line widths thin out automatically as the number of plays grows, so a full season stays readable.

### `draw_field(ax=None, ymin=-10, ymax=40, figsize=None, labels=True)`

Just the empty field: turf, sideline bands, 5- and 10-yard lines with `+10`/`+20` labels, 1-yard ticks
along the hashes and sidelines, and the blue line of scrimmage. It uses the data's own coordinates, so
anything you draw afterwards with `ax.plot` / `ax.scatter` lines up. Returns the `Axes`.

```python
import matplotlib.pyplot as plt
from next_gen_scrapy import draw_field, plot

ax = draw_field(ymin=-5, ymax=30)
ax.scatter([-10, 12], [8, 22], color="white")            # your own marks, in field yards

# side-by-side panels; `passes` here is match_to_pbp output, so pbp columns like `down` exist
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
plot(passes[passes.down == 3], kind="pass", ax=axes[0], title="3rd down")
plot(passes[passes.down != 3], kind="pass", ax=axes[1], title="Other downs")
```

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

#### `load_pbp_with_ftn(seasons, participation=True, gamecenter=True, gamecenter_cache=None, ngs_distance=True, ngs_distance_cache=None)`

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
- **NGS Game Center** (`gamecenter=True`, one request per game; `gamecenter_cache=` a folder to keep them in) -
  NGS's own per-game numbers, on every pass play. Since 2023 these are the only public source of NGS's measured air
  yards, and `match_to_pbp` uses them to pin passes down:

  | Column | Meaning |
  |---|---|
  | `ngs_receiver_targets`, `ngs_receiver_avg_air_yards` | The targeted receiver's targets in the game and NGS's average air yards over them (each team's top five receivers) - his targets' depths must add up to this |
  | `ngs_passer_zones` | The passer's completions and yards per zone, as `"depth,section,completions,yards;..."` (depth 0 behind the line / 1 0-9 / 2 10-19 / 3 20+ air yards; section 0-2 the left / middle / right third of the field) - the completions drawn in a zone must add up to its yards |
- **NGS ball-carrier distance** (`ngs_distance=True`, ~1,400 requests a season - two lists per team-week;
  `ngs_distance_cache=` a folder for finished weeks): how far the runner or receiver went on the play, from NGS's
  public leader lists, for every game 2018 on (~91% of runs, ~97% of catches). It is the length of the carry's or
  completed route's chart line, so `match_to_pbp` uses it to tell a back's look-alike runs apart.

  | Column | Meaning |
  |---|---|
  | `ngs_carrier_dist` | Yards the ball carrier covered on the play (NGS tracking) |
  | `ngs_max_speed` | His top speed (mph) |
  | `ngs_dist_below` | For a play missing from its team-week's full list: it went less far than this |

  `add_gamecenter(pbp, cache_dir=None)` adds these to a pbp table you already have; `next_gen_scrapy.gamecenter`
  has the raw tables (`game_tables(game_ids)`).

Returns a pandas DataFrame, one row per pbp play.

### `match_to_pbp`

#### `match_to_pbp(chart_df, pbp_df, kind, route_matches=None, return_params=False, calibration="auto", context=None, tracking=None)`

| Argument | Meaning |
|---|---|
| `chart_df` | Chart data as returned by `get_*` / read from the CSVs (point-per-row for routes and carries is fine; it's reduced internally) |
| `pbp_df` | Plain nflverse pbp, or `load_pbp_with_ftn`'s output. Uses FTN `starting_hash`, participation `ngs_air_yards` / `target_route` / `offense_formation` when present; works without them, less precisely |
| `kind` | `"pass"`, `"route"` or `"carry"` |
| `route_matches` | *(passes only)* The output of `match_to_pbp(..., "route")` for the same games. A targeted receiver's route chart shows the same throw again (its catch point), which is used as extra evidence of where the ball went. **Match routes first, then pass them in.** |
| `return_params` | Also return the calibrated model parameters: `(matched, params)` |
| `context` | *(carries)* match_to_pbp output for the same games' other charts (QB passes, routes, other backs): where each play ended, so where the next snap was |
| `tracking` | *(carries)* tracking from your own copy of the NFL Big Data Bowl 2027 data: `next_gen_scrapy.bdb.load_play_tracking(season)` gives every tracked player (the 2023-25 draft classes - someone is on the field for 64% / 90% / 98% of 2023 / 24 / 25 runs) at the snap, handoff, first contact and end of each play. The matcher scores each line's end against where those players stood when the run ended (the tackler is at the runner, the others have converged on him), and against the ball's spot at the next snap from tracked offensive linemen. (`load_tackle_points` - tacklers only - still works.) Not shipped; the app picks it up from `$NGS_BDB_DIR` or `~/Downloads/nfl-big-data-bowl-2027` |
| `calibration` | How tight each piece of evidence is taken to be. `"auto"` (default): the league-wide calibration shipped with the package for the data's season (the nearest season if not shipped), so matching one player gives exactly what matching the whole league gives for that player. `"self"`: fit it on this data instead - fine for a whole season of charts, but one player's charts give it little to go on. Or a params dict from `return_params=True`. |

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
| `implausible` | *(carries)* The line can't be this play - its end is 4+ yds from the play's gain, or 2+ yds off in the wrong colour. The one-to-one assignment had to place it, but no play fits (usually a misread line: a short run hidden under a longer one, read as a branch of it). Its probabilities are capped at 0.08 - about how often these turn out right. The app shows such lines as "no play fits". |
| `match_cost`, `match_margin` | The matched pair's negative log-likelihood, and how much worse the next-best play would have fit (nats) |
| `swap_yds` | The farthest a plausible rival chart entity sits from the chosen one (0 if none) |
| `residual` | `|pbp yardage - chart yardage|` (air yards for pass/route, yards gained for carries), for reference |

Both probabilities are checked against evidence the model doesn't use (see [How matching works](#how-matching-works)).

### Re-reading tangled route charts: `retrace_routes`

Where several routes cross, a reading can join the wrong pieces and give one route another route's catch point -
then no matching can put that play in the right place. `retrace_routes` re-reads those charts: for every chart
where a catch's match is unsure - or, given `pass_matches=` (the QB charts' passes, matched without route links),
where a catch lands 2.5+ yds from the QB's dot for the same throw - it collects alternative readings (the original
tracer's different ways of joining the crossing pieces, plus both stroke-graph readers' readings), scores each -
and the chart's current reading - against that receiver's real targets in that game and the QB's dots, and keeps
whichever fits best. Graded on tracked receivers' real catch points (Big Data Bowl, 2024-25), completed routes
whose catch lands within 2 yds of the real one go from 83.6% to 88.5%; `images=` takes charts already in memory
(the app re-reads every route chart it uses this way). Checked against the QBs' pass charts (independent of it), it
lifts route matching by 1-2 points even on top of the graph reader (2021: 96.1% -> 98.1%, 2024: 94.8% -> 95.9%).

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
| Pass | **NGS Game Center totals**: each receiver's targets must add up to his NGS air yards, and the completions in each of the passer's zones (depth band x third of the field) to the zone's yards - every round, each pass gets the depth and yards its receiver's other targets and its zone's other completions leave for it, and the assignment is redone until nothing changes · Depth vs NGS air yards (2018-22) or pbp air yards, with a separate offset and spread per outcome and depth band (a completion's dot sits ~2x closer to its air yards than an incompletion's, and short passes closer than deep ones) · across-field spot vs pbp pass location **and** NGS route type of the target (an out or go lands near the sideline, a slant inside), shifted by the hash the ball was snapped from (FTN, 2022+: a throw from the left hash lands ~3 yds further left) · the receiver's route-chart catch point for the same throw, when there is one · for incompletions, how likely NGS was to draw the play at all (it skips ~22%, mostly passes with no intended receiver) |
| Route | Catch depth and spot as for passes · after-catch run vs pbp yards after catch · ends at the sideline vs pbp out of bounds · **the line's shape vs NGS's route type for the target** (a small shape model, `route_type.py`, gives each line's probability of being a GO, SLANT, OUT, ...) · route length before the catch vs time to throw · the QB's pass-chart dot for the same throw (`pass_matches=`, completions) · for completions, where the line ends vs FTN's hash for the next snap (2022+: the receiver went down outside a hash or between them) |
| Carry | End point vs yards gained (or, when a foul enforced at a spot cut pbp's yards short, the run's yards from the play description) · line colour (loss / 0-5 / 6+) vs yards gained · **the line's shape vs pbp run gap** (a small shape model, `carry_type.py`, gives each line's probability of each gap) and whether it came from shotgun (before 2022) · where the run ends across the field vs pbp run location · handoff depth vs formation (shotgun ~-4.4 yds, under center ~-6.4) · **start vs the snap spot**: FTN's starting hash (2022+) or, chained, where the previous play ended, plus where a back lines up for the formation · **how long the play took**: the shape model reads the seconds from snap to whistle off the line, vs pbp's `end_clock_time - time_of_day` (2022+) · **where the line ends vs FTN's hash for the next snap** (2022+: a run that ends outside a hash puts the next snap on it, one that ends between them in the middle) · ends at the sideline vs out of bounds · fumble ring vs fumble lost · *(with `tracking=`)* where the line ends vs where every tracked player stood when the run ended, and vs the next snap's ball spot |

**Calibrated on the whole league.** Every term's offset and spread is measured on the plays that can only match
one way (one chart entity, one pbp play), then re-fit on the first pass's confident matches across all outcomes
(the one-way plays are mostly touchdowns, which would otherwise skew it), and the match is run again. Carries keep
touchdowns and runs out of bounds apart for the yards term: their lines end further past the official spot (~1.2
and ~0.7 yds, against ~0.4 for other runs). This is
done on every chart of each season, 2018-2025, and shipped with the package (`pbp_calibration.json`), so a
single player's charts are matched with the same calibration as the whole league's
(`calibration="self"` fits it on the data passed in instead). Each term is
robust to outliers, and a missing pbp field counts as "no information", never as a match.

**Probabilities.** The assignment's uncertainty comes from Sinkhorn-balanced assignment probabilities
(respecting that each dot belongs to one play), calibrated against evidence the model being graded doesn't use:

- **Passes and routes** grade each other. A QB's pass dot and the receiver's catch point are the same throw drawn
  on two separate charts, so with each matched independently, the distances between them are a mixture of
  "same play" (as on the plays that can only match one way) and "another play in the same group". The mixture
  weight is the share matched right - checked by deliberately mis-assigning a known share, which it tracks one
  for one. Raw pass probabilities came out far too cautious and are sharpened; the route-chart link
  (`route_matches=`) is a measured geometric likelihood and enters as it is.
- **Carries** are graded directly against tracking: for 2023-25 runs where the NFL Big Data Bowl tracked the
  tackler, his spot says where the run really ended (the line ends ~1.15 yds past him; even a near-certain match
  ends within 3 yds of him only 90% of the time, which the grading allows for).
  Their `location_prob` goes through a map fitted on that (`_CARRY_LOC_CALIBRATION`, with and without FTN's hashes),
  and `match_prob`'s temperature was set against FTN's starting hash and participation's formation, each left out
  of the model it grades.

For passes the result then goes through a calibration map (`_LOC_CALIBRATION`, one with NGS per-play air yards and
one without): the raw numbers ran low at the bottom - they count a rival dot as a threat even when another play
has already claimed it - and a little high in the upper middle, so e.g. a raw 0.66 that proves right 79% of the
time is reported as ~0.8. A play with a single candidate stays at 1.

The temperatures are `_TEMP` in `pbp_match.py`. Calibration never changes which play is chosen, only the
reported confidence.

**The limit.** Plays that look the same to both the chart and pbp - same player, outcome, depth and side,
e.g. two 2-yard runs up the middle - can't be told apart by anyone. The matcher then picks one, and
`match_prob` says so; if their dots sit together, `location_prob` still says the location is right.

---

## Explorer web app

`service/` is a web app on top of the package. Pick any player and season: it pulls that player's charts live from
Next Gen Stats, reads them, matches every pass, route or carry to its play-by-play row with
[`match_to_pbp`](#match_to_pbp), and draws the result. There's no database; results are cached in memory while
the server runs. A QB's passes are matched first, then pinned again with their receivers' route charts.

<p align="center">
  <img src="https://raw.githubusercontent.com/rj7002/next-gen-scrapy/main/docs/explorer.png" width="100%" alt="Ja'Marr Chase's 2024 routes in the explorer, colored by route type on a tilted field">
</p>

- **The field**: every play as an animated route, carry or pass dot (hover for the play, click for its detail
  sheet), or as a heatmap; color by outcome, EPA, route type, down or match confidence; optional 3D tilt.
- **Filters**: play-by-play, FTN and NGS participation columns (down, coverage, pressure, formation...).
- **Ask in plain English**: "deep passes vs zone in the 4th quarter" sets the filters, and can switch player
  or season. A built-in parser handles most questions; with `GEMINI_API_KEY` in `.env`, Gemini fills in the words
  it doesn't understand.
- **Insights, plays table, games**, and a CSV export of the full matched data.

Run it from a clone of the repo (the UI build needs Node 20.19+ or 22.12+):

```
pip install -e ".[pbp]" fastapi "uvicorn[standard]"
cd service/web && npm install && npm run build && cd ../..
uvicorn service.app:app --port 8000        # open http://127.0.0.1:8000
```

Players load from pre-read, pre-matched season files when the app can reach them: the
[NFL-NGS-Tracking-Data](https://github.com/rj7002/NFL-NGS-Tracking-Data) pipeline publishes every season (and refreshes
the current one daily) as assets of its `data` release. Charts in those files load in a couple of seconds; anything
newer is read live. Set `NGS_DATA_TOKEN` to a GitHub token that can read that repo (it's private), or `NGS_DATA_REPO` to
use another; without them every chart is read live, which takes from a few seconds to about two minutes the first time,
depending on how many charts the player has and whether that season's play-by-play is already loaded. To work on the UI, run uvicorn with
`--reload`, and `npm run dev` in `service/web` (Vite proxies `/api` to port 8000). The UI is React with
[shadcn/ui](https://ui.shadcn.com) components.

**Deploying it.** `deploy/huggingface/` holds a Docker setup for a [Hugging Face Space](https://huggingface.co/docs/hub/spaces-sdks-docker)
(the free CPU tier is enough): after `hf auth login`, `deploy/huggingface/push.sh <you>/<space-name>` stages just the
package and `service/` (no caches, data or `.env`) and pushes it; the Space builds the UI and starts the server on
port 7860. Add `GEMINI_API_KEY` as a Space secret if you want the "ask" box. The deployed app runs with
`NGS_API_GUARD=1`: `/api` then answers only the app's own page in a browser (same-site requests carrying the session
cookie the page sets, rate-limited per visitor), so the API can't be scripted casually. Set `NGS_SESSION_SECRET` to
keep sessions valid across restarts. A local Big Data Bowl copy is never
uploaded, so a deployed app matches 2023-25 carries without tracking.

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

#### `detect_routes(image, expected=None, method="best") -> (routes, info)`

Traces every route. `expected = {"receptions", "touchdowns", "targets", "max_yards"}` (targets set how many grey
routes to look for - when they're missing, 2023+ charts have them printed in the stat box and they are read from
there; `max_yards` - the game's receiving yards - caps how deep one route can be). Returns a list of
`dict(route_type, td, pts, seg, start_ok, gap)` (`pts` = `Nx2` yards, `seg` = `route` / `after_catch` per point,
`gap` = the route includes an undrawn jump into a TD ring; the tracer never sets it) and
an `info` dict (`method` = which reader produced it, `depth_yd`, TD rings, deep truncations, ...).

`method`: `"graph"` reads every route at once from one stroke graph of all lines (`route_graph.py`); `"layers"`
does the same from one skeleton per colour, which keeps lines of different colours whole where they touch or
overlap (an after-catch run coming back along its own route, a grey route sharing a white route's stem);
`"tracer"` is the original per-colour tracer; `"best"` (default) tries them in that order, stops as soon as one
explains the chart perfectly, and otherwise keeps the reading that explains it best
(`routes.reading_score`: counts vs the metadata, share of drawn line pixels covered, share covered by a route of
the matching colour, penalties for impossible structure).

#### `next_gen_scrapy.routes.route_alternatives(image, expected, max_alts=40) -> [(geo_cost, routes, info), ...]`

Several complete readings of one chart, for when crossing routes can be joined more than one way. The first
is always `detect_routes`' own reading (`geo_cost` 0); the rest are ordered by how natural their joins are.
Used by [`retrace_routes`](#re-reading-tangled-route-charts-retrace_routes).

### `detect_carries`

#### `detect_carries(image, expected=None, method="best") -> (carries, info)`

Traces every carry. `expected = {"carries", "touchdowns", "max_yards"}`. Returns a list of
`dict(color, td, fumble, handoff_ok, pts, gap)` (`color` = `LOSS` / `SHORT` / `LONG`, `pts` in yards, starting in
the backfield; `gap` as for routes, never set by the tracer) and an `info` dict (`method`, `depth_yd`, `n_lines`,
`n_td_rings`, `n_fumble_rings`, deep truncations).

`method`: `"layers"` reads every carry at once from one stroke graph per colour (`carry_graph.py`); `"graph"` does
the same from one skeleton of all lines; `"tracer"` is the original line tracer. `"best"` (the default) keeps the
`"layers"` reading unless it fails a sanity check (counts, each carry's colour vs where it ends, line coverage),
in which case it also tries the other two and keeps the reading that best explains the chart. Without a carries
count only the tracer runs.

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
- **Carries** (`carry_graph.py`): each colour is skeletonised on its own (a carry is one colour from handoff to
  tackle), and NGS's drawing order - red, then yellow, then green on top - says where a line can be hidden: a red
  line only under yellow or green, a yellow line only under green. Pieces of one line are joined across such
  stretches by smooth curves; V-shaped tips behind the LOS (two carries taken at one spot) become nodes. Candidate
  carries run from a handoff (deep behind the LOS) to a line end whose depth fits the colour - red at or behind the
  LOS, yellow 0-5 yards, green 5+ or a TD ring - and an integer program picks all of them together: #carries and
  #TDs from the metadata, every line covered, every line end explained, each lost-fumble ring ending one carry,
  with costs for implausible shapes (shallow handoffs, carries running behind their own handoff, tiny "carries"
  that are really glimpses of a line under the bundle, losses deeper than ~2.5 yards).
- **Original carry and route tracers** (`paths.py` + `carries.py` / `routes.py`): the drawn lines are
  skeletonised, split at junctions and re-joined through crossings by the smoothest continuation, bridging gaps
  where one line passes under another. After-catch (green) segments are attached to their route; touchdown and
  lost-fumble rings to the line they end. Lines sharing a start point are separated, and the metadata counts
  merge stray fragments.
- **Routes** (`route_graph.py`): every drawn line - white route, grey incomplete route, green after-catch run -
  goes into a stroke graph (crossings, line ends and colour changes are its nodes; a white -> green change is a
  catch point; every TD ring is a node a completed route can end on), built either from one skeleton of all lines
  or from one skeleton per colour, with optional bridges wherever a line is hidden: under a TD ring, the LOS bar,
  the TD label or another line, or at the catch seam. A line end that stops short of another line may attach to
  it, and dim, thin far-away after-catch runs are kept whole. Grey vs white is decided by stroke-centre brightness, so the very light
  grey of some 2023+ charts isn't read as white. Candidate routes are the cheapest smooth paths from every possible
  start near the LOS to every line end, and all routes are then chosen together by an integer program: every
  stroke covered, every loose end explained, #completed = receptions, #incomplete = targets - receptions,
  #ending on a ring = touchdowns, least turning and colour mismatch. Now and then NGS draws a TD ring (perhaps with
  a short piece of the run into it) but no line leading to it; when the TD count needs it, a completed route may
  then jump to it in a straight line, at a high cost, and the route is flagged `gap`. `detect_routes` keeps this
  reading or the tracer's, whichever explains the chart better.

Counts come from the chart metadata plus, for route charts, the **target count**: read off the chart's own stat
box on 2023+ charts (`statbox.py`), otherwise from the NGS receiving statboard (merged in by `ngs-scrape`). The
statboard omits low-volume players, so about 13% of pre-2023 route charts have no target count and fall back to
receptions only.

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

### Matching (2018-2025)

Checked against independent evidence (see [How matching works](#how-matching-works)): passes and routes against
each other's chart, carries against FTN's hash / participation's formation held out of the model. "Right" for
passes and routes means the attached location is the play's (two look-alike plays swapped on both charts still
count); carries the same, graded against the tracked tackle spots of 2023-25 runs.

| Kind | Est. right, 2018-22 | 2023-25 | `location_confident`, 2018-22 | 2023-25 | Right when confident |
|---|---|---|---|---|---|
| Passes (with `route_matches=`) | ~98%† | ~92%† | 83% | 46% | ~97-100%† |
| Routes | ~98% | ~91% | 77% | 71% | ~98-100% |
| Carries | ~78%* | ~84%* | 32% | 47% | ~99% |

\* carries' eras are 2018-21 and 2022-25 (FTN's hashes and pbp's play durations start in 2022); 2018-21 is
estimated on 2023-25 runs with those taken away. The *exact* play is right less often, ~46% / ~62% (calibrated
`match_prob`), because a back's look-alike runs in one game often end within a few yards of each other. With your
own Big Data Bowl tracking (`tracking=`, 2023-25) carries reach ~86-89% (2025: half the plays near-certain).
† completions, graded on receivers' route charts the pass match never saw. Incompletions have no clean check
(see below); their calibrated `location_prob` averages ~95% (2018-22) and ~85% (2023-25). A random pass of any
outcome: ~97% (2018-22), ~90% (2023-25). `location_prob` is at least 0.7 for 98.5% of passes (2018-22) and 94% (2023-25).

- **Passes and routes from 2023 on** are less precise because NGS stopped publishing its measured air yards after
  2022, leaving pbp's scorer-spotted ones. No free per-play source replaces them. Errors concentrate in busy games
  (a QB with 18+ completions: ~89% right in 2023+, vs ~95% with fewer), and on plays whose target has no route
  chart - pass `route_matches=` so the plays that do have one are pinned by the receiver's catch point (~98%).
- **Incomplete passes** are checked against the receiver's grey route; that check can't see much above ~90%
  (an incompletion's ball often lands yards from the receiver), but agreement doesn't drop in busier games, so
  their matching errors are few - in line with completions.
- **Carry charts that don't add up.** Given the back's pbp runs (`extract.process_carry(image, meta, plays=...)`,
  which the app does), a chart whose line colours don't match them - red for lost yards, yellow 0-5, green 6+ or a
  touchdown - is read again by every reader with pbp's colour counts held, and the reading that best fits the plays
  is kept. About 40% of charts disagree somewhere; re-reading them lifts carries landing within 3 yds of the tracked
  tackle spot from 76% to 78% (2024-25).
- **Carries** are the hard case: a back's 1-5 yard runs in one game look alike on the chart (the line's end is
  within about a yard of the official gain) and pbp has little else to tell them apart. Since 2022 FTN's hashes and
  pbp's play durations narrow it a lot; before that, a run is often matched to *a* plausible play rather than
  certainly *the* play - `match_prob` says how likely each is, and is calibrated, so filter on it (or
  `location_confident`) when the exact play matters.
- **No defender-in-coverage** is available from any free source; pbp only names defenders who made a play
  (tackler, pass defended, interception, QB hit, ...), and those columns come along with every match.

---

## What's new

0.6.2: **Match-aware chart reading.**

- **Carry and route readers take the player's plays**: where pieces of lines meet (under the line-of-scrimmage bar,
  in a bundle, at a crossing), every candidate line is scored against the player's best-fitting play on everything
  the matcher uses (`pbp_match.line_fit`) - yards, NGS distance, start spot, the next snap's hash, the QB's dot -
  so the reading whose lines look like real plays wins. Carry lines read clearly wrong (3+ yds off every run) fall
  from ~4% to ~2%; carries +1.3 pt and routes +0.4-0.7 pt within the tracked spot (2024-25).
- Carry charts are re-read when their lines can't be paired with the back's yards, or have fewer sideline ends
  than he had runs out of bounds; `load_pbp_with_ftn` adds each run's / completion's next-snap hash
  (`add_snap_context`) for that scoring.

0.6.1: **Outside data for matching: NGS ball-carrier distances.**

- **Carries and routes**: how far the ball carrier went on each play, from NGS's public leader lists
  (`next_gen_scrapy/ngs_distance.py`, added by `load_pbp_with_ftn`), against the length of the chart line - every
  season 2018 on, including the one in progress. Carries +0.6 pt (with FTN) / +1.1 pt (without) within 3 yds of
  the tracked end, routes +0.3 / +0.7 pt (without participation), more confident matches throughout. Charts that
  are re-read use it too.
- **Passes**: the hash moves a throw's expected x by depth (short throws ~2 yds per hash, deep ones not at all),
  and a completion's next-snap hash counts; routes use FTN's screen flag while participation's routes are missing.

0.6.0: **Much better matching to play-by-play, and the explorer web app.**

- **Carries**: right location for ~84% of 2022+ runs and ~78% before (graded on Big Data Bowl tracked tackle
  spots). New evidence: how long the play took (`end_clock_time`), FTN's hash for the *next* snap, the run's yards
  from the play description when a foul cut pbp's, a shape model of the line (`carry_type.py`: gap, shotgun,
  duration), the snap spot chained from the previous play, and yards offsets kept apart for touchdowns / runs out
  of bounds. Implausible matches are flagged (`implausible`). Optional `tracking=` from your own Big Data Bowl
  copy (`bdb.load_play_tracking`).
- **Passes**: NGS Game Center receiver air yards and passer zones (`load_pbp_with_ftn(gamecenter=True)`), the
  snap hash, a calibrated `location_prob`, and incompletions NGS didn't draw. The reader finds stacked and
  peeking rings and reads the chart's own stat box.
- **Routes**: a route-type model (`route_type.py`), time to throw, the next snap's hash, and the QB's pass-chart
  dot for the same throw (`match_to_pbp(kind="route", pass_matches=...)`). `retrace_routes` re-reads tangled
  charts using those dots and every reader (`pass_matches=`, `images=`), and the route reader takes catch hints.
- **Carry charts that don't add up** to the back's pbp runs by colour are re-read (`extract.process_carry(plays=)`).
- Shipped league calibration (`pbp_calibration.json`, `calibration="auto"`), so one player's charts match like
  the whole league's.
- **Explorer web app** (`service/`): any player and season read live, with filters, heatmaps, play details and
  exports; `deploy/huggingface/` runs it on a Hugging Face Space.

0.5.0: **Rebuilt route and carry readers.** Every line on a chart is now read at once - a stroke graph of the
drawn lines, with all routes / carries chosen together by an integer program - so the chart's own counts always
hold:

- **Routes**: receptions, incompletions and touchdowns all right on 99.7% of charts (was 77%). Target counts for
  2023+ charts are read off the chart's stat box. `detect_routes(method=...)` picks the reader.
- **Carries**: 94% of carry ends within 2 yards of the play-by-play yards gained (was 86%), counts right on 99.9%
  of charts, and every carry's colour now fits where it ends. `detect_carries(method=...)` picks the reader.
- New **`gap`** column on routes and carries: the rare TD ring NGS draws with no line leading into it is reached
  by a straight jump, flagged so it can be filtered out. `chart_qc.csv` records which reader read each chart.
- Requires `scipy>=1.9`.

0.4.0: **`plot`** and **`draw_field`** - passes, routes and carries on an NGS-style field, as the plays
or as heatmaps (`pip install "next-gen-scrapy[plot]"`).

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
