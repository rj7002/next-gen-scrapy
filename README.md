# Intro to `next-gen-scrapy`

## Summary
This is the second version released of `next-gen-scrapy`. The first generation of next-gen-scrapy was built to allow users to extract all pass locations -  completions, incompletions, interceptions, and touchdowns - from the regular season and postseason pass charts provided by Next Gen Stats beginning in 2017. This version was created by Sarah Mallepalle and her team at Carnegie Mellon University - you can find detailed description of that work here: https://arxiv.org/abs/1906.03339. Thanks to Sarah for such an amazing contribution to the public!

I have contributed an additional script which modified the original code to scrape the Next Gen Stats route charts, instead of the passing locations. This scrapes all completed, incomplete, and yards after completion using largely the same framework as the original version. The difference is that instead of a clustering algorithm to identify the center of the pass locations, the entire route is used and each pixel from that route is translated into coordinates. The outputs are the same as the pass data, with the exception of a "position" column to reflect the position of the targeted receiver. Additionally, I have added a script to scrape all of the carry charts.

Below is Sarah's description of the tool:

"The pass charts are scraped from the NFL's official Next Gen Stats website, https://nextgenstats.nfl.com/charts/list/pass. After obtaining all of the pass chart images from the website, for every pass chart, the field is then undistorted, and all pass locations on the field relative to the line of scrimmage are extracted in (x,y) coordinate format. 

The file `pass_and_game_data.csv` is the final version of all pass location data for the 2017 and 2018 regular seasons and postseasons. After all Python and R scripts are run, for every available pass chart, the data contains Game ID, home team, away team, week, season, player, type of pass, and pass location from the line of scrimmage. This repo will be maintained regularly for bug fixes and new, exciting features and updates - including wide receiver route locations coming soon! Thank you to Sam Ventura, Kostas Pelechrinis, and Ron Yurko for all your help and guidance with this project!"

> **Status: revived for the current Next Gen Stats site (built and checked on 2025; 2022 and 2024 charts spot-checked).** The site is now a
> single-page app and the chart art was redesigned, so the original HTML-scraping and image-undistortion code no
> longer works. It is preserved in [`archive/`](archive/) (2017-2018 layout); the pipeline below replaces it.
> `pass_and_game_data.csv` / `all_pass_locations.csv` are the original 2017-2018 outputs.

## What it does now

The pipeline is a proper Python package, `next_gen_scrapy` (source in [`next_gen_scrapy/`](next_gen_scrapy/)),
installed with its own two command-line tools:

| Step | Command | Module | Output |
|---|---|---|---|
| 1. Download chart images + metadata | `ngs-scrape` | `next_gen_scrapy/scrape.py` | `Pass_Charts/`, `Route_Charts/`, `Carry_Charts/` |
| 2. Extract coordinates from the images | `ngs-extract` | `next_gen_scrapy/extract.py` | `pass_locations.csv`, `route_coords.csv`, `carry_coords.csv`, `chart_qc.csv` |

```
pip install -e .                                 # developed on Python 3.14; no R needed

ngs-scrape --type pass  -s 2025                  # also: --type route, --type carry
ngs-scrape --type route -s 2025 -t MIN KC -w 1 2
ngs-extract -s 2025                              # extract everything that was scraped
```

Both commands read/write relative to the current directory (`Pass_Charts/`, `pass_locations.csv`, ...), same
as before packaging - `cd` into wherever you want the charts and CSVs to live, then run them from there.
`ngs-extract` also accepts `--root` (where the `*_Charts` folders are) and `--out` (where to write the CSVs)
if you want those somewhere else.

The extraction logic is also importable directly, if you want it in your own script rather than through the
CSVs:

```python
from next_gen_scrapy import calibrate, detect_passes, detect_routes, detect_carries
import cv2

im = cv2.imread("Pass_Charts/MIN/2025/1/images/McCarthy_Jonathan_QB.jpeg")
_, passes, layout = detect_passes("Pass_Charts/MIN/2025/1/images/McCarthy_Jonathan_QB.jpeg")
```

`scrape.py` calls the same JSON API the website uses (`/api/content/microsite/chart`, which needs a
`nextgenstats.nfl.com` Referer header) instead of parsing HTML. It resumes where it left off.
Weeks are `1`-`18` for the regular season and `post-19`, `post-20`, ... for the postseason.

## Output data

All coordinates are in **yards on the field**: `x_coord` from -26.67 (left sideline) to +26.67 (right sideline), 0 at
the middle of the field; `y_coord` is yards past the line of scrimmage (negative = backfield). Points slightly
outside the sidelines are real (out-of-bounds throws).

Every file starts with `game_id, season, season_type, week, team, esb_id, name, position`, followed by:

**`pass_locations.csv`** - one row per pass: `pass_type` (COMPLETE / INCOMPLETE / INTERCEPTION / TOUCHDOWN),
`located`, `x_coord`, `y_coord`. Every completion, touchdown and interception in the box score gets a row, so the
counts always tie out; on the rare pass whose ring is drawn completely underneath another one, `located` is False
and the coordinates are empty rather than guessed.

**`route_coords.csv`** - one row per point along each route (0.5 yd spacing, ordered from the start of the route):
`route_id`, `route_type` (COMPLETE = white line, INCOMPLETE = gray line), `touchdown`, `point`,
`segment` (`route` or `after_catch`), `start_ok`, `x_coord`, `y_coord`. `start_ok` is False if the route's start
could not be traced back to the line of scrimmage (it starts more than 1 yd ahead of it, about 6% of routes) - filter
these out if you need clean starts. Starts within ~1 yd of the line are within tracing noise.

**`carry_coords.csv`** - one row per point along each carry (start in the backfield -> where the runner was stopped):
`carry_id`, `gain_class` (LOSS / SHORT = 0-5 yds / LONG = 5+ yds or TD, straight from the line colour), `touchdown`,
`fumble_lost`, `handoff_ok`, `point`, `x_coord`, `y_coord`. `handoff_ok` is False if the traced line could not be
followed back to a handoff behind the line of scrimmage (about 0.4% of carries) - filter those out if you need clean starts.

**`chart_qc.csv`** - one row per chart comparing what was detected with what the API metadata says (completions,
touchdowns, interceptions, receptions, carries), with a `counts_ok` flag. **Filter on it** before trusting a chart.

## How it works

**Every chart is calibrated from its own pixels** (`next_gen_scrapy/calib.py`). NGS picks the zoom per chart so the deepest play
fits, and in one season's charts the view ranges from 54 to 110 yards deep, so there is no fixed layout to hard-code.
Each image is measured instead:

- the blue LOS bar gives the image row of `y = 0`;
- the two grey sideline bands give `x = +/-26.67 yd`, and where they meet gives the horizon;
- the 5-yard lines pin the remaining degree of freedom (the scale), matched to the grid by vote so that shadows and
  drawn lines cannot mislead it.

That yields a homography per chart. It calibrates 99.6% of charts (the rest are blank images NGS occasionally
publishes), it reproduces a hand-measured calibration to within 0.07 yd, and the recovered yard lines sit on the
5-yard grid to about 0.02 yd. Then:

- `next_gen_scrapy/passes.py` finds the coloured rings (green/white/red/blue). Ring size is predicted from the chart's own
  geometry, and overlapping rings are separated using the completion/touchdown/interception counts from the
  metadata - those counts are exact, so the job is to place that many rings rather than to re-count them.
  Touchdown arcs and the LOS bar are excluded.
- `next_gen_scrapy/paths.py` + `routes.py` / `carries.py` trace the drawn lines: skeletonise, split at junctions, and
  re-join through crossings by choosing the smoothest continuation, bridging gaps where one line is drawn over another.
  After-catch (green) segments are attached to their route, and touchdown / lost-fumble rings are attached to the line
  they end. Lines sharing a start point are separated, and the metadata counts merge stray fragments.

Counts come from the chart metadata, plus - for route charts - the **target count** from the NGS receiving statboard,
which `ngs-scrape` merges into the stored metadata. A route chart draws one line per target, so that count bounds the
incomplete (grey) routes too; the chart's own metadata only gives receptions, which bounds the completed ones. The
statboard omits low-volume players, so about 13% of route charts have no target count and fall back to receptions only.

## Accuracy and known limitations

Measured over a full season of scraped charts (1,870 charts: 2025 plus early 2026), counting a chart as consistent
only when every count it can be checked against matches the box score (`counts_ok` in `chart_qc.csv`):

| Chart | Charts fully consistent | What is checked |
|---|---|---|
| Pass  | 595 / 607 (98.0%) | completions, touchdowns, interceptions |
| Route | 597 / 642 (93.0%) | receptions, touchdown rings |
| Carry | 602 / 621 (96.9%) | carries, touchdown rings |

Of 17,668 extracted passes, 17,658 (99.94%) have coordinates; the 10 without are rings drawn underneath another ring.

- **Incomplete passes:** as in the original version, the chart doesn't draw every incompletion the metadata counts
  (throwaways, spikes, etc.), so detected incompletes are fewer than `attempts - completions - interceptions`.
  This is *not* an error and is *not* padded - only rings actually on the chart are returned. Occasionally an
  interception is missing from the chart too, even though the box score has one.
- **Passes** are the most reliable, because a ring centre is a well-defined point. The only real failure is a ring
  drawn completely underneath another one, which is reported (`located = False`) rather than guessed.
- **Routes and carries** are traced from pixels, so a *count* that matches is not proof that each individual line is
  right: where several lines overlap (many carries leaving the same handoff, or routes crossing at their catch points)
  a line can still be mis-split or mis-joined. Treat individual paths as approximate, and prefer aggregate measures.
  `start_ok` / `handoff_ok` flag the lines whose start could not be traced back to where it must begin.
- **Blank charts:** NGS sometimes publishes a chart with no field drawn (8 of 1,870 here). These are reported as
  errors in `chart_qc.csv`, not silently skipped.
- **If the chart art changes**, a chart that cannot be calibrated is reported in `chart_qc.csv` rather than silently
  mis-scaled. Because calibration is per chart, a new zoom level needs no code change.

## Original 2017-2018 pipeline

See `archive/` (Python 3.7 + R + nflscrapR). Original credits: Sarah Mallepalle and the CMU team, Sam Ventura,
Kostas Pelechrinis, Ron Yurko; route and carry extensions by Arrowhead Analytics.
