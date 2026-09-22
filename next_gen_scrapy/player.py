"""
The simple, player-first API: give a player's name, a season and (optionally) which weeks, get a
tidy DataFrame back in field yards. No chart images, no file paths, no scrape/extract split, and
nothing is ever written to disk - every call downloads the matching chart images straight into
memory and decodes them there.

    from next_gen_scrapy import get_passes, get_routes, get_carries

    get_passes("Josh Allen", 2025)                            # every charted pass, every week found
    get_carries("Bijan Robinson", 2025, weeks=[1, 2, 3])       # just those three weeks

Player names must match the exact display name Next Gen Stats uses (e.g. "Ja'Marr Chase",
"Amon-Ra St. Brown"); a near-miss raises an error listing the closest matches it found.

If you'd rather have the chart images saved to disk (so a repeat call doesn't re-download, or so
you can inspect the images yourself), use `ngs-scrape` + `ngs-extract`, or the one-image-at-a-time
`detect_passes` / `detect_routes` / `detect_carries` on files `ngs-scrape` already fetched.
"""
import difflib
import warnings

import pandas as pd

from . import extract
from . import scrape as _scrape

PROCESSORS = extract.PROCESSORS       # {"pass": process_pass, "route": process_route, "carry": process_carry}


def _norm(name):
    return " ".join(name.split()).casefold()


def _week_sort_key(w):
    """Numeric order, regular season before postseason ('1' < '2' < ... < 'post-19' < 'post-20')."""
    return (1, int(w[len("post-"):])) if w.startswith("post-") else (0, int(w))


def _gather(kind, player, season, weeks, quiet):
    """Page the chart API for this player's charts and download each match's image into memory."""
    season = str(season)
    want_weeks = None if weeks is None else {str(w) for w in weeks}

    if not quiet:
        print("looking up %s %s charts for %s, %s..." % (season, kind, player,
              "weeks " + ", ".join(sorted(want_weeks)) if want_weeks else "every week"), flush=True)

    abbrs = _scrape.get_team_abbrs()          # {teamId -> abbreviation, e.g. "0610" -> "BUF"}
    found, all_names, matched_any = {}, set(), False
    for chart in _scrape.fetch_charts(kind, season):
        if _norm(chart["playerName"]) != _norm(player):
            all_names.add(chart["playerName"])
            continue
        matched_any = True
        week = _scrape.week_label(chart)
        if want_weeks is not None and week not in want_weeks:
            continue
        if not quiet:
            print("  downloading", week, chart["playerName"], flush=True)
        chart = dict(chart, team=abbrs.get(chart["teamId"], chart["teamId"]))
        found[week] = (_scrape.fetch_chart_image(chart), chart)

    if not matched_any:
        close = difflib.get_close_matches(player, all_names, n=5, cutoff=0.5)
        hint = (" Closest names found in %s %s charts: %s." % (season, kind, ", ".join(close))
               if close else "")
        raise ValueError("no %s charts found for %r in %s.%s" % (kind, player, season, hint))

    missing = (want_weeks - found.keys()) if want_weeks else set()
    if missing:
        warnings.warn("no %s chart found for %r in %s, week(s): %s"
                      % (kind, player, season, ", ".join(sorted(missing))))
    if not found:
        raise ValueError("no %s charts found for %r in %s for the requested week(s)." % (kind, player, season))

    return [found[w] for w in sorted(found, key=_week_sort_key)]


def _get(kind, player, season, weeks, quiet):
    charts = _gather(kind, player, season, weeks, quiet)
    process = PROCESSORS[kind]
    rows = []
    for image, meta in charts:
        try:
            r, _qc = process(image, meta)
            rows += r
        except Exception as e:
            warnings.warn("skipped %s week %s: %r" % (meta["playerName"], _scrape.week_label(meta), e))
    return pd.DataFrame(rows)


def get_passes(player, season, weeks=None, quiet=False):
    """
    Every charted pass for one QB, one season (optionally limited to `weeks`, e.g. [1, 2, "post-19"]).
    Downloads each matching chart's image straight into memory - nothing is saved to disk.
    Returns a DataFrame: game_id, season, week, team, name, pass_type, located, x_coord, y_coord, ...
    (pass_type: COMPLETE / INCOMPLETE / INTERCEPTION / TOUCHDOWN; located=False -> no coordinates,
    the ring was hidden under another one; see the README for what these mean.)
    """
    return _get("pass", player, season, weeks, quiet)


def get_routes(player, season, weeks=None, quiet=False):
    """
    Every route point for one receiver, one season. One row per point along each route (0.5 yd
    spacing); group by (game_id, route_id) for one row per route. See get_passes for the other args.
    """
    return _get("route", player, season, weeks, quiet)


def get_carries(player, season, weeks=None, quiet=False):
    """
    Every carry point for one running back, one season. One row per point along each carry; group
    by (game_id, carry_id) for one row per carry. See get_passes for the other args.
    """
    return _get("carry", player, season, weeks, quiet)
