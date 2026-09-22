"""
The simple, player-first API: give a player's name, a season and (optionally) which weeks, get a
tidy DataFrame back in field yards. No chart images, no file paths, no scrape/extract split -
whatever isn't already on disk is fetched automatically.

    from next_gen_scrapy import get_passes, get_routes, get_carries

    get_passes("Justin Jefferson", 2025)                     # every charted pass, every week found
    get_carries("Bijan Robinson", 2025, weeks=[1, 2, 3])      # just those three weeks
    get_routes("Ja'Marr Chase", 2025, download=False)         # only what's already been scraped

Player names must match the exact display name Next Gen Stats uses (e.g. "Ja'Marr Chase",
"Amon-Ra St. Brown"); a near-miss raises an error listing the closest matches it found.
"""
import difflib
import glob
import json
import os
import warnings

import pandas as pd

from . import extract
from . import scrape as _scrape

TYPES = extract.TYPES                 # {"pass": "Pass_Charts", "route": "Route_Charts", "carry": "Carry_Charts"}
PROCESSORS = extract.PROCESSORS       # {"pass": process_pass, "route": process_route, "carry": process_carry}


def _norm(name):
    return " ".join(name.split()).casefold()


def _week_sort_key(w):
    """Numeric order, regular season before postseason ('1' < '2' < ... < 'post-19' < 'post-20')."""
    return (1, int(w[len("post-"):])) if w.startswith("post-") else (0, int(w))


def _local_matches(kind, player, season, root, weeks):
    """
    (image_path, meta) for this player's charts already saved on disk, restricted to `weeks` when
    given (globs only those week folders - fast), plus every player name seen along the way.
    """
    out_root = os.path.join(root, TYPES[kind])
    week_parts = ["*"] if weeks is None else sorted(weeks)      # glob has no brace expansion - one pass per week
    found, all_names = {}, set()
    for week_part in week_parts:
        pattern = os.path.join(out_root, "*", str(season), week_part, "data", "*.json")
        for data_file in glob.glob(pattern):
            with open(data_file) as f:
                meta = json.load(f)
            all_names.add(meta["playerName"])
            if _norm(meta["playerName"]) != _norm(player):
                continue
            img_file = data_file[:-len(".json")].replace(os.sep + "data" + os.sep, os.sep + "images" + os.sep) + ".jpeg"
            if os.path.exists(img_file):
                found[_scrape.week_label(meta)] = (img_file, meta)
    return found, all_names


def _gather(kind, player, season, weeks, root, download, quiet):
    season = str(season)
    want_weeks = None if weeks is None else {str(w) for w in weeks}

    found, all_names = _local_matches(kind, player, season, root, want_weeks)
    have_enough = found and (want_weeks is None or want_weeks.issubset(found))

    if download and not have_enough:
        out_root = os.path.join(root, TYPES[kind])
        abbrs = _scrape.get_team_abbrs()
        matched_any = False
        if not quiet:
            print("looking up %s %s charts for %s, %s..." % (season, kind, player,
                  "weeks " + ", ".join(sorted(want_weeks)) if want_weeks else "every week"), flush=True)
        for chart in _scrape.fetch_charts(kind, season):
            if _norm(chart["playerName"]) != _norm(player):
                all_names.add(chart["playerName"])
                continue
            matched_any = True
            week = _scrape.week_label(chart)
            if want_weeks is not None and week not in want_weeks:
                continue
            if week in found:
                continue
            team = abbrs.get(chart["teamId"], chart["teamId"])
            img_file, data_file, downloaded = _scrape.save_chart(chart, team, out_root)
            if not quiet and downloaded:
                print("  downloaded", team, week, chart["lastName"], chart["firstName"], flush=True)
            with open(data_file) as f:
                found[week] = (img_file, json.load(f))
        if not matched_any and not found:
            close = difflib.get_close_matches(player, all_names, n=5, cutoff=0.5)
            hint = (" Closest names found in %s %s charts: %s." % (season, kind, ", ".join(close))
                   if close else "")
            raise ValueError("no %s charts found for %r in %s.%s" % (kind, player, season, hint))

    if want_weeks is not None:                 # local scan may have picked up other weeks too - trim to what was asked
        found = {w: v for w, v in found.items() if w in want_weeks}

    if not found:
        extra = "" if download else " (try download=True, or run ngs-scrape first)"
        raise ValueError("no locally-saved %s charts for %r in %s%s" % (kind, player, season, extra))

    missing = (want_weeks - found.keys()) if want_weeks else set()
    if missing:
        warnings.warn("no %s chart found for %r in %s, week(s): %s"
                      % (kind, player, season, ", ".join(sorted(missing))))

    return [found[w] for w in sorted(found, key=_week_sort_key)]


def _get(kind, player, season, weeks, root, download, quiet):
    charts = _gather(kind, player, season, weeks, root, download, quiet)
    process = PROCESSORS[kind]
    rows = []
    for image_path, meta in charts:
        try:
            r, _qc = process(image_path, meta)
            rows += r
        except Exception as e:
            warnings.warn("skipped %s (week %s): %r" % (image_path, _scrape.week_label(meta), e))
    return pd.DataFrame(rows)


def get_passes(player, season, weeks=None, root=".", download=True, quiet=False):
    """
    Every charted pass for one QB, one season (optionally limited to `weeks`, e.g. [1, 2, "post-19"]).
    Downloads whatever charts aren't already saved under `root` unless download=False.
    Returns a DataFrame: game_id, season, week, team, name, pass_type, located, x_coord, y_coord, ...
    (pass_type: COMPLETE / INCOMPLETE / INTERCEPTION / TOUCHDOWN; located=False -> no coordinates,
    the ring was hidden under another one; see the README for what these mean.)
    """
    return _get("pass", player, season, weeks, root, download, quiet)


def get_routes(player, season, weeks=None, root=".", download=True, quiet=False):
    """
    Every route point for one receiver, one season. One row per point along each route (0.5 yd
    spacing); group by (game_id, route_id) for one row per route. See get_passes for the other args.
    """
    return _get("route", player, season, weeks, root, download, quiet)


def get_carries(player, season, weeks=None, root=".", download=True, quiet=False):
    """
    Every carry point for one running back, one season. One row per point along each carry; group
    by (game_id, carry_id) for one row per carry. See get_passes for the other args.
    """
    return _get("carry", player, season, weeks, root, download, quiet)
