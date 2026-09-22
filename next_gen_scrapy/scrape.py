"""
Scrape Next Gen Stats pass / route / carry chart images and metadata.

The NGS site is now a Vue single-page app, so there is no chart JSON embedded in the
HTML anymore. Instead we call the same JSON API the site itself uses:

    GET https://nextgenstats.nfl.com/api/content/microsite/chart
        ?type=pass|route|carry &season=YYYY|all &week=N|all &teamId=all &esbId=all
        &count=N &offset=N

Notes:
  * The API returns an empty body unless a nextgenstats.nfl.com Referer is sent.
  * Responses are paginated: {"total": N, "charts": [...]}; we page with count/offset.

Folder format:
    ./{Pass,Route,Carry}_Charts/[team_abbr]/[season]/[week]/{images,data}/[last]_[first]_[pos].{jpeg,json}
    where [week] is "1".."18" for the regular season and "post-<n>" for the postseason.

Example:
    ngs-scrape -s 2025 -w 1 2 -t MIN
    ngs-scrape --type route -s 2024 2025
"""
import argparse
import glob
import json
import os
import sys
import time

import requests

BASE = "https://nextgenstats.nfl.com"
CHART_API = BASE + "/api/content/microsite/chart"
TEAMS_API = BASE + "/api/league/teams"
RECEIVING_API = BASE + "/api/statboard/receiving"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": BASE + "/charts/list/pass",
}
PAGE_SIZE = 100

session = requests.Session()
session.headers.update(HEADERS)


def get(url, retries=4, **kwargs):
    """GET with retries; returns the response or raises after the last attempt."""
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30, **kwargs)
            if r.status_code == 200 and r.content:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError("failed to fetch " + url)


def get_team_abbrs():
    """Map API teamId -> team abbreviation (e.g. '3000' -> 'MIN')."""
    teams = get(TEAMS_API).json()
    return {t["teamId"]: t["abbr"] for t in teams}


def fetch_charts(chart_type, season):
    """Yield every chart of one type in one season, following pagination."""
    offset = 0
    while True:
        params = dict(type=chart_type, season=season, week="all", teamId="all",
                      esbId="all", count=PAGE_SIZE, offset=offset)
        data = get(CHART_API, params=params).json()
        charts = data["charts"]
        yield from charts
        offset += len(charts)
        if not charts or offset >= data["total"]:
            return


def fetch_targets(season, season_type, week):
    """
    {esbId: targets} for one week. A route chart draws one line per target, so this gives the total
    number of routes on the chart - the metadata in the chart itself only gives receptions, which
    pins down the completed (white) routes but not the incomplete (grey) ones.
    """
    try:
        data = get(RECEIVING_API, params=dict(season=season, seasonType=season_type, week=week)).json()
    except Exception:
        return {}
    out = {}
    for s in data.get("stats", []):
        esb = (s.get("player") or {}).get("esbId")
        if esb and s.get("targets") is not None:
            out[esb] = int(s["targets"])
    return out


def backfill_targets(out_root, seasons):
    """Add a `targets` field to the stored route-chart metadata (safe to re-run)."""
    cache, n = {}, 0
    for data_file in sorted(glob.glob(os.path.join(out_root, "*", "*", "*", "data", "*.json"))):
        with open(data_file) as f:
            chart = json.load(f)
        if chart.get("targets") is not None or (seasons and str(chart["season"]) not in seasons):
            continue
        key = (chart["season"], chart["seasonType"], chart["week"])
        if key not in cache:
            cache[key] = fetch_targets(*key)
        t = cache[key].get(chart["esbId"])
        if t is None:
            continue
        chart["targets"] = t
        with open(data_file, "w") as f:
            json.dump(chart, f)
        n += 1
    print("  added target counts to %d route charts" % n)


def week_label(chart):
    return str(chart["week"]) if chart["seasonType"] == "REG" else "post-" + str(chart["week"])


def main():
    parser = argparse.ArgumentParser(description="Download charts from NFL Next Gen Stats")
    parser.add_argument("--type", choices=["pass", "route", "carry"], default="pass")
    parser.add_argument("-s", "--seasons", nargs="+", default=[str(time.gmtime().tm_year)])
    parser.add_argument("-t", "--teams", nargs="+", default=None,
                        help="team abbreviations, e.g. MIN KC (default: all)")
    parser.add_argument("-w", "--weeks", nargs="+", default=None,
                        help="week labels, e.g. 1 2 post-19 (default: all)")
    parser.add_argument("--size", choices=["small", "medium", "large", "extraLarge"],
                        default="extraLarge", help="image size (extraLarge = 1200x1200)")
    parser.add_argument("--delay", type=float, default=0.2, help="seconds between image downloads")
    args = parser.parse_args()

    abbrs = get_team_abbrs()
    want_teams = {t.upper() for t in args.teams} if args.teams else None
    want_weeks = set(args.weeks) if args.weeks else None
    out_root = args.type.capitalize() + "_Charts"

    n_new = n_skipped = 0
    for season in args.seasons:
        print("Season", season, "...", flush=True)
        for chart in fetch_charts(args.type, season):
            team = abbrs.get(chart["teamId"], chart["teamId"])
            week = week_label(chart)
            if (want_teams and team not in want_teams) or (want_weeks and week not in want_weeks):
                continue

            name = "_".join([chart["lastName"], chart["firstName"], chart["position"]]).replace(os.sep, "-")
            folder = os.path.join(out_root, team, str(chart["season"]), week)
            img_file = os.path.join(folder, "images", name + ".jpeg")
            data_file = os.path.join(folder, "data", name + ".json")

            if os.path.exists(img_file) and os.path.exists(data_file):
                n_skipped += 1
                continue

            os.makedirs(os.path.dirname(img_file), exist_ok=True)
            os.makedirs(os.path.dirname(data_file), exist_ok=True)

            img = get("https:" + chart[args.size + "Img"])
            with open(img_file, "wb") as f:
                f.write(img.content)
            chart["team"] = team
            with open(data_file, "w") as f:
                json.dump(chart, f)

            n_new += 1
            print("  ", team, week, name, flush=True)
            time.sleep(args.delay)

    if args.type == "route":
        print("Fetching target counts (they bound how many routes a chart draws)...", flush=True)
        backfill_targets(out_root, set(args.seasons) if args.seasons else None)

    print("Done. %d downloaded, %d already present." % (n_new, n_skipped))


if __name__ == "__main__":
    sys.exit(main())
