"""
next_gen_scrapy: scrape NFL Next Gen Stats pass/route/carry charts and extract field coordinates
from the chart images.

The simple way in - name a player, a season, optionally which weeks - downloads the matching chart
images straight into memory and hands back a tidy DataFrame in field yards. Nothing is saved to disk:

    from next_gen_scrapy import get_passes, get_routes, get_carries

    get_passes("Josh Allen", 2025)
    get_carries("Bijan Robinson", 2025, weeks=[1, 2, 3])

If you want the chart images saved to disk instead (so a repeat call doesn't re-download, or so you
can inspect them yourself), use the `ngs-scrape` / `ngs-extract` console scripts, or the lower-level,
one-chart-image-at-a-time functions on files `ngs-scrape` already fetched:

    from next_gen_scrapy import calibrate, detect_passes, detect_routes, detect_carries

See the package README for the full scrape -> extract -> CSVs pipeline.
"""
from .calib import CalibrationError, Chart, calibrate
from .carries import detect_carries
from .passes import detect_blue_rings, detect_passes
from .pbp_match import load_pbp_with_ftn, match_to_pbp
from .player import get_carries, get_passes, get_routes
from .reshape import to_paths
from .routes import detect_routes

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # player-first API
    "get_passes",
    "get_routes",
    "get_carries",
    # one row per carry/route instead of one row per point
    "to_paths",
    # matching a chart to real plays (needs nflreadpy - pip install "next-gen-scrapy[pbp]")
    "load_pbp_with_ftn",
    "match_to_pbp",
    # one-image-at-a-time API
    "CalibrationError",
    "Chart",
    "calibrate",
    "detect_passes",
    "detect_blue_rings",
    "detect_routes",
    "detect_carries",
]
