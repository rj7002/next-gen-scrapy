"""
next_gen_scrapy: scrape NFL Next Gen Stats pass/route/carry charts and extract field coordinates
from the chart images.

    from next_gen_scrapy import calibrate, detect_passes, detect_routes, detect_carries

See the package README for the full pipeline (scrape -> extract -> CSVs) and the `ngs-scrape` /
`ngs-extract` console scripts installed alongside it.
"""
from .calib import CalibrationError, Chart, calibrate
from .carries import detect_carries
from .passes import detect_blue_rings, detect_passes
from .routes import detect_routes

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CalibrationError",
    "Chart",
    "calibrate",
    "detect_passes",
    "detect_blue_rings",
    "detect_routes",
    "detect_carries",
]
