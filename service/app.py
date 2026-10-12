"""next-gen-scrapy explorer: a web app over the next_gen_scrapy package - no database.

Pick any player and season: the app pulls that player's Next Gen Stats charts live, reads them with the package's
chart readers, matches every pass / route / carry to its nflverse play-by-play row (match_to_pbp), and serves the
result to the React UI in service/web (shadcn/ui). Results are cached in memory only (service/engine.py).

Run (from a clone of the repo):
    pip install -e ".[pbp]" fastapi "uvicorn[standard]"
    cd service/web && npm install && npm run build && cd ../..
    uvicorn service.app:app --port 8000          # then open http://127.0.0.1:8000

UI development: `uvicorn service.app:app --port 8000 --reload` and, in service/web, `npm run dev` (Vite proxies
/api to :8000).
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import engine, nl
from .filters import GROUPS, for_table

Kind = Literal["pass", "route", "carry"]
WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"


def _load_env(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines) - real environment variables win. GEMINI_API_KEY lives here."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_env(Path(__file__).resolve().parent.parent / ".env")

app = FastAPI(
    title="next-gen-scrapy explorer",
    description="Next Gen Stats charts read live by the next_gen_scrapy package and matched to play-by-play.",
    version="3.0.0",
)


@app.get("/api/seasons")
def seasons():
    return {"seasons": engine.seasons(), "current": engine.current_season()}


@app.get("/api/players")
def players(season: int = Query(...)):
    """Everyone with a chart this season (from the NGS chart listing), most active first."""
    try:
        return {"season": season, "players": engine.players(season)}
    except Exception as e:  # NGS unreachable
        raise HTTPException(502, f"couldn't reach Next Gen Stats: {e}")


class LoadRequest(BaseModel):
    kind: Kind
    esb_id: str
    season: int


@app.post("/api/load")
def load(req: LoadRequest):
    """Start (or reuse) the live pipeline for one player's season of one chart kind; poll /api/jobs/{id}."""
    try:
        return engine.start(req.kind, req.esb_id, req.season).public()
    except LookupError as e:
        raise HTTPException(404, str(e))


def _job(job_id: str) -> engine.Job:
    job = engine.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown or expired job - load the player again")
    return job


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _job(job_id).public()


@app.get("/api/data/{job_id}")
def job_data(job_id: str):
    job = _job(job_id)
    if job.data is None:
        raise HTTPException(409, "not ready yet")
    return job.data


@app.get("/api/play/{job_id}/{play_id}")
def play_detail(job_id: str, play_id: str):
    """Everything about one play of a loaded dataset (the panel that opens when a play is clicked)."""
    job = _job(job_id)
    out = engine.play_detail(job, play_id)
    if out is None:
        raise HTTPException(404, "no such play in this dataset")
    return out


@app.get("/api/export/{job_id}")
def export_csv(job_id: str):
    """Every matched play with every chart, pbp, FTN and participation column."""
    job = _job(job_id)
    if job.frame is None:
        raise HTTPException(409, "not ready yet")
    buf = io.StringIO()
    frame = job.frame.assign(drawn=True)
    if job.undrawn is not None and len(job.undrawn):   # the incompletions NGS didn't draw, with estimated spots
        frame = pd.concat([frame, job.undrawn.assign(drawn=False)], ignore_index=True)
    frame.to_csv(buf, index=False)
    fname = f"{job.name.replace(' ', '_')}_{job.season}_{engine.TABLE[job.kind]}.csv"
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.get("/api/filters")
def filters(kind: Kind = "pass"):
    table = engine.TABLE[kind]
    return {"groups": GROUPS, "filters": for_table(table)}


@app.get("/api/ask")
def ask(q: str = Query(..., min_length=1, max_length=300), kind: Kind = "pass", season: int | None = None,
        job: str | None = None):
    """Plain-English question -> filters (+ player / season / chart kind when the question names them). Rules
    first; Gemini (GEMINI_API_KEY) only for words the rules couldn't place, validated against the filter list."""
    season = season or engine.current_season()
    rows = []
    for k in engine.KINDS:
        for c in engine.listing(k, season):
            rows.append((c["playerName"], c["team"], season, engine.TABLE[k], 1))
    choices: dict[str, set] = {}
    j = engine.get(job) if job else None
    if j is not None and j.frame is not None and len(j.frame):
        for f in for_table(engine.TABLE[j.kind]):
            if f["kind"] in ("choice", "select") and not f["choices"] and f["col"] in j.frame:
                choices[f["col"]] = {v for v in j.frame[f["col"]].dropna().unique().tolist() if v != ""}
    parsed = nl.parse(q, nl.Players(rows), engine.TABLE[kind], {"positions": set(), "choices": choices}).as_dict()
    if parsed.get("name"):
        hit = [c for k in engine.KINDS for c in engine.listing(k, season) if c["playerName"] == parsed["name"]]
        parsed["esb_id"] = hit[0]["esbId"] if hit else None
    return parsed


# --------------------------------------------------------------------------------------------- the UI
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str):
    if not WEB_DIST.exists():
        return HTMLResponse("<p style='font-family:system-ui;padding:2rem'>The UI isn't built yet: "
                            "<code>cd service/web && npm install && npm run build</code></p>", status_code=503)
    f = WEB_DIST / path
    if path and f.is_file():
        return FileResponse(f)
    return FileResponse(WEB_DIST / "index.html")
