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

import hashlib
import hmac
import io
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
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
    docs_url=None, redoc_url=None, openapi_url=None,     # don't publish a map of the API
)

# --------------------------------------------------------------------------------------------- API guard
# With NGS_API_GUARD=1 (the deployed app), /api answers only the app's own page running in a browser: requests must
# come from the same site (browsers send Sec-Fetch-Site; scripts and other sites don't, or send something else) and
# carry the session cookie the page sets, and each visitor is rate-limited. A determined user could still replay a
# browser session - nothing the browser can request is truly hidden - but casual and scripted use is refused.
API_GUARD = os.environ.get("NGS_API_GUARD", "0") == "1"
SESSION_SECRET = (os.environ.get("NGS_SESSION_SECRET") or secrets.token_hex(32)).encode()
SESSION_COOKIE, SESSION_TTL = "ngs_s", 12 * 3600
RATE = {"/api/load": (30, 600), "/api/ask": (60, 600), "": (600, 60)}   # path prefix: (requests, per seconds)
_hits: dict[tuple[str, str], list[float]] = {}
_hits_lock = threading.Lock()


def _sign(ts: int) -> str:
    return hmac.new(SESSION_SECRET, str(ts).encode(), hashlib.sha256).hexdigest()[:32]


def _session_ok(token: str | None) -> bool:
    try:
        ts, sig = token.split(".", 1)
        return time.time() - int(ts) < SESSION_TTL and hmac.compare_digest(sig, _sign(int(ts)))
    except (AttributeError, ValueError):
        return False


def _set_session(request: Request, response: Response) -> None:
    ts = int(time.time())
    https = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    # over https, also usable when the page is embedded in another site's frame (Hugging Face shows a Space inside
    # huggingface.co): SameSite=None needs Secure, and Partitioned keeps the cookie to that one embedding
    if https:
        response.set_cookie(SESSION_COOKIE, f"{ts}.{_sign(ts)}", max_age=SESSION_TTL, httponly=True,
                            samesite="none", secure=True, partitioned=True)
    else:
        response.set_cookie(SESSION_COOKIE, f"{ts}.{_sign(ts)}", max_age=SESSION_TTL, httponly=True, samesite="lax")


def _rate_ok(request: Request) -> bool:
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    path = request.url.path
    key = next(k for k in RATE if path.startswith(k))
    limit, window = RATE[key]
    now = time.time()
    with _hits_lock:
        hits = [t for t in _hits.get((ip, key), []) if now - t < window]
        if len(hits) >= limit:
            _hits[(ip, key)] = hits
            return False
        hits.append(now)
        _hits[(ip, key)] = hits
    return True


@app.middleware("http")
async def api_guard(request: Request, call_next):
    if not API_GUARD:
        return await call_next(request)
    path = request.url.path
    if path.startswith("/api/"):
        if request.headers.get("sec-fetch-site") != "same-origin":
            return JSONResponse({"detail": "Not available."}, status_code=403)
        if not _session_ok(request.cookies.get(SESSION_COOKIE)):
            return JSONResponse({"detail": "Session expired - reload the page."}, status_code=401)
        if not _rate_ok(request):
            return JSONResponse({"detail": "Too many requests - slow down a little."}, status_code=429)
        return await call_next(request)
    response = await call_next(request)
    if not path.startswith("/assets/") and not _session_ok(request.cookies.get(SESSION_COOKIE)):
        _set_session(request, response)            # the page itself (and /session): hand out a session
    return response


@app.get("/session", include_in_schema=False)
def session():
    """The page calls this to renew an expired session (the guard middleware sets the cookie)."""
    return Response(status_code=204)


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
