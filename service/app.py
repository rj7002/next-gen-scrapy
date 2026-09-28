"""Analytics API + data exporter for next_gen_scrapy's pbp-matched pass/route/carry data.

Every row here is a chart entity (pass, route, or carry) matched to its real nflverse play-by-play
+ FTN charting row (see next_gen_scrapy.pbp_match.match_to_pbp) - so on top of the chart's own
recovered (x, y) coordinates, every play-level feature (down, distance, EPA, WP, coverage,
play-action, ...) is filterable and exportable too.

Run (from a clone of the repo; the site is not part of the PyPI package):
    pip install -e ".[pbp]" fastapi "uvicorn[standard]"
    .venv/bin/python service/build_matches.py     # once per season, needs network (nflreadpy)
    .venv/bin/python service/build_db.py           # rebuild the queryable DB
    .venv/bin/uvicorn service.app:app --reload

Docs: http://127.0.0.1:8000/docs   Analytics UI: http://127.0.0.1:8000/analytics
"""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from . import nl
from .filters import GROUPS, for_table

DB_PATH = Path(__file__).resolve().parent / "ngs.db"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _load_env(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines) - real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_env(Path(__file__).resolve().parent.parent / ".env")

app = FastAPI(
    title="next-gen-scrapy analytics API",
    description=(
        "Every pass, route and carry recovered from NFL Next Gen Stats chart images, matched to "
        "its real play-by-play + FTN charting row. Filter on ANY column - chart fields (season, "
        "team, pass_type, ...) or play fields (down, epa, is_play_action, n_pass_rushers, ...) - "
        "with `?column=value`, `?column__gte=value`, `?column__lte=value`, `?column__ne=value`, "
        "or `?column__in=a,b,c` (e.g. `week__in=1,2,3`). "
        "Every endpoint returns JSON by default or CSV with `?format=csv`."
    ),
    version="2.0.0",
)

TABLES = {"passes": "passes", "routes": "routes", "carries": "carries"}
ARRAY_COLS = {"path_x", "path_y", "path_segment"}
MAX_LIMIT = 5000
RESERVED_PARAMS = {"format", "limit", "offset", "sort"}
OPS = {"gte": ">=", "lte": "<=", "gt": ">", "lt": "<", "ne": "!="}


def get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="database not built yet - run `python service/build_matches.py` then `python service/build_db.py`",
        )
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def table_columns(table: str) -> list[str]:
    con = get_conn()
    try:
        return [row[1] for row in con.execute(f"PRAGMA table_info({table})")]
    finally:
        con.close()


def _normalize(value: str) -> str:
    if value.lower() in ("true", "false"):
        return "1" if value.lower() == "true" else "0"
    return value


def build_where(params: dict[str, str], columns: set[str], reserved: set[str]) -> tuple[str, list]:
    """query params -> (' WHERE ...' or '', bind values), rejecting anything not a real column."""
    clauses, binds = [], []
    for key, value in params.items():
        if key in reserved:
            continue
        col, op = key, "="
        if "__" in key:
            base, suffix = key.rsplit("__", 1)
            if suffix in OPS or suffix == "in":
                col, op = base, OPS.get(suffix, "IN")
        if col not in columns:
            raise HTTPException(status_code=400, detail=f"'{col}' is not a column on this endpoint")
        if op == "IN":
            values = [_normalize(v.strip()) for v in value.split(",") if v.strip()]
            if not values:
                continue
            clauses.append(f'"{col}" IN ({", ".join("?" * len(values))})')
            binds.extend(values)
        else:
            clauses.append(f'"{col}" {op} ?')
            binds.append(_normalize(value))
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", binds


def query_table(table: str, request: Request, limit: int, offset: int, sort: str | None):
    columns = set(table_columns(table))
    where_sql, params = build_where(dict(request.query_params), columns, RESERVED_PARAMS)

    sql = f"SELECT * FROM {table}" + where_sql
    if sort:
        sort_col = sort.lstrip("-")
        if sort_col not in columns:
            raise HTTPException(status_code=400, detail=f"cannot sort on '{sort_col}'")
        sql += f' ORDER BY "{sort_col}" {"DESC" if sort.startswith("-") else "ASC"}'
    sql += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    con = get_conn()
    try:
        rows = [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()

    for row in rows:
        for col in ARRAY_COLS:
            if row.get(col) is not None:
                row[col] = json.loads(row[col])
    return rows


def rows_to_csv(rows: list[dict]) -> StreamingResponse:
    buf = io.StringIO()
    fieldnames = list(rows[0].keys()) if rows else []
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items()})
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=export.csv"},
    )


def respond(rows: list[dict], fmt: str):
    if fmt == "csv":
        return rows_to_csv(rows)
    return {"count": len(rows), "results": rows}


def _list_endpoint(table: str, request: Request, format: str, limit: int, offset: int, sort: str | None):
    rows = query_table(table, request, limit, offset, sort)
    return respond(rows, format)


@app.get("/passes")
def get_passes(
    request: Request,
    format: Literal["json", "csv"] = "json",
    limit: int = Query(500, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
    sort: str | None = None,
):
    """One row per pass matched to its pbp/FTN play. Filter on any column, e.g.
    `?season=2024&team=BUF&pass_type=TOUCHDOWN&epa__gte=1&is_play_action=true`."""
    return _list_endpoint("passes", request, format, limit, offset, sort)


@app.get("/routes")
def get_routes(
    request: Request,
    format: Literal["json", "csv"] = "json",
    limit: int = Query(500, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
    sort: str | None = None,
):
    """One row per route matched to its pbp/FTN play, including the full traced path
    (`path_x`/`path_y`/`path_segment` arrays). Filter on any column."""
    return _list_endpoint("routes", request, format, limit, offset, sort)


@app.get("/carries")
def get_carries(
    request: Request,
    format: Literal["json", "csv"] = "json",
    limit: int = Query(500, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
    sort: str | None = None,
):
    """One row per carry matched to its pbp/FTN play, including the full traced path
    (`path_x`/`path_y` arrays). Filter on any column."""
    return _list_endpoint("carries", request, format, limit, offset, sort)


@app.get("/{table}/fields")
def get_fields(table: Literal["passes", "routes", "carries"]):
    """Column metadata for building a filter UI: name, sqlite type, and (for low-cardinality text
    columns) the distinct values seen, so a frontend can render dropdowns instead of free text."""
    con = get_conn()
    try:
        cols = con.execute(f"PRAGMA table_info({table})").fetchall()
        fields = []
        for _, name, sqltype, *_ in cols:
            entry = {"name": name, "type": sqltype}
            if sqltype == "TEXT" and name not in ARRAY_COLS:
                n_distinct = con.execute(f'SELECT COUNT(DISTINCT "{name}") FROM {table}').fetchone()[0]
                if 0 < n_distinct <= 40:
                    values = con.execute(
                        f'SELECT DISTINCT "{name}" FROM {table} WHERE "{name}" IS NOT NULL ORDER BY 1'
                    ).fetchall()
                    entry["values"] = [v[0] for v in values]
            fields.append(entry)
        return {"table": table, "fields": fields}
    finally:
        con.close()


@app.get("/{table}/distinct")
def get_distinct(
    table: Literal["passes", "routes", "carries"],
    request: Request,
    columns: str = Query(..., description="comma-separated column names, e.g. `season_type,week`"),
):
    """Distinct combinations of `columns`, after applying any filters - e.g.
    `/passes/distinct?columns=season_type,week&season=2024` lists the weeks that season has data for."""
    all_cols = set(table_columns(table))
    wanted = [c.strip() for c in columns.split(",") if c.strip()]
    bad = [c for c in wanted if c not in all_cols]
    if not wanted or bad:
        raise HTTPException(status_code=400, detail=f"unknown columns: {bad or columns}")
    where_sql, binds = build_where(dict(request.query_params), all_cols, RESERVED_PARAMS | {"columns"})
    cols_sql = ", ".join(f'"{c}"' for c in wanted)
    con = get_conn()
    try:
        rows = con.execute(
            f"SELECT DISTINCT {cols_sql} FROM {table}{where_sql} ORDER BY {cols_sql}", binds
        ).fetchall()
    finally:
        con.close()
    return {"columns": wanted, "values": [list(r) for r in rows]}


HEAT_X = (-30.0, 30.0)     # sidelines are +/-26.67; a little room for out-of-bounds throws
HEAT_Y = (-20.0, 100.0)    # hard clip on depth


@app.get("/{table}/heatmap")
def get_heatmap(
    table: Literal["passes", "routes", "carries"],
    request: Request,
    mode: Literal["path", "end"] = Query("path", description="routes/carries: every point along the path, or just where it ended (catch point / tackle spot). Passes are always their location."),
    smooth: float = Query(1.5, ge=0, le=6, description="Gaussian smoothing, in yards"),
    cell: float = Query(1.0, ge=0.5, le=3, description="grid cell size, in yards"),
):
    """Density of where the matching plays happened, over EVERY matching row (no row limit), as a
    smoothed grid. Filter exactly like the list endpoints. Each play counts once in total - a long
    route is spread along its path rather than counting more than a short one."""
    import numpy as np
    from scipy.ndimage import gaussian_filter

    all_cols = set(table_columns(table))
    where_sql, binds = build_where(dict(request.query_params), all_cols,
                                   RESERVED_PARAMS | {"mode", "smooth", "cell"})
    if table == "passes":
        sql = f"SELECT x_coord, y_coord FROM {table}{where_sql}"
    elif table == "routes":
        sql = f"SELECT path_x, path_y, path_segment FROM {table}{where_sql}"
    else:
        sql = f"SELECT path_x, path_y FROM {table}{where_sql}"

    xs, ys, ws = [], [], []
    n_plays = 0
    con = get_conn()
    try:
        for row in con.execute(sql, binds):
            if table == "passes":
                if row[0] is None or row[1] is None:
                    continue
                xs.append(row[0]); ys.append(row[1]); ws.append(1.0)
            else:
                if not row[0]:
                    continue
                px, py = json.loads(row[0]), json.loads(row[1])
                if not px:
                    continue
                if mode == "end":
                    i = len(px) - 1
                    if table == "routes" and row[2]:
                        seg = json.loads(row[2])
                        pre = [k for k, s in enumerate(seg) if s != "after_catch"]
                        i = pre[-1] if pre else i
                    xs.append(px[i]); ys.append(py[i]); ws.append(1.0)
                else:
                    w = 1.0 / len(px)
                    xs.extend(px); ys.extend(py); ws.extend([w] * len(px))
            n_plays += 1
    finally:
        con.close()

    if not n_plays:
        return {"n_plays": 0, "grid": [], "x0": HEAT_X[0], "y0": 0, "cell": cell, "nx": 0, "ny": 0, "max": 0}

    xs, ys, ws = np.asarray(xs, float), np.asarray(ys, float), np.asarray(ws, float)
    # depth window: the bulk of the data plus room for the smoothing tail, never less than -10..+20
    pad = 3 * smooth + 2
    lo = max(HEAT_Y[0], min(-10.0, np.floor(np.percentile(ys, 0.5) - pad)))
    hi = min(HEAT_Y[1], max(20.0, np.ceil(np.percentile(ys, 99.5) + pad)))
    nx = int(round((HEAT_X[1] - HEAT_X[0]) / cell))
    ny = int(round((hi - lo) / cell))
    grid, _, _ = np.histogram2d(ys, xs, bins=[ny, nx], range=[[lo, hi], list(HEAT_X)], weights=ws)
    if smooth > 0:
        grid = gaussian_filter(grid, sigma=smooth / cell, mode="constant")
    grid = grid / n_plays / (cell * cell)        # share of plays per square yard
    peak = float(grid.max())
    return {
        "n_plays": n_plays, "mode": mode if table != "passes" else "end",
        "x0": HEAT_X[0], "y0": float(lo), "cell": cell, "nx": nx, "ny": ny, "max": peak,
        # row-major by depth (row 0 = deepest-behind-LOS row), rounded to keep the payload small
        "grid": np.round(grid / peak if peak else grid, 4).tolist(),
    }


@lru_cache(maxsize=None)
def _filter_spec(table: str) -> dict:
    """Curated filters for one table, with real choice values and numeric ranges from the data."""
    con = get_conn()
    try:
        cols = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
        out = []
        for f in for_table(table):
            col = f["col"]
            if col not in cols:
                continue
            entry = {k: v for k, v in f.items() if k != "tables"}
            if f["kind"] == "range":
                lo, hi = con.execute(f'SELECT MIN("{col}"), MAX("{col}") FROM {table}').fetchone()
                if lo is None:
                    continue
                entry["min"], entry["max"] = lo, hi
            elif f["kind"] in ("choice", "select"):
                present = [r[0] for r in con.execute(
                    f'SELECT DISTINCT "{col}" FROM {table} WHERE "{col}" IS NOT NULL ORDER BY 1')]
                # pbp stores downs/quarters as REAL (3.0) - compare as "3"
                present = [int(v) if isinstance(v, float) and v.is_integer() else v for v in present]
                present_s = {str(v) for v in present}
                if f["choices"]:
                    entry["choices"] = [c for c in f["choices"] if str(c[0]) in present_s]
                else:
                    entry["choices"] = [[v, v] for v in present if str(v).strip() not in ("", "0")]
                if not entry["choices"]:
                    continue
            out.append(entry)
        return {"table": table, "groups": GROUPS, "filters": out}
    finally:
        con.close()


@app.get("/filters")
def get_filters(dataset: Literal["passes", "routes", "carries"] = "carries"):
    """The curated filter set for a dataset (label, group, control type, choices / min-max) -
    what the explorer renders. The list endpoints still accept any column for power users."""
    return _filter_spec(dataset)


@lru_cache(maxsize=1)
def _nl_context():
    from .filters import FILTERS
    con = get_conn()
    try:
        rows, choices = [], {}
        dynamic = [f["col"] for f in FILTERS if f["kind"] in ("choice", "select") and not f["choices"] and f["col"] != "defteam"]
        for t in TABLES:
            cols = set(table_columns(t))
            rows += [(n, tm, s, t, c) for n, tm, s, c in
                     con.execute(f"SELECT name, team, season, COUNT(*) FROM {t} GROUP BY 1, 2, 3")]
            for col in dynamic:
                if col in cols:
                    choices.setdefault(col, set()).update(
                        r[0] for r in con.execute(f'SELECT DISTINCT "{col}" FROM {t}') if r[0] not in (None, ""))
        return nl.Players(rows), {"positions": choices.get("position", set()), "choices": choices}
    finally:
        con.close()


@app.get("/ask")
def ask(q: str = Query(..., min_length=1, max_length=300),
        dataset: Literal["passes", "routes", "carries"] = "carries"):
    """Natural-language question -> filters. A rule-based parser handles common football phrasing;
    only when it leaves words it couldn't place does Gemini refine the result (if GEMINI_API_KEY is
    set), and every Gemini suggestion is validated against the curated filter list.
    `dataset` is the fallback when the question doesn't name one."""
    players, dyn = _nl_context()
    return nl.parse(q, players, dataset, dyn).as_dict()


@app.get("/", response_class=HTMLResponse)
def landing_page():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/analytics", response_class=HTMLResponse)
def analytics_page():
    return (STATIC_DIR / "analytics.html").read_text()
