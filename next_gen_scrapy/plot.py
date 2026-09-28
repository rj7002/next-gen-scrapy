"""
Plot passes, routes and carries on a Next Gen Stats-style field - as the plays themselves, or as a
heatmap of where they happened.

    from next_gen_scrapy import get_routes, plot

    routes = get_routes("Justin Jefferson", 2024, weeks=[1, 2, 3])
    plot(routes, kind="route")                      # every route, NGS colours
    plot(routes, kind="route", heatmap=True)        # where his routes go
    plot(routes, kind="route", heatmap=True, heat_of="end")   # where he catches the ball

`data` can be anything this package produces for that kind: get_passes / get_routes /
get_carries output, the ngs-extract CSVs, to_paths() output, or match_to_pbp() output (filter it
by down, coverage, ... first, then plot).

The field is drawn top-down in the same coordinates as the data: x across the field (0 = middle,
+/-26.67 = sidelines), y = yards past the line of scrimmage.
"""
import json

import numpy as np
import pandas as pd

from .reshape import to_paths

HALF_WIDTH = 160.0 / 6.0        # 26.67 yd: the sidelines
HASH_X = 18.5 / 6.0             # 3.08 yd: NFL hash marks either side of the middle
BAND = 4.0                      # grey sideline band drawn outside each sideline

# Next Gen Stats' own chart palette
COLORS = {
    "page": "#0d0e10", "turf": "#2a2d31", "line5": "#474b50", "line10": "#5d6268", "tick": "#7d8288",
    "band": "#6b6f74", "label": "#ecece8", "los": "#2356e8",
    "complete": "#7bd03a", "incomplete": "#e9e9e6", "interception": "#e5322d", "touchdown": "#2f6bff",
    "route": "#f4f4f2", "route_incomplete": "#8e9297", "yac": "#7bd03a",
    "LONG": "#7bd03a", "SHORT": "#f2c318", "LOSS": "#e5322d", "fumble": "#e5322d",
}
_PASS_STYLE = {   # pass_type -> (ring colour, label)
    "COMPLETE": (COLORS["complete"], "complete"),
    "INCOMPLETE": (COLORS["incomplete"], "incomplete"),
    "INTERCEPTION": (COLORS["interception"], "interception"),
    "TOUCHDOWN": (COLORS["touchdown"], "touchdown"),
}
_CARRY_STYLE = {"LONG": "5+ yds", "SHORT": "0-5 yds", "LOSS": "loss"}


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('plotting needs matplotlib: pip install "next-gen-scrapy[plot]"') from e
    return plt


# --------------------------------------------------------------------------------------------------
# the field
# --------------------------------------------------------------------------------------------------
def draw_field(ax=None, ymin=-10, ymax=40, figsize=None, labels=True):
    """
    Draw an empty Next Gen Stats-style field on `ax` (a new figure if None) covering yards
    [ymin, ymax] past the line of scrimmage, and return the axes. Coordinates are field yards, so
    anything plotted afterwards with ax.plot / ax.scatter lines up with the data.
    """
    plt = _plt()
    from matplotlib.patches import Rectangle

    ymin, ymax = float(ymin), float(ymax)
    if ax is None:
        width = 2 * (HALF_WIDTH + BAND)
        w_in = 7.0 if figsize is None else figsize[0]
        h_in = w_in * (ymax - ymin) / width + 0.9 if figsize is None else figsize[1]
        fig, ax = plt.subplots(figsize=(w_in, h_in))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.55 / h_in, top=1 - 0.35 / h_in)
    ax.figure.patch.set_facecolor(COLORS["page"])      # the title / legend text is light
    ax.set_facecolor(COLORS["page"])

    x0, x1 = -HALF_WIDTH - BAND, HALF_WIDTH + BAND
    ax.add_patch(Rectangle((-HALF_WIDTH, ymin), 2 * HALF_WIDTH, ymax - ymin, color=COLORS["turf"], zorder=0, lw=0))
    for side in (-1, 1):
        ax.add_patch(Rectangle((side * HALF_WIDTH if side > 0 else x0, ymin), BAND, ymax - ymin,
                               color=COLORS["band"], zorder=0.1, lw=0))

    # yard lines: every 5, bolder every 10
    for y in range(int(np.ceil(ymin / 5)) * 5, int(ymax) + 1, 5):
        if y == 0:
            continue
        major = y % 10 == 0
        ax.plot([-HALF_WIDTH, HALF_WIDTH], [y, y], color=COLORS["line10" if major else "line5"],
                lw=1.3 if major else 0.9, zorder=0.5, solid_capstyle="butt")
        if labels and major and y > 0 and ymin + 1.5 < y < ymax - 1.5:
            for side in (-1, 1):
                ax.text(side * (HALF_WIDTH + BAND / 2), y, f"+{y}", color=COLORS["label"], ha="center",
                        va="center", fontsize=9, fontweight="bold", zorder=4)

    # 1-yard ticks: along both hash lines, and just inside/outside each sideline
    ys = [y for y in range(int(np.ceil(ymin)), int(ymax) + 1) if y % 5]
    segs = []
    for y in ys:
        for hx in (-HASH_X, HASH_X):
            segs.append([(hx - 0.35, y), (hx + 0.35, y)])
        for side in (-1, 1):
            e = side * HALF_WIDTH
            segs.append([(e - side * 0.9, y), (e, y)])
            segs.append([(e + side * 0.35, y), (e + side * 1.1, y)])
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors=COLORS["tick"], linewidths=0.8, zorder=0.6))

    # line of scrimmage, across the bands too
    if ymin < 0 < ymax:
        ax.plot([x0, x1], [0, 0], color=COLORS["los"], lw=3.2, zorder=2.5, solid_capstyle="butt")
        if labels:
            for side in (-1, 1):
                ax.text(side * (HALF_WIDTH + BAND / 2), 0.5, "LOS", color=COLORS["label"], ha="center",
                        va="bottom", fontsize=8.5, fontweight="bold", zorder=4)

    ax.set_xlim(x0, x1)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


# --------------------------------------------------------------------------------------------------
# input normalisation
# --------------------------------------------------------------------------------------------------
def _arr(v):
    if isinstance(v, str):
        v = json.loads(v)
    return np.asarray(v, dtype=object if len(v) and isinstance(v[0], str) else float)


def _flag(row, *cols):
    for c in cols:
        if c in row and pd.notna(row[c]):
            v = row[c]
            return bool(v) if not isinstance(v, str) else v.strip().lower() in ("true", "1")
    return False


def _paths(data, kind):
    """One row per route/carry with path_x/path_y arrays, whatever shape `data` came in."""
    if "path_x" not in data.columns:
        data = to_paths(data, kind)
    return data


def _catch_index(seg):
    if seg is None:
        return None
    seg = list(seg)
    pre = [i for i, s in enumerate(seg) if s != "after_catch"]
    return pre[-1] if pre else len(seg) - 1


def _is_td(row):
    if row.get("bucket") == "TOUCHDOWN" or row.get("pass_type") == "TOUCHDOWN":
        return True
    return _flag(row, "chart_touchdown", "touchdown")


def _y_range(ys, ymin, ymax):
    ys = np.asarray([y for y in ys if y is not None and np.isfinite(y)], float)
    lo = min(-10.0, np.floor((ys.min() - 3) / 5) * 5) if len(ys) else -10.0
    hi = max(30.0, np.ceil((ys.max() + 5) / 5) * 5) if len(ys) else 40.0
    return (lo if ymin is None else ymin), (hi if ymax is None else ymax)


# --------------------------------------------------------------------------------------------------
# plays
# --------------------------------------------------------------------------------------------------
def _ring(ax, x, y, color, size=110, z=5, center=None):
    ax.scatter(x, y, s=size, facecolors=center or "#15171a", edgecolors=color, linewidths=2.2, zorder=z)


def _width(n):
    """Line width / alpha for n overlapping paths: bold for a handful, thin once they pile up."""
    return float(np.clip(2.4 - 0.25 * np.log2(max(n, 1) / 10), 1.1, 2.4)), float(np.clip(1.15 - n / 1000, 0.7, 1.0))


def _line(ax, xs, ys, color, lw=2.2, alpha=1.0, z=3, dashed=False):
    import matplotlib.patheffects as pe
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=z, solid_capstyle="round",
            ls=(0, (3, 2)) if dashed else "-",
            path_effects=[pe.Stroke(linewidth=lw + 1.6, foreground="#101114", alpha=0.85 * alpha), pe.Normal()])


def _legend(ax, entries):
    from matplotlib.lines import Line2D
    handles = []
    for kind, color, label in entries:
        if kind == "ring":
            handles.append(Line2D([], [], ls="", marker="o", markersize=8, markerfacecolor="#15171a",
                                  markeredgecolor=color, markeredgewidth=2.2, label=label))
        else:
            handles.append(Line2D([], [], color=color, lw=2.6, label=label))
    if handles:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=min(len(handles), 3),
                  fontsize=8.5, frameon=False, labelcolor=COLORS["label"], handlelength=1.8, columnspacing=1.6)


def _plot_passes(ax, data, legend):
    d = data.dropna(subset=["x_coord", "y_coord"])
    order = ["INCOMPLETE", "COMPLETE", "INTERCEPTION", "TOUCHDOWN"]
    entries = []
    for pt in order:
        g = d[d["pass_type"] == pt]
        if not len(g):
            continue
        color, label = _PASS_STYLE[pt]
        center = color if pt == "TOUCHDOWN" else None
        _ring(ax, g["x_coord"], g["y_coord"], color, z=5 + order.index(pt), center=None)
        if pt == "TOUCHDOWN":
            ax.scatter(g["x_coord"], g["y_coord"], s=14, color=center, zorder=9)
        entries.append(("ring", color, f"{label} ({len(g)})"))
    if legend:
        _legend(ax, entries)


def _plot_routes(ax, data, legend):
    n = {"COMPLETE": 0, "INCOMPLETE": 0, "td": 0, "yac": False}
    rows = sorted(data.to_dict("records"), key=lambda r: r.get("route_type") == "COMPLETE")   # incompletes underneath
    lw, alpha = _width(len(rows))
    for r in rows:
        px, py = _arr(r["path_x"]), _arr(r["path_y"])
        if len(px) < 2:
            continue
        complete = r.get("route_type") == "COMPLETE"
        n["COMPLETE" if complete else "INCOMPLETE"] += 1
        seg = _arr(r["path_segment"]) if "path_segment" in r and r["path_segment"] is not None else None
        ci = _catch_index(seg) if seg is not None else len(px) - 1
        _line(ax, px[:ci + 1], py[:ci + 1], COLORS["route"] if complete else COLORS["route_incomplete"],
              lw=lw, alpha=alpha, z=4 if complete else 3)
        if ci < len(px) - 1:
            _line(ax, px[ci:], py[ci:], COLORS["yac"], lw=lw, alpha=alpha, z=4.5)
            n["yac"] = True
        if _is_td(r):
            _ring(ax, px[ci], py[ci], COLORS["touchdown"], size=130, z=8)
            ax.scatter(px[ci], py[ci], s=14, color=COLORS["touchdown"], zorder=9)
            n["td"] += 1
        elif complete:
            _ring(ax, px[ci], py[ci], COLORS["yac"], size=46, z=7, center="#f4f4f2")
    if legend:
        entries = []
        if n["COMPLETE"]:
            entries.append(("line", COLORS["route"], f"complete ({n['COMPLETE']})"))
        if n["INCOMPLETE"]:
            entries.append(("line", COLORS["route_incomplete"], f"incomplete ({n['INCOMPLETE']})"))
        if n["yac"]:
            entries.append(("line", COLORS["yac"], "yards after catch"))
        if n["td"]:
            entries.append(("ring", COLORS["touchdown"], f"touchdown ({n['td']})"))
        _legend(ax, entries)


def _plot_carries(ax, data, legend):
    n = {k: 0 for k in _CARRY_STYLE}
    n.update(td=0, fumble=0)
    order = {"LOSS": 0, "SHORT": 1, "LONG": 2}
    rows = sorted(data.to_dict("records"), key=lambda r: order.get(r.get("gain_class"), 1))
    lw, alpha = _width(len(rows))
    for r in rows:
        px, py = _arr(r["path_x"]), _arr(r["path_y"])
        if len(px) < 2:
            continue
        gc = r.get("gain_class") if r.get("gain_class") in _CARRY_STYLE else "SHORT"
        n[gc] += 1
        _line(ax, px, py, COLORS[gc], lw=lw, alpha=alpha, z=3 + order[gc] * 0.1)
        if _is_td(r):
            _ring(ax, px[-1], py[-1], COLORS["touchdown"], size=130, z=8)
            ax.scatter(px[-1], py[-1], s=14, color=COLORS["touchdown"], zorder=9)
            n["td"] += 1
        if _flag(r, "chart_fumble_lost", "fumble_lost"):
            _ring(ax, px[-1], py[-1], COLORS["fumble"], size=130, z=8)
            n["fumble"] += 1
    if legend:
        entries = [("line", COLORS[k], f"{lab} ({n[k]})") for k, lab in _CARRY_STYLE.items() if n[k]]
        if n["td"]:
            entries.append(("ring", COLORS["touchdown"], f"touchdown ({n['td']})"))
        if n["fumble"]:
            entries.append(("ring", COLORS["fumble"], f"fumble lost ({n['fumble']})"))
        _legend(ax, entries)


# --------------------------------------------------------------------------------------------------
# heatmap
# --------------------------------------------------------------------------------------------------
def _heat_points(data, kind, heat_of):
    """(xs, ys, weights): every play counts once in total - a long route is spread along its path."""
    xs, ys, ws = [], [], []
    if kind == "pass":
        d = data.dropna(subset=["x_coord", "y_coord"])
        return d["x_coord"].to_numpy(float), d["y_coord"].to_numpy(float), np.ones(len(d))
    for r in data.to_dict("records"):
        px, py = _arr(r["path_x"]), _arr(r["path_y"])
        if not len(px):
            continue
        if heat_of == "end":
            seg = _arr(r["path_segment"]) if kind == "route" and r.get("path_segment") is not None else None
            i = _catch_index(seg) if seg is not None else len(px) - 1
            xs.append(px[i]); ys.append(py[i]); ws.append(1.0)
        else:
            xs.extend(px); ys.extend(py); ws.extend([1.0 / len(px)] * len(px))
    return np.asarray(xs, float), np.asarray(ys, float), np.asarray(ws, float)


def _plot_heatmap(ax, x, y, w, ymin, ymax, smooth, cmap, colorbar):
    from scipy.ndimage import gaussian_filter

    plt = _plt()
    cell = 0.5
    x_edges = np.linspace(-HALF_WIDTH, HALF_WIDTH, int(round(2 * HALF_WIDTH / cell)) + 1)
    y_edges = np.arange(ymin, ymax + cell, cell)
    grid, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=w)
    if smooth > 0:
        grid = gaussian_filter(grid, sigma=smooth / cell, mode="constant")
    peak = grid.max()
    if peak <= 0:
        return
    v = grid / peak
    # skip the colour map's near-black end (it reads as a hole in the turf) and fade the thin
    # tails out so the field shows through where nothing happens
    rgba = plt.get_cmap(cmap)(0.22 + 0.78 * v)
    rgba[..., 3] = np.clip((v - 0.03) / 0.2, 0, 1) ** 0.8 * 0.92
    ax.imshow(rgba, extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]), origin="lower",
              interpolation="bilinear", zorder=0.4, aspect="auto")      # under the yard lines
    if colorbar:
        import matplotlib as mpl
        colors = plt.get_cmap(cmap)(np.linspace(0.22, 1, 256))
        sm = mpl.cm.ScalarMappable(cmap=mpl.colors.ListedColormap(colors), norm=mpl.colors.Normalize(0, 1))
        cax = ax.inset_axes([0.3, -0.035, 0.4, 0.014])
        cb = ax.figure.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_ticks([0, 1], labels=["fewer", "more"])
        cb.ax.tick_params(labelcolor=COLORS["label"], color=COLORS["label"], labelsize=8, length=0)
        cb.outline.set_visible(False)


# --------------------------------------------------------------------------------------------------
# public
# --------------------------------------------------------------------------------------------------
def plot(data, kind="pass", heatmap=False, ax=None, title=None, legend=True, heat_of="path", smooth=1.5,
         cmap="inferno", colorbar=True, ymin=None, ymax=None, figsize=None):
    """
    Plot passes, routes or carries on a Next Gen Stats-style field and return the matplotlib axes.

    data     : passes / routes / carries from this package - get_* output, the ngs-extract CSVs,
               to_paths() output or match_to_pbp() output (filter it first to plot a situation).
    kind     : 'pass', 'route' or 'carry'.
    heatmap  : False = draw every play in the charts' own colours (pass rings: green complete, white
               incomplete, red interception, blue touchdown; routes: white / grey to the catch, a
               catch-point dot, green after the catch; carries: green 5+ yds, yellow 0-5, red loss;
               touchdown and fumble rings). True = a smoothed density of where they happened.
    heat_of  : heatmaps of routes / carries only: 'path' (everywhere the route / carry went, each
               play counting once in total) or 'end' (the catch or target point / where the carry ended).
    smooth   : heatmap smoothing, in yards.
    cmap     : heatmap colour map (any matplotlib name).
    ax       : draw into an existing axes (the field is drawn on it) instead of a new figure.
    title, legend, colorbar, ymin, ymax, figsize : as named; the depth range fits the data if None.
    """
    if kind not in ("pass", "route", "carry"):
        raise ValueError("kind must be 'pass', 'route' or 'carry'")
    if heat_of not in ("path", "end"):
        raise ValueError("heat_of must be 'path' or 'end'")
    if data is None or not len(data):
        raise ValueError("nothing to plot: data is empty")

    if kind == "pass":
        if "x_coord" not in data.columns:
            raise ValueError("pass data needs x_coord / y_coord columns")
        ys = data["y_coord"].dropna().to_numpy(float)
    else:
        data = _paths(data, kind)
        ys = np.concatenate([_arr(p) for p in data["path_y"]]) if len(data) else np.array([])
    ymin, ymax = _y_range(ys, ymin, ymax)

    ax = draw_field(ax, ymin=ymin, ymax=ymax, figsize=figsize)
    if heatmap:
        x, y, w = _heat_points(data, kind, heat_of)
        _plot_heatmap(ax, x, y, w, ymin, ymax, smooth, cmap, colorbar)
    elif kind == "pass":
        _plot_passes(ax, data, legend)
    elif kind == "route":
        _plot_routes(ax, data, legend)
    else:
        _plot_carries(ax, data, legend)

    if title:
        ax.set_title(title, color=COLORS["label"], fontsize=12, fontweight="bold", loc="left", pad=8)
    return ax
