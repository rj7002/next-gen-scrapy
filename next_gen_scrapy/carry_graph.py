"""
Global carry-chart reader (detect_carries(method="layers" / "graph"); method="best" picks between readers).

Same idea as route_graph: every drawn carry line (red = tackled for loss, yellow = 0-5 yards, green = 5+ yards
or TD) goes into one stroke graph, with a node for every TD ring and every lost-fumble ring, and all carries are
chosen together by an integer program:

  * every visible stroke is covered by at least one carry, every loose line end is a handoff (a start behind the
    line of scrimmage) or the end of exactly one carry - except small glimpses of a line under the bundle and
    short spurs, which may stay unexplained at a modest cost,
  * #carries = the box score's carries, #ending on a TD ring = touchdowns, each fumble ring ends one carry,
  * a carry is one colour from handoff to tackle, and its colour must match where it ends: red at or behind the
    line of scrimmage, yellow 0-5 yards past it, green 5+ yards (or on a TD ring),
  * least total turning, fewest bridges, and costs for implausible shapes: shallow handoffs, running behind
    one's own handoff, "carries" too short to be one, losses deeper than ~2.5 yards.

mode="layers" (the default reader) skeletonises each colour on its own and uses NGS's drawing order - red, then
yellow, then green on top: a red line can only be hidden under yellow or green, a yellow one only under green,
so pieces of one line are joined across exactly such stretches, by smooth curves. mode="union" uses one skeleton
of all lines with soft colour costs. Sharp V tips behind the LOS (two carries taken at one spot) become nodes.

The bundles at the handoff - 15-25 lines leaving one spot - are where a local tracer guesses; the end-depth rule
and the counts let the integer program pull them apart.
"""
import heapq
import warnings

import numpy as np
import cv2
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds

from . import calib as K
from . import route_graph as RG

LOSS, SHORT, LONG = 3, 4, 5                 # colour ids (distinct from route_graph's so its catch logic never fires)
ANY = 6                                     # a stroke whose colour can't be seen (all of it inside crossings)
NAMES = {LOSS: "LOSS", SHORT: "SHORT", LONG: "LONG"}
COLORS = {v: k for k, v in NAMES.items()}
COLOR_SLACK = 400.0                          # integer-program penalty per carry off a given per-colour count

START_MAX_Y, START_MIN_Y = 1.0, -15.0       # a carry starts at the handoff, behind the line of scrimmage
DEPTH_MARGIN = 0.8                           # yd of slack on the colour / end-depth rule (calibration noise)
BACKWARD = 1.5                               # per yd a carry's end lies deeper than its start (rare: runner retreats)
START_FREE_Y = -2.5                          # handoffs are taken 2.5+ yd behind the LOS (90%; sneaks aside):
SHALLOW_PER_YD = 8.0                         # ...shallower starts cost this per yd (a loss line's end is no handoff)
RETREAT_FREE, RETREAT_PER_YD = 1.0, 4.0      # a carry rarely runs deeper than its handoff (1.7%): per yd beyond
MIN_LEN_YD, SHORT_PER_YD = 3.0, 12.0         # 99% of carries run 2+ yd, 95% 3.5+: a shorter "carry" is a fragment
PROGRESS_MIN, PROGRESS_PER_YD = 0.5, 10.0    # 95% of carries end 0.5+ yd upfield of their handoff
DEEP_END_Y, DEEP_PER_YD = -2.5, 6.0          # losses: each yard past -2 is about half as likely as the one before
NODE_START = 6.0                             # a handoff where several lines leave one spot (V tip / crossing)
STEM_Y = -1.5                                # lines overlap in the backfield (same handoff spot): cheaper colour
CORNER_DEG, CORNER_ARM = 100.0, 9            # ...mismatch there; a V tip turns this sharply within this many px
FRAG_LEN, FRAG_COST = 60.0, 15.0            # px: a piece of line this short on its own; per end left unexplained
STUB_WIDTHS, STUB_COST = 3.5, 10.0           # spurs up to this many line widths long: leaving one out costs this
CAND_MAX_COST = 220.0
HIDDEN_END_MAX_COST = 60.0
CAND_PER_PAIR = 2
MIN_CARRY_PX = 12.0                          # a carry runs from the backfield to at least the LOS area: never a speck
# expected["yards"] (the back's pbp runs, touchdowns aside): a carry whose end is far from every one of them costs
# YARDS_HINT_COST per yd past YARDS_HINT_FREE, up to YARDS_HINT_CAP - where pieces of lines meet under the LOS bar
# or a bundle, the pairing whose ends match real runs wins (a line that "ends" at the LOS when no run did is a
# piece whose continuation went to another line)
YARDS_HINT_OFFSET = 0.4                      # a line ends ~0.4 yd past pbp's spot
YARDS_HINT_FREE, YARDS_HINT_COST, YARDS_HINT_CAP = 1.2, 6.0, 30.0
# expected["fit"] (callable): how well each candidate fits the back's best-fitting pbp run on every clue the matcher
# uses (yards, NGS distance, start spot, duration, ...), in nats - a candidate worse than the chart's typical one costs
# FIT_COST per nat beyond, up to FIT_CAP
FIT_COST, FIT_CAP = 10.0, 80.0


def masks(hsv, lay):
    """Per-colour line masks over the field card. Thin far-away runs are dim and less saturated, so the
    thresholds are looser than the legend colours; nothing else on a carry chart is red / yellow / green."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    shape = hsv.shape[:2]
    area = np.zeros(shape, bool)                # field rows incl. the sideline bands (runs go out of bounds there,
    area[:K.FIELD_ROWS] = True                  # TDs are scored there; the '+10' and 'LOS' labels are white) -
    area &= masks_area(lay, shape)              # minus the logo, whose 'NEXT GEN' is green
    red = ((h <= 8) | (h >= 170)) & (s > 110) & (v > 70)
    yellow = (h >= 14) & (h <= 34) & (s > 110) & (v > 100)
    green = (h >= 36) & (h <= 80) & (s > 60) & (v > 60)
    return {LOSS: red & area, SHORT: yellow & area, LONG: green & area}


LOGO = (78, 205)                             # rows, cols of the NFL shield + 'NEXT GEN STATS' (both art sizes)


def masks_area(lay, shape):
    a = np.ones(shape, bool)
    a[0:LOGO[0], 0:LOGO[1]] = False
    return a


def line_pixels(im, hsv, lay, td_rings, fumble_rings):
    """(col, union, hide, layers) in route_graph's format, for carry colours."""
    shape = im.shape[:2]
    m = masks(hsv, lay)
    ring_mask = np.zeros(shape, np.uint8)
    for x, y, w, _ in td_rings:
        rr = int(max(6, 0.45 * w))
        cv2.ellipse(ring_mask, (int(round(x)), int(round(y))), (rr, max(3, rr // 2)), 0, 0, 360, 1, -1)
    for x, y in fumble_rings:
        cv2.circle(ring_mask, (int(x), int(y)), 14, 1, -1)
    ring_mask = ring_mask.astype(bool)
    td_label = RG._R().td_label_mask(shape, [(r[0], r[1]) for r in td_rings])
    for c in m:
        m[c] &= ~ring_mask & ~td_label
    col = np.full(shape, -1, np.int8)
    for c in (LONG, SHORT, LOSS):
        col[m[c]] = c
    union = col >= 0
    union = cv2.morphologyEx(union.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    closed = cv2.morphologyEx(union.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    fill = closed & (col < 0)
    if fill.any():
        from scipy import ndimage as ndi
        _, (iy, ix) = ndi.distance_transform_edt(col < 0, return_indices=True)
        col[fill] = col[iy[fill], ix[fill]]
    union = closed
    rows = np.zeros(shape, bool)
    rows[:K.FIELD_ROWS] = True
    los = cv2.dilate(lay.los_bar_mask(shape, pad=2).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    rows[0:78, 0:245] = False
    hide = union | ring_mask | td_label | los | (rows & ~masks_area(lay, shape))
    layers = {c: _drop_specks(union & (col == c), SPECK_AREA) for c in (LOSS, SHORT, LONG)}
    return col, union, hide, layers


SPECK_AREA = 45                              # px: smaller blobs of one colour are anti-aliasing where two other
                                             # colours meet (red + green edges blend to yellow-green), not lines


def _drop_specks(m, area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= area
    return keep[lab]


def depth_ok(color, end_y, on_td):
    if color == LOSS:
        return end_y <= DEPTH_MARGIN
    if color == SHORT:
        return -DEPTH_MARGIN <= end_y <= 5.0 + DEPTH_MARGIN
    return on_td or end_y >= 5.0 - DEPTH_MARGIN


# ----------------------------------------------------------------------------------------------- paths
def _edge_step(G, e, color):
    E = G.edges[e]
    c, L = E["color"], E["length"]
    if c == ANY:
        return E["extra"] + 0.02 * L
    if c != color and G.hard:
        return np.inf                                 # one skeleton per colour: a carry stays on its own colour
    if E["bridge"]:
        return E["extra"] + (0.0 if c == color else RG.MISMATCH_LONG * L)
    if c == color:
        return 0.0
    crossing = L <= RG.CROSS_LEN * G.width
    return (RG.MISMATCH if (crossing or E["near_start"]) else RG.MISMATCH_LONG) * L


def _relax(G, heap, best, prev, color, penalty):
    while heap:
        c, (e, n) = heapq.heappop(heap)
        if c > best.get((e, n), np.inf):
            continue
        kind = G.nodes[n]["kind"]
        if kind in ("ring", "fumble"):
            continue
        d_in = None if G.edges[e].get("jump") else G.arrive(e, n)
        for f in G.live_edges(n):
            if f == e:
                continue
            tc = 0.0
            if d_in is not None and not G.edges[f].get("jump"):
                turn = RG._angle(d_in, G.tangent(f, n))
                if turn > RG.MAX_TURN:
                    continue
                tc = RG.turn_cost(turn)
            state = (f, G.other(f, n))
            c2 = c + tc + _edge_step(G, f, color) + penalty.get(f, 0.0)
            if c2 < best.get(state, np.inf) and c2 < np.inf:
                best[state], prev[state] = c2, (e, n)
                heapq.heappush(heap, (c2, state))
    return best, prev


def _shortest(G, s, s_cost, color, penalty):
    best, prev, heap = {}, {}, []
    for e in G.live_edges(s):
        state = (e, G.other(e, s))
        c = s_cost + _edge_step(G, e, color) + penalty.get(e, 0.0)
        if c < best.get(state, np.inf) and c < np.inf:
            best[state], prev[state] = c, None
            heapq.heappush(heap, (c, state))
    return _relax(G, heap, best, prev, color, penalty)


def _rebuild(prev, state, s):
    edges, nodes = [], []
    while state is not None:
        e, n = state
        edges.append(e)
        nodes.append(n)
        state = prev[state]
    return [s] + nodes[::-1], edges[::-1]


def _end_cost(G, m):
    kind = G.nodes[m]["kind"]
    if kind in ("ring", "fumble") or G.loose(m):
        return 0.0
    if kind in ("junction", "split", "color", "corner"):
        return RG.HIDDEN_END
    return None


def candidate_paths(G, lay, alts=3):
    """Cheapest few carries of each colour from every possible handoff to every possible end whose depth fits
    that colour: tuples (color, nodes, edges, cost, end kind)."""
    fy = lambda n: float(lay.img_to_field(G.nodes[n]["pos"][None])[0, 1])
    starts = []
    for n, N in enumerate(G.nodes):
        if N["kind"] in ("ring", "fumble") or not G.live_edges(n):
            continue
        y = fy(n)
        if START_MIN_Y <= y <= START_MAX_Y:
            shallow = SHALLOW_PER_YD * max(0.0, y - START_FREE_Y)
            if G.loose(n):
                starts.append((n, shallow, y))
            elif N["kind"] in ("junction", "color", "corner"):
                starts.append((n, NODE_START + shallow, y))
    G.starts = [(n, c) for n, c, _ in starts]
    G.start_y = {n: y for n, _, y in starts}
    G.fwd = {}
    ends_y = {}
    out, seen, per_pair = [], set(), {}
    for s, s_cost, sy in starts:
        colours = (LOSS, SHORT, LONG) if not G.hard else \
            sorted({G.edges[e]["color"] for e in G.real_edges(s)} & {LOSS, SHORT, LONG})
        for color in colours:
            penalty = {}
            for k in range(alts):
                best, prev = _shortest(G, s, s_cost, color, penalty)
                if k == 0:
                    G.fwd[(s, color)] = (best, prev)
                by_end = {}
                for (e, n), c in best.items():
                    kind = G.nodes[n]["kind"]
                    if n == s or (G.edges[e]["bridge"] and kind not in ("ring", "fumble")):
                        continue
                    ec = _end_cost(G, n)
                    if ec is None or (ec > 0 and k > 0):
                        continue
                    if n not in ends_y:
                        ends_y[n] = fy(n)
                    if not depth_ok(color, ends_y[n], kind == "ring"):
                        continue
                    if kind == "ring" and color != LONG:
                        continue                      # a touchdown is a 5+ yard (green) carry
                    tot = c + ec + BACKWARD * max(0.0, sy - ends_y[n])
                    if tot < by_end.get(n, (np.inf,))[0]:
                        by_end[n] = (tot, (e, n))
                snap, used = dict(penalty), []
                for n, (tot, state) in by_end.items():
                    nodes, edges = _rebuild(prev, state, s)
                    key = (color, tuple(edges))
                    if key in seen or sum(G.edges[e]["length"] for e in edges) < MIN_CARRY_PX:
                        continue
                    seen.add(key)
                    cost = tot - sum(snap.get(e, 0.0) for e in edges) + RG.PATH_BASE
                    if cost > CAND_MAX_COST or (_end_cost(G, n) > 0 and cost > HIDDEN_END_MAX_COST):
                        continue
                    if per_pair.get((s, n, color), 0) >= CAND_PER_PAIR:
                        continue
                    per_pair[(s, n, color)] = per_pair.get((s, n, color), 0) + 1
                    out.append((color, nodes, edges, cost, G.nodes[n]["kind"]))
                    used += edges
                for e in used:
                    penalty[e] = penalty.get(e, 0.0) + 15.0
                if not by_end:
                    break
    G.ends_y = ends_y
    return out


def _retreat(G, lay, cands):
    """Add the shape costs the path search can't see: yards the path runs behind its own start (beyond
    RETREAT_FREE), and being too short to be a whole carry."""
    out = []
    for color, nodes, edges, cost, kind in cands:
        px = RG._path_pixels(G, nodes, edges)
        f = lay.img_to_field(np.vstack([px[::4], px[-1:]]) if len(px) > 8 else px)
        y = f[:, 1]
        length = float(np.linalg.norm(np.diff(f, axis=0), axis=1).sum())
        extra = RETREAT_PER_YD * max(0.0, float(y[0] - y.min()) - RETREAT_FREE)
        extra += SHORT_PER_YD * max(0.0, MIN_LEN_YD - length)
        extra += PROGRESS_PER_YD * max(0.0, PROGRESS_MIN - float(y[-1] - y[0]))
        extra += DEEP_PER_YD * max(0.0, DEEP_END_Y - float(y[-1]))
        out.append((color, nodes, edges, cost + extra, kind))
    return out


def through_paths(G, lay, targets):
    """For each stroke in `targets`, the cheapest valid carry (handoff -> stroke -> end) of each colour."""
    out = []
    fy = lambda n: G.ends_y[n] if n in G.ends_y else float(lay.img_to_field(G.nodes[n]["pos"][None])[0, 1])
    for color in (LOSS, SHORT, LONG):
        fwd = {}
        for s, _ in G.starts:
            best, prev = G.fwd.get((s, color), ({}, {}))
            for st, c in best.items():
                if c < fwd.get(st, (np.inf,))[0]:
                    fwd[st] = (c, s, prev)
        for e in targets:
            for n in (G.edges[e]["a"], G.edges[e]["b"]):
                st = (e, n)
                if st not in fwd:
                    continue
                c0, s, prev0 = fwd[st]
                best, prev = _relax(G, [(c0, st)], {st: c0}, {st: None}, color, {})
                pick = None
                for (f, m), c in best.items():
                    kind = G.nodes[m]["kind"]
                    if m == s or (G.edges[f]["bridge"] and kind not in ("ring", "fumble")):
                        continue
                    ec = _end_cost(G, m)
                    if ec is None or not depth_ok(color, fy(m), kind == "ring") or (kind == "ring" and color != LONG):
                        continue
                    tot = c + ec + BACKWARD * max(0.0, G.start_y.get(s, 0.0) - fy(m))
                    if pick is None or tot < pick[0]:
                        pick = (tot, (f, m))
                if pick is None:
                    continue
                tail_nodes, tail_edges = _rebuild(prev, pick[1], None)
                head_nodes, head_edges = _rebuild(prev0, st, s)
                nodes = head_nodes + tail_nodes[2:]
                if sum(G.edges[f]["length"] for f in head_edges + tail_edges[1:]) < MIN_CARRY_PX:
                    continue
                out.append((color, nodes, head_edges + tail_edges[1:], pick[0] + RG.PATH_BASE, G.nodes[nodes[-1]]["kind"]))
    return out


# ----------------------------------------------------------------------------------------------- choose
def _yards_hint(cands, G, yards):
    """cands with each non-touchdown carry's cost raised by how far its end is from every pbp run's yards."""
    y = np.asarray([v for v in yards if v is not None and np.isfinite(v)], float) + YARDS_HINT_OFFSET
    if not len(y):
        return cands
    out = []
    for color, nodes, edges, cost, kind in cands:
        ey = G.ends_y.get(nodes[-1])
        if kind != "ring" and ey is not None:
            miss = float(np.min(np.abs(y - ey)))
            cost = cost + min(YARDS_HINT_CAP, YARDS_HINT_COST * max(0.0, miss - YARDS_HINT_FREE))
        out.append((color, nodes, edges, cost, kind))
    return out


def _fit_hint(cands, G, lay, fit):
    """cands with each carry's cost raised by how badly its traced line fits the back's runs (expected["fit"])."""
    if not cands:
        return cands
    lines = []
    for color, nodes, edges, cost, kind in cands:
        px = RG._path_pixels(G, nodes, edges)
        pts = RG._R().resample_field(lay, px, ceiling=None)[0] if len(px) >= 2 else []
        lines.append(dict(color=NAMES[color], td=kind == "ring", fumble=kind == "fumble", pts=pts,
                          handoff_ok=bool(len(pts) and pts[0][1] <= 0.5), gap=False))
    ok = [i for i, l in enumerate(lines) if len(l["pts"]) >= 2]
    try:
        nats = fit([lines[i] for i in ok])
    except Exception:
        return cands
    vals = np.full(len(cands), np.nan)
    for k, i in enumerate(ok):
        vals[i] = nats.get(k + 1, np.nan)
    if not np.isfinite(vals).any():
        return cands
    typical = float(np.nanmedian(vals))
    out = []
    for (color, nodes, edges, cost, kind), v in zip(cands, vals):
        if np.isfinite(v):
            cost = cost + min(FIT_CAP, FIT_COST * max(0.0, v - typical))
        out.append((color, nodes, edges, cost, kind))
    return out


def choose(G, cands, n_carries, td, min_cover, by_color=None):
    """by_color (optional): {colour id: carries of that colour} - e.g. from play-by-play (red: lost yards, yellow:
    0-5, green: 6+ or a touchdown) - held softly: missing it costs COLOR_SLACK per carry."""
    P = len(cands)
    if P == 0:
        return [], {}
    end_nodes = [n for n in range(len(G.nodes)) if G.loose(n)]
    rings = [n for n in range(len(G.nodes)) if G.nodes[n]["kind"] == "ring"]
    fumbles = [n for n in range(len(G.nodes)) if G.nodes[n]["kind"] == "fumble"]
    V = len(end_nodes)
    cover = [e for e, E in enumerate(G.edges) if not E["dead"] and not E["bridge"] and E["length"] >= min_cover]
    C, Fn = len(cover), len(fumbles)
    by_color = {k: v for k, v in (by_color or {}).items() if v is not None}
    K = len(by_color)
    # x_p (P) | start flags (V) | count slacks (4) | unexplained ends (V) | uncovered (C) | unexplained fumble rings (F)
    # | colour count slacks (2 per colour given)
    o_y, o_cs, o_ue, o_uc, o_uf = P, P + V, P + V + 4, P + V + 4 + V, P + V + 4 + V + C
    o_cc = o_uf + Fn
    nvar = o_cc + 2 * K
    c = np.zeros(nvar)
    c[:P] = [p[3] for p in cands]
    c[o_cs:o_cs + 4] = RG.SLACK
    c[o_ue:o_ue + V] = RG.SLACK / 4
    c[o_uc:o_uc + C] = RG.SLACK / 4
    # a small piece of line on its own is a glimpse of a line running under others (red under the handoff
    # bundle): it may stay unexplained for a modest cost instead of forcing a carry of its own
    parent = list(range(len(G.nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for E in G.edges:
        if not E["dead"] and not E["bridge"]:
            parent[find(E["a"])] = find(E["b"])
    comp_len = {}
    for E in G.edges:
        if not E["dead"] and not E["bridge"]:
            comp_len[find(E["a"])] = comp_len.get(find(E["a"]), 0.0) + E["length"]
    for i, n in enumerate(end_nodes):
        L = comp_len.get(find(n), np.inf)
        if L <= FRAG_LEN:
            c[o_ue + i] = FRAG_COST + 0.3 * L
    for k, e in enumerate(cover):
        L = comp_len.get(find(G.edges[e]["a"]), np.inf)
        if L <= FRAG_LEN:
            c[o_uc + k] = 0.3 * G.edges[e]["length"]
    # a short spur hanging off a line (skeleton noise at a fork, a shadow corner) may stay unexplained cheaply
    stub_len = STUB_WIDTHS * G.width
    for i, n in enumerate(end_nodes):
        e = G.real_edges(n)[0]
        E = G.edges[e]
        if E["length"] <= stub_len and G.real_edges(G.other(e, n)) and not G.loose(G.other(e, n)):
            c[o_ue + i] = STUB_COST + E["length"]
            if e in cover:
                c[o_uc + cover.index(e)] = STUB_COST + E["length"]
    c[o_uf:o_cc] = RG.SLACK / 4
    c[o_cc:] = COLOR_SLACK
    rows, cols, vals, lo, hi = [], [], [], [], []

    def add(entries, l, h):
        r = len(lo)
        for j, v in entries:
            rows.append(r)
            cols.append(j)
            vals.append(v)
        lo.append(l)
        hi.append(h)
    by_edge, ends_at, starts_at, through = {}, {}, {}, {}
    for j, p in enumerate(cands):
        for e in set(p[2]):
            by_edge.setdefault(e, []).append(j)
        ends_at.setdefault(p[1][-1], []).append(j)
        starts_at.setdefault(p[1][0], []).append(j)
        for n in set(p[1][1:-1]):
            through.setdefault(n, []).append(j)
    for k, e in enumerate(cover):
        add([(j, 1) for j in by_edge.get(e, [])] + [(o_uc + k, 1)], 1, np.inf)
    for i, n in enumerate(end_nodes):
        ends = ends_at.get(n, [])
        add([(j, 1) for j in ends], 0, 1)
        add([(j, 1) for j in ends] + [(j, 1) for j in through.get(n, []) if j not in ends]
            + [(o_y + i, 1), (o_ue + i, 1)], 1, np.inf)
        add([(j, 1) for j in starts_at.get(n, [])] + [(o_y + i, -1)], 0, np.inf)
    for r in rings:
        js = ends_at.get(r, [])
        if js:
            add([(j, 1) for j in js], 0, G.ring_cap.get(r, 1))
    for i, r in enumerate(fumbles):                    # each lost-fumble ring ends exactly one carry
        add([(j, 1) for j in ends_at.get(r, [])] + [(o_uf + i, 1)], 1, 1)
    if n_carries is not None:
        add([(j, 1) for j in range(P)] + [(o_cs, -1), (o_cs + 1, 1)], n_carries, n_carries)
    if td is not None and rings:
        add([(j, 1) for j, p in enumerate(cands) if p[4] == "ring"] + [(o_cs + 2, -1), (o_cs + 3, 1)], td, td)
    for k, (colour, n) in enumerate(by_color.items()):
        add([(j, 1) for j, p in enumerate(cands) if p[0] == colour] + [(o_cc + 2 * k, -1), (o_cc + 2 * k + 1, 1)], n, n)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(lo), nvar))
    integ = np.zeros(nvar)
    integ[:o_cs] = 1
    ub = np.full(nvar, np.inf)
    ub[:o_cs] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(np.zeros(nvar), ub),
                   options={"time_limit": RG.MILP_SECONDS, "mip_rel_gap": 0.01, "threads": 1})
    if res.x is None:
        return [], dict(status=res.status)
    x = res.x
    dbg = dict(uncovered=[cover[k] for k in range(C) if x[o_uc + k] > 0.5],
               unexplained=[end_nodes[i] for i in range(V) if x[o_ue + i] > 0.5],
               count_slack=np.round(x[o_cs:o_cs + 4], 2).tolist(), color_slack=float(x[o_cc:].sum()), obj=float(res.fun))
    return [cands[j] for j in range(P) if x[j] > 0.5], dbg


# ----------------------------------------------------------------------------------------------- read
ABOVE = {LOSS: (SHORT, LONG), SHORT: (LONG,), LONG: ()}   # NGS draws red, then yellow, then green on top
HID_MAX = {LOSS: 140.0, SHORT: 90.0, LONG: 40.0}          # px: longest same-colour gap bridged under lines above it
HID_TURN = 135.0                             # deg: most a hidden stretch may bend in all
HID_FRAC = 0.75                              # ...and this share of it must be covered by what is drawn above


JUMP_COST = 80.0                             # a TD carry whose line NGS didn't draw into its ring (see _add_jumps)


def _add_jumps(G, lay, rings):
    """
    Now and then NGS draws a TD ring with no carry line reaching it. For each such ring, let any green line end
    jump - in a straight line, for JUMP_COST - to the far end of whatever piece is drawn into the ring (or to the
    ring itself). Used only when a TD has to be explained; returns the number of jumps added.
    """
    parent = list(range(len(G.nodes)))

    def f(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for E in G.edges:
        if not E["dead"]:
            parent[f(E["a"])] = f(E["b"])
    loose = [n for n in range(len(G.nodes)) if G.loose(n)]
    added = 0
    for r in rings:
        attached = [n for n in loose if f(n) == f(r) and G.edges[G.real_edges(n)[0]]["color"] == LONG]
        t = max(attached, key=lambda n: float(np.hypot(*(G.nodes[n]["pos"] - G.nodes[r]["pos"])))) if attached else r
        for a in loose:
            if f(a) == f(r) or G.edges[G.real_edges(a)[0]]["color"] != LONG:
                continue
            d = float(np.hypot(*(G.nodes[t]["pos"] - G.nodes[a]["pos"])))
            G.add_edge(a, t, np.vstack([G.nodes[a]["pos"], G.nodes[t]["pos"]]), LONG, bridge=True, jump=True,
                       extra=JUMP_COST + 0.02 * d)
            added += 1
    G._tan.clear()
    return added


def _jump_to_ring(G, cands):
    """A jump exists only to reach a TD ring: drop candidates that take one and end anywhere else."""
    return [p for p in cands if p[4] == "ring" or not any(G.edges[e].get("jump") for e in p[2])]


def _hermite(pa, ta, pb, tb, n=24):
    """Smooth curve leaving pa along ta and arriving at pb against tb (tb: outward direction at b)."""
    d = float(np.hypot(*(pb - pa)))
    t = np.linspace(0, 1, n)[:, None]
    h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return h00 * pa + h10 * (ta * d) + h01 * pb + h11 * (-tb * d)


def _turning(c):
    v = np.diff(c, axis=0)
    ang = np.arctan2(v[:, 1], v[:, 0])
    dt = np.diff(ang)
    dt = (dt + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(np.abs(dt).sum()))


def _end_dir(G, n, want=16.0, skip=3):
    """Outward direction at loose end n, measured along up to `want` px of its line - through short stubs and
    pass-through (two-edge) nodes, which a single edge's tangent can't see past."""
    pts, cur, e = [G.nodes[n]["pos"]], n, G.real_edges(n)[0]
    while True:
        p = G.oriented(e, cur)
        pts.extend(p[1:])
        cur = G.other(e, cur)
        L = float(np.linalg.norm(np.diff(np.asarray(pts), axis=0), axis=1).sum())
        nxt = [f for f in G.real_edges(cur) if f != e]
        if L >= want or len(nxt) != 1 or cur == n:
            break
        e = nxt[0]
    P = np.asarray(pts)
    seg = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    i0 = int(np.searchsorted(seg, min(skip, seg[-1] / 3)))
    i1 = int(np.searchsorted(seg, min(want, seg[-1]))) if seg[-1] > 0 else 0
    v = P[min(i0, len(P) - 1)] - P[min(i1, len(P) - 1)]
    nv = float(np.hypot(v[0], v[1]))
    return v / nv if nv > 1e-9 else np.zeros(2)


def _hidden_bridges(G, layers, lay, static):
    """Same-colour loose end -> loose end along a smooth curve that runs under lines drawn above it (or under
    the LOS bar / a ring / a label: `static`) - a line keeps its course, bending at most HID_TURN, while hidden."""
    over = {c: cv2.dilate((np.logical_or.reduce([layers[k] for k in ABOVE[c]] + [static])).astype(np.uint8),
                          np.ones((5, 5), np.uint8)).astype(bool) for c in layers}
    h, w = static.shape
    ends = [n for n in range(len(G.nodes)) if G.loose(n)]
    info = []
    for n in ends:
        e = G.real_edges(n)[0]
        info.append((n, G.edges[e]["color"], _end_dir(G, n), G.nodes[n]["pos"]))
    for i in range(len(info)):
        a, ca, ta, pa = info[i]
        if ca not in over:
            continue
        for j in range(i + 1, len(info)):
            b, cb, tb, pb = info[j]
            if ca != cb:
                continue
            gap = pb - pa
            d = float(np.hypot(gap[0], gap[1]))
            if d < 4.0 or d > HID_MAX[ca]:
                continue
            u = gap / d
            if float(np.dot(ta, u)) < 0.0 or float(np.dot(tb, -u)) < 0.0:
                continue                                  # both ends must at least face each other
            if any(G.other(f, a) == b for f in G.nodes[a]["edges"]):
                continue
            c = _hermite(pa, ta, pb, tb)
            turn = _turning(c)
            if turn > HID_TURN:
                continue
            q = np.round(c[2:-2]).astype(int)
            q[:, 0] = np.clip(q[:, 0], 0, w - 1)
            q[:, 1] = np.clip(q[:, 1], 0, h - 1)
            if over[ca][q[:, 1], q[:, 0]].mean() < HID_FRAC:
                continue
            arc = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
            G.add_edge(a, b, c, ca, bridge=True, curved=True, extra=RG.BRIDGE_COST * arc + 0.15 * turn)
    G._tan.clear()


def _corners(G, lay):
    """Split every stroke at a sharp V tip behind the LOS: two carries taken at one spot leave it in different
    directions, and the skeleton draws them as one stroke bending back on itself there."""
    for e in range(len(G.edges)):
        E = G.edges[e]
        if E["dead"] or E["bridge"] or len(E["pts"]) < 2 * CORNER_ARM + 3:
            continue
        p = E["pts"]
        a = CORNER_ARM
        u = p[a:-a] - p[:-2 * a]
        v = p[2 * a:] - p[a:-a]
        nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
        cos = (u * v).sum(1) / np.maximum(nu * nv, 1e-9)
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        i = int(np.argmax(ang))
        if ang[i] < CORNER_DEG:
            continue
        k = i + a
        if lay.img_to_field(p[k][None])[0, 1] > START_MAX_Y:
            continue
        m = G.split_edge(e, k)
        G.nodes[m]["kind"] = "corner"


def _recolour(G, col):
    """Strokes that lie wholly inside crossings come out of build_graph with no carry colour: give each the
    majority colour of its own pixels, or ANY when none is visible (bridges take their end's colour or ANY)."""
    h, w = col.shape
    for E in G.edges:
        if E["color"] in (LOSS, SHORT, LONG):
            continue
        if not E["bridge"]:
            p = np.round(E["pts"]).astype(int)
            v = col[np.clip(p[:, 1], 0, h - 1), np.clip(p[:, 0], 0, w - 1)]
            v = v[v >= LOSS]
            E["color"] = int(np.bincount(v).argmax()) if len(v) else ANY
        else:
            E["color"] = ANY


def read_carries(image, expected, mode="union", lay=None):
    """
    expected: carries, touchdowns, max_yards. Returns (carries, info): carries in detect_carries() format -
    color ('LOSS' / 'SHORT' / 'LONG'), td, fumble, handoff_ok, pts (field yd, handoff -> end) - plus px.
    """
    from .carries import detect_red_rings, carry_masks
    im = K.read_image(image)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    lay = lay or K.calibrate(im)
    td_rings = RG.find_rings(hsv, lay, im.shape[:2])
    fumble_rings = detect_red_rings(carry_masks(hsv, lay, [(r[0], r[1]) for r in td_rings])["LOSS"], hsv)
    col, union, hide, layers = line_pixels(im, hsv, lay, td_rings, fumble_rings)
    width = RG.stroke_width(union)
    if mode == "layers":
        G = RG.Graph()
        for colour, m in layers.items():
            if m.any():
                RG.build_graph(None, m, G=G, width=width, layer=colour)
        G.width = width
    else:
        G = RG.build_graph(col, union, width=width)
    _corners(G, lay)
    _corners(G, lay)                                  # a stroke can hold two tips
    if mode == "layers":
        # a gap in one colour's skeleton is hidden only where something else is drawn over it - another colour's
        # line, a ring, the LOS bar, a label; finish_graph's generic bridges get the non-line hiders, and
        # gaps under other lines are bridged separately with a strict alignment test (bundles are dense)
        G = RG.finish_graph(G, lay, td_rings, hide & ~union, attach=True)
        _hidden_bridges(G, layers, lay, hide & ~union)
    else:
        G = RG.finish_graph(G, lay, td_rings, hide, attach=True)
    for x, y in fumble_rings:                          # lost-fumble rings: a carry ends on each
        r = G.add_node((x, y), "fumble")
        for n, N in enumerate(G.nodes):
            if n != r and N["kind"] not in ("ring", "fumble", "color") and G.live_edges(n):
                d = float(np.hypot(N["pos"][0] - x, N["pos"][1] - y))
                if d <= 30:
                    e0 = G.real_edges(n)
                    G.add_edge(n, r, np.vstack([N["pos"], [x, y]]), G.edges[e0[0]]["color"] if e0 else LOSS,
                               bridge=True, extra=RG.BRIDGE_COST * d)
    _recolour(G, col)
    G.hard = mode == "layers"
    if G.hard:                                        # bridges onto another colour's stroke lead nowhere
        for E in G.edges:
            if E["bridge"] and not E["dead"]:
                for n in (E["a"], E["b"]):
                    N = G.nodes[n]
                    if N["kind"] in ("ring", "fumble"):
                        continue
                    if not any(G.edges[f]["color"] == E["color"] for f in G.real_edges(n)):
                        E["dead"] = True
    for E in G.edges:
        if not E["bridge"] and not E["dead"]:
            E["near_start"] = bool(lay.img_to_field(E["pts"])[:, 1].max() <= STEM_Y)
    G._tan.clear()
    cands = _retreat(G, lay, candidate_paths(G, lay))
    if expected.get("touchdowns"):
        reached = {p[1][-1] for p in cands}
        orphans = [n for n, N in enumerate(G.nodes) if N["kind"] == "ring" and n not in reached]
        if orphans and _add_jumps(G, lay, orphans):
            cands = _jump_to_ring(G, _retreat(G, lay, candidate_paths(G, lay)))
    n_carries, td = expected.get("carries"), expected.get("touchdowns")
    by_color = {COLORS[k]: v for k, v in (expected.get("by_color") or {}).items()}
    if expected.get("yards") is not None:
        cands = _yards_hint(cands, G, expected["yards"])
    if expected.get("fit") is not None:
        cands = _fit_hint(cands, G, lay, expected["fit"])
    min_cover = 2.5 * G.width
    chosen, dbg = choose(G, cands, n_carries, td, min_cover, by_color)
    seen = {(p[0], tuple(p[2])) for p in cands}
    for _ in range(2):
        todo = dbg.get("uncovered", [])
        if not todo:
            break
        new = _jump_to_ring(G, _retreat(G, lay, [p for p in through_paths(G, lay, todo) if (p[0], tuple(p[2])) not in seen]))
        if expected.get("yards") is not None:
            new = _yards_hint(new, G, expected["yards"])
        if expected.get("fit") is not None:
            new = _fit_hint(new, G, lay, expected["fit"])
        if not new:
            break
        seen |= {(p[0], tuple(p[2])) for p in new}
        cands += new
        chosen, dbg = choose(G, cands, n_carries, td, min_cover, by_color)
    out = []
    for color, nodes, edges, cost, end_kind in chosen:
        px = RG._path_pixels(G, nodes, edges)
        if len(px) < 2:
            continue
        pts, truncated = RG._R().resample_field(lay, px, ceiling=None)
        if len(pts) < 2:
            continue
        out.append(dict(color=NAMES[color], td=end_kind == "ring", fumble=end_kind == "fumble", pts=pts, px=px,
                        handoff_ok=bool(pts[0][1] <= 0.5), truncated=bool(truncated),
                        gap=any(G.edges[e].get("jump") for e in edges)))
    out.sort(key=lambda c: (c["pts"][0][0], c["pts"][0][1]))
    return out, dict(graph=G, cands=len(cands), lay=lay, dbg=dbg, mode=mode, n_td_rings=len(td_rings),
                     n_fumble_rings=len(fumble_rings))
