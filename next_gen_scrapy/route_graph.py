"""
Global route-chart reader (detect_routes(method="graph" / "layers"); method="best" picks between readers).

Every drawn line (white route, grey incomplete route, green after-catch run) is thinned into a stroke graph.
Two ways of building it:

  * mode="union": one skeleton of all lines together. Nodes are line ends, crossings and colour changes
    (white -> green along a stroke is a catch point).
  * mode="layers": one skeleton per colour. Where strokes of different colours touch or run alongside each
    other (an after-catch run coming back along its own route, a grey route sharing a white route's stem)
    the combined skeleton is a knot; per colour each line stays whole, and the layers are linked by bridges.

In both, short gaps where a line is hidden (under a TD ring, the LOS bar, the TD label or another line, at the
white/green catch seam) are offered as optional bridges, a loose end may attach to a node or to the middle of
another line it stops short of, and every TD ring is a node of its own. A route is a path through the graph,
and all routes are chosen together by an integer program:

  * every visible stroke is covered by at least one chosen route,
  * every loose line end is the start of routes (near the line of scrimmage), the end of exactly one route,
    or a point a route passes through over a bridge,
  * #complete routes = receptions, #incomplete = targets - receptions, #ending on a TD ring = touchdowns,
    and no ring takes more touchdowns than it can hold,
  * completed routes run white then green (one catch point), incomplete routes run grey,
  * least total turning at crossings (routes are smooth curves), least colour mismatch, fewest bridges.
"""
import heapq
import math
import warnings

import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize, disk

from . import calib as K
from . import paths as NP

WHITE, GRAY, GREEN = 0, 1, 2
WHITE_V = 240                             # stroke-centre brightness: white routes >= this, grey routes below

# where routes start (field yd past the LOS)
START_MAX_Y, START_MIN_Y = 3.0, -12.0
START_FREE_Y = 1.5                        # starts beyond this cost START_PER_YD per extra yard
START_PER_YD = 6.0
STEM_Y = 6.0                              # strokes entirely within this of the LOS may be shared stems

# path costs
TURN_A, TURN_B = 0.02, 0.0016             # turn cost at a crossing: TURN_A * deg + TURN_B * deg^2
MAX_TURN = 160.0                          # deg: sharper turns at a crossing are not continuations
CATCH_TURN = 0.02                         # per deg turned at a catch point (receivers turn upfield sharply)
MISMATCH = 0.15                           # per px on another colour at a crossing / shared stem (overlap)
MISMATCH_LONG = 1.0                       # per px on another colour anywhere else
CROSS_LEN = 8.0                           # x line width: longest stretch of another colour a crossing can show
HIDDEN_END, HIDDEN_START = 12.0, 35.0     # a route ending / starting at a crossing (tip hidden)
GREEN_START = 40.0                        # completed route whose white part is too short to see (screens)
GREEN_START_MAX_Y = 1.0                   # ...which only happens for catches at / behind the LOS
PATH_BASE = 1.0

# bridges across gaps
BRIDGE_MAX, BRIDGE_RING_MAX, BRIDGE_CATCH_MAX = 26.0, 48.0, 18.0
BRIDGE_CATCH_HIDDEN = 40.0                # px: catch seam hidden under the LOS bar / another line
BRIDGE_HIDDEN_MAX = 110.0                 # px: longer gaps are bridged when something drawn on top hides them
HIDDEN_FRAC = 0.6                         # ...i.e. when at least this share of the gap is covered
BRIDGE_COST = 0.2                         # per px bridged, plus a straightness term
ATTACH_MAX = 20.0                         # px: a loose end may attach to the middle of a line this close ahead
RING_REACH = 30.0                         # px: a line end this close to a ring centre may end on the ring
RING_REACH_HIDDEN = 110.0                 # ...or this far when it points at the ring over hidden pixels

# candidate pruning (the integer program slows down sharply with thousands of hopeless candidates)
CAND_MAX_COST = 220.0                     # dearer than leaving a line unexplained: never chosen
HIDDEN_END_MAX_COST = 60.0                # routes ending at a crossing / split (tip hidden) are kept only if cheap
CAND_PER_PAIR = 2                         # alternatives kept per (start, end, route type)

MIN_COMPONENT = 30.0                      # px: isolated fragments shorter than this (text, specks) are dropped
SLACK = 1000.0                            # integer-program penalty for breaking a count / coverage rule
JUMP_COST = 80.0                          # a TD route whose line NGS didn't draw into its ring (see _add_jumps)
MILP_SECONDS = 4.0                        # per solve; on the densest charts the best solution found by then is kept


def turn_cost(deg):
    return TURN_A * deg + TURN_B * deg * deg


def _angle(u, v):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(u[0] * v[0] + u[1] * v[1])))))


# ----------------------------------------------------------------------------------------------- pixels
def find_rings(hsv, lay, shape):
    """TD rings are the only blue marks on the field besides the LOS bar: (x, y, width, area) of every blue blob."""
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    area = np.zeros(shape, bool)                     # the whole field card - a TD can be scored at the sideline,
    area[:K.FIELD_ROWS] = True                       # or deep enough to land beside the logo
    area[0:70, 0:70] = False                         # ...but the NFL shield is blue too
    blue = (h >= 98) & (h <= 128) & (sat > 120) & (val > 70) & area
    blue &= ~cv2.dilate(lay.los_bar_mask(shape).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    n, lab, st, cen = cv2.connectedComponentsWithStats(blue.astype(np.uint8), 8)
    blobs = [(float(cen[i][0]), float(cen[i][1]), float(st[i, cv2.CC_STAT_WIDTH]), float(st[i, cv2.CC_STAT_AREA]))
             for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 6]
    rings = []                                       # a ring split by a line crossing it is one ring
    for b in sorted(blobs):
        if rings and np.hypot(b[0] - rings[-1][0], b[1] - rings[-1][1]) < 16:
            a = rings[-1]
            rings[-1] = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, max(a[2], b[2]), a[3] + b[3])
        else:
            rings.append(b)
    return rings


def _R():
    from . import routes
    return routes


def line_colors(im, hsv, lay, rings):
    """
    Line pixels and their colours. Returns (col, union, hide, layers):
      col    per-pixel colour id (-1 off the lines); grey vs white by stroke-centre brightness, since grey routes
             range from dark (2018) to nearly white (some 2025 charts)
      union  every line pixel
      hide   pixels that hide whatever is drawn under them (other lines, rings, the LOS bar, the TD label)
      layers {colour: mask} - each colour on its own, for mode="layers"
    """
    shape = im.shape[:2]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    td_label = _R().td_label_mask(shape, [(r[0], r[1]) for r in rings])
    allowed = lay.on_card_field(shape) & ~td_label
    # The overlay mask blanks a ~105 px band outside each sideline (for the '+10', '+20' ... labels), which
    # cuts every route that runs out of bounds. Keep band pixels that belong to a stroke reaching the field.
    rows = np.zeros(shape, bool)
    rows[:K.FIELD_ROWS] = True
    band = rows & ~allowed & ~td_label
    band[0:78, 0:245] = False                              # logo
    r0, r1 = int(lay.B) - 30, int(lay.B) + 26              # 'LOS' corner labels
    band[max(0, r0):r1, 0:110] = False
    band[max(0, r0):r1, shape[1] - 110:] = False
    ring_mask = np.zeros(shape, np.uint8)
    for x, y, w, _ in rings:                               # the ring hides whatever passes under it
        rr = int(max(6, 0.45 * w))
        cv2.ellipse(ring_mask, (int(round(x)), int(round(y))), (rr, max(3, rr // 2)), 0, 0, 360, 1, -1)
    ring_mask = ring_mask.astype(bool)
    allowed &= ~ring_mask
    area = allowed | band
    neutral = (s < 16) & (v > 95) & area
    neutral = cv2.morphologyEx(neutral.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    # green: nothing else on the chart is green, so thin far-away runs (dim, low saturation) can be kept whole
    green = (h >= 30) & (h <= 85) & (s > 45) & (v > 60) & area
    _, lab = cv2.connectedComponents((neutral | green).astype(np.uint8), connectivity=8)
    inside = np.unique(lab[(neutral | green) & allowed])
    keep = np.isin(lab, inside[inside > 0])
    neutral &= keep | allowed
    green &= keep | allowed
    allowed = allowed | (band & keep)
    union = neutral | green
    union = cv2.morphologyEx(union.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool) & allowed
    vmax = cv2.dilate(v, np.ones((3, 3), np.uint8))
    is_white = vmax >= WHITE_V
    green_near = cv2.dilate(green.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    col = np.full(shape, -1, np.int8)
    col[union] = GRAY
    col[union & is_white] = WHITE
    col[union & ((green_near & ~is_white) | green)] = GREEN
    los = cv2.dilate(lay.los_bar_mask(shape, pad=2).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    hide = union | ring_mask | (rows & ~allowed) | los      # masked overlay areas hide lines too
    # per-colour layers: the anti-aliased fringe of a white / green stroke is not a grey line
    lw = union & (col == WHITE)
    lg = union & (col == GREEN)
    fringe = cv2.dilate((lw | lg).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    lgr = union & (col == GRAY) & ~fringe
    lgr = cv2.morphologyEx(lgr.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    layers = {WHITE: lw, GRAY: lgr, GREEN: lg}
    return col, union, hide, layers


# ----------------------------------------------------------------------------------------------- graph
class Graph:
    def __init__(self):
        self.nodes, self.edges = [], []
        self._tan = {}
        self.width = 5.0

    def add_node(self, pos, kind):
        self.nodes.append(dict(pos=np.asarray(pos, float), kind=kind, edges=[]))
        return len(self.nodes) - 1

    def add_edge(self, a, b, pts, color, **kw):
        pts = np.asarray(pts, float)
        L = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0
        E = dict(a=a, b=b, pts=pts, color=int(color), length=L, bridge=False, dead=False, near_start=False,
                 catch_bridge=False, extra=0.0)
        E.update(kw)
        self.edges.append(E)
        e = len(self.edges) - 1
        self.nodes[a]["edges"].append(e)
        if b != a:
            self.nodes[b]["edges"].append(e)
        return e

    def split_edge(self, e, i):
        """Cut edge e at its point i into two edges meeting at a new 'split' node; returns that node."""
        E = self.edges[e]
        m = self.add_node(E["pts"][i], "split")
        old_b, tail = E["b"], E["pts"][i:]
        E["pts"] = E["pts"][:i + 1]
        E["length"] = float(np.linalg.norm(np.diff(E["pts"], axis=0), axis=1).sum())
        self.nodes[old_b]["edges"].remove(e)
        E["b"] = m
        self.nodes[m]["edges"].append(e)
        kw = {k: E[k] for k in ("bridge", "dead", "near_start", "catch_bridge", "extra")}
        self.add_edge(m, old_b, tail, E["color"], **kw)
        self._tan.clear()
        return m

    def other(self, e, n):
        E = self.edges[e]
        return E["b"] if E["a"] == n else E["a"]

    def oriented(self, e, n):
        E = self.edges[e]
        return E["pts"] if E["a"] == n else E["pts"][::-1]

    def tangent(self, e, n, skip=6, k=14):
        """Unit vector pointing from node n into edge e, measured clear of the crossing (cached)."""
        key = (e, n, skip, k)
        t = self._tan.get(key)
        if t is not None:
            return t
        p = self.oriented(e, n)
        if len(p) < 2:
            t = np.zeros(2)
        else:
            if self.edges[e]["bridge"] and not self.edges[e].get("curved"):
                v = p[-1] - p[0]
            else:
                i0 = min(skip, max(0, len(p) - 3))
                i1 = min(i0 + k, len(p) - 1)
                v = p[i1] - p[i0]
                if v[0] * v[0] + v[1] * v[1] < 1e-12:
                    v = p[-1] - p[0]
            nv = math.hypot(v[0], v[1])
            t = v / nv if nv else v
        self._tan[key] = t
        return t

    def arrive(self, e, n, k=20):
        """Unit travel direction when arriving at node n along edge e."""
        return -self.tangent(e, n, k=k)

    def real_edges(self, n):
        return [e for e in self.nodes[n]["edges"] if not self.edges[e]["dead"] and not self.edges[e]["bridge"]]

    def live_edges(self, n):
        return [e for e in self.nodes[n]["edges"] if not self.edges[e]["dead"]]

    def loose(self, n):
        return self.nodes[n]["kind"] == "end" and len(self.real_edges(n)) == 1


def _runs(colors, min_run):
    """Colour runs along a chain; unknown (-1) pixels and runs shorter than min_run join their neighbours."""
    c = np.asarray(colors).copy()
    known = np.flatnonzero(c >= 0)
    if not len(known):
        return [(0, len(c), WHITE)]
    for i in range(len(c)):
        if c[i] < 0:
            c[i] = c[known[np.argmin(np.abs(known - i))]]
    runs, s0 = [], 0
    for i in range(1, len(c) + 1):
        if i == len(c) or c[i] != c[s0]:
            runs.append([s0, i, int(c[s0])])
            s0 = i
    while len(runs) > 1:
        k = min(range(len(runs)), key=lambda j: runs[j][1] - runs[j][0])
        if runs[k][1] - runs[k][0] >= min_run:
            break
        if k == 0:
            runs[1][0] = runs[0][0]
        elif k == len(runs) - 1:
            runs[-2][1] = runs[-1][1]
        else:
            left, right = runs[k - 1], runs[k + 1]
            if (left[1] - left[0]) >= (right[1] - right[0]):
                left[1] = runs[k][1]
            else:
                right[0] = runs[k][0]
        runs.pop(k)
        merged = [runs[0]]
        for r in runs[1:]:
            if r[2] == merged[-1][2]:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        runs = merged
    return [tuple(r) for r in runs]


def _pixel_graph(sk):
    """8-connected skeleton adjacency without redundant diagonal links (a diagonal step is only a link when
    neither of the two pixels it skips over is on the skeleton). Returns {pixel: [neighbours]}."""
    pix = set(zip(*np.nonzero(sk)))
    adj = {}
    for (y, x) in pix:
        nb = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (y + dy, x + dx) in pix:
                nb.append((y + dy, x + dx))
        for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            q = (y + dy, x + dx)
            if q in pix and (y + dy, x) not in pix and (y, x + dx) not in pix:
                nb.append(q)
        adj[(y, x)] = nb
    return adj


def stroke_width(union):
    dist = ndi.distance_transform_edt(union)
    sk = skeletonize(union)
    return max(3.0, 2 * float(np.median(dist[sk]))) if sk.any() else 5.0


def build_graph(col, union, G=None, width=None, layer=None):
    """Add the skeleton of `union` to graph G (a new Graph if None). `col` gives the colour per pixel; with
    `layer` set every stroke takes that colour (one colour per skeleton, so no colour-change nodes)."""
    sk = skeletonize(union)
    sk = NP._prune_spurs(sk, 8)
    sk = skeletonize(sk)
    if G is None:
        G = Graph()
    if width is None:
        dist = ndi.distance_transform_edt(union)
        width = max(3.0, 2 * float(np.median(dist[sk]))) if sk.any() else 5.0
    G.width = width
    adj = _pixel_graph(sk)
    junc = np.zeros(sk.shape, bool)
    for p, nb in adj.items():
        if len(nb) >= 3:
            junc[p] = True
    jz = cv2.dilate(junc.astype(np.uint8), disk(max(2, int(round(width * 0.9)))).astype(np.uint8)).astype(bool) & sk   # one node per crossing
    jl, nj = ndi.label(jz, structure=np.ones((3, 3)))
    near_junc = (ndi.distance_transform_edt(~junc) <= 2.0 * width) if (junc.any() and layer is None) \
        else np.zeros(sk.shape, bool)
    jnode = {}
    for j, sl in enumerate(ndi.find_objects(jl), start=1):
        ys, xs = np.nonzero(jl[sl] == j)
        jnode[j] = G.add_node((xs.mean() + sl[1].start, ys.mean() + sl[0].start), "junction")
    in_j = lambda p: jl[p] > 0
    endnode = {}
    visited = set()

    def node_at(p):
        if in_j(p):
            return jnode[int(jl[p])]
        if p not in endnode:
            endnode[p] = G.add_node((p[1], p[0]), "end")
        return endnode[p]

    def add_chain(order, a, b):
        pts = np.array([(x, y) for y, x in order], float)
        if layer is not None:
            colors = [layer] * len(order)
        else:
            colors = [int(col[y, x]) if not near_junc[y, x] else -1 for y, x in order]
        if G.nodes[a]["kind"] == "junction":
            pts = np.vstack([G.nodes[a]["pos"], pts]); colors = [-1] + colors
        if G.nodes[b]["kind"] == "junction":
            pts = np.vstack([pts, G.nodes[b]["pos"]]); colors = colors + [-1]
        if len(pts) < 2:
            return
        runs = _runs(colors, min_run=max(4, int(width)))
        prev = a
        for k, (s0, s1, cc) in enumerate(runs):
            last = k == len(runs) - 1
            nxt = b if last else G.add_node(pts[s1 - 1], "color")
            seg = pts[s0:s1]
            if k > 0:
                seg = np.vstack([G.nodes[prev]["pos"], seg])
            if len(seg) >= 2:
                G.add_edge(prev, nxt, seg, cc, layered=layer is not None)
            prev = nxt

    def walk(p0, q):
        """Walk from p0 (an end, or a pixel inside a crossing) through q until the next end or crossing."""
        order, prev, cur = [], p0, q
        while True:
            if in_j(cur):
                return order, cur
            order.append(cur)
            nxt = [r for r in adj[cur] if r != prev]
            if len(nxt) != 1:
                return order, cur
            prev, cur = cur, nxt[0]
            if cur == q and prev != p0:                # closed loop
                return order, None

    for p, nb in adj.items():                          # chains leaving crossings
        if not in_j(p):
            continue
        for q in nb:
            if in_j(q) or (p, q) in visited:
                continue
            order, last = walk(p, q)
            if not order:
                continue
            visited.add((p, q))
            if last is not None and in_j(last):
                visited.add((last, order[-1]))
                add_chain(order, node_at(p), node_at(last))
            else:
                add_chain(order, node_at(p), node_at(order[-1]))
    seen_px = {(int(round(y)), int(round(x))) for E in G.edges for x, y in E["pts"]}
    for p, nb in adj.items():                          # chains between two loose ends (no crossing at all)
        if len(nb) == 1 and not in_j(p) and p not in seen_px:
            order, prev, cur = [p], p, nb[0]
            while not in_j(cur):
                order.append(cur)
                nxt = [r for r in adj[cur] if r != prev]
                if len(nxt) != 1:
                    break
                prev, cur = cur, nxt[0]
            if in_j(cur):
                continue                               # reached a crossing: already walked from that side
            seen_px.update(order)
            if len(order) >= 2:
                add_chain(order, node_at(order[0]), node_at(order[-1]))
    return G


def _hidden_frac(a, b, hide):
    n = max(2, int(np.linalg.norm(b - a)))
    t = np.linspace(0, 1, n)[1:-1] if n > 2 else np.array([0.5])
    p = np.round(a[None] + t[:, None] * (b - a)[None]).astype(int)
    p[:, 0] = np.clip(p[:, 0], 0, hide.shape[1] - 1)
    p[:, 1] = np.clip(p[:, 1], 0, hide.shape[0] - 1)
    return float(hide[p[:, 1], p[:, 0]].mean())


def finish_graph(G, lay, rings, hide=None, attach=False):
    """Drop isolated specks, tag strokes near the LOS, add a node per TD ring, and offer bridges across short
    hidden gaps. attach=True also lets a loose end attach to the middle of a line it stops short of."""
    parent = list(range(len(G.nodes)))

    def f(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for E in G.edges:
        parent[f(E["a"])] = f(E["b"])
    comp_len = {}
    for E in G.edges:
        comp_len[f(E["a"])] = comp_len.get(f(E["a"]), 0.0) + E["length"]
    los_rows = (lay.los_first - 6, lay.los_last + 22)
    near_los, small = {}, set()
    for E in G.edges:
        ys = E["pts"][:, 1]
        if ((ys >= los_rows[0]) & (ys <= los_rows[1])).any():
            near_los[f(E["a"])] = True
    # a short fragment next to another stroke is a real piece of line (a tiny white route before a catch);
    # only an isolated one is a speck
    pts_all = [(E["pts"], f(E["a"])) for E in G.edges]
    if pts_all:
        P = np.vstack([p for p, _ in pts_all])
        C = np.concatenate([np.full(len(p), c) for p, c in pts_all])
        tree = cKDTree(P)
        for c in {c for c, L in comp_len.items() if L < MIN_COMPONENT}:
            mine = P[C == c]
            if not any(np.any(C[idx] != c) for idx in tree.query_ball_point(mine, r=10.0)):
                small.add(c)
    for E in G.edges:
        E["dead"] = f(E["a"]) in small and not near_los.get(f(E["a"]), False)
        if not E["dead"]:
            E["near_start"] = bool(lay.img_to_field(E["pts"])[:, 1].max() <= STEM_Y)
    ends = [n for n in range(len(G.nodes)) if G.loose(n)]
    cos45 = math.cos(math.radians(45))

    def bridge(a, b, color, extra, catch=False):
        if a == b or any(G.other(e, a) == b for e in G.nodes[a]["edges"]):
            return
        G.add_edge(a, b, np.vstack([G.nodes[a]["pos"], G.nodes[b]["pos"]]), color, bridge=True,
                   extra=extra, catch_bridge=catch)

    # loose end -> middle of a line it points at (shared stems, lines stopping short of another line)
    if attach and ends:
        for b in list(ends):
            eb = G.real_edges(b)
            if not eb:
                continue
            eb = eb[0]
            tb = -G.tangent(eb, b)
            pb = G.nodes[b]["pos"]
            best = None
            for e, E in enumerate(G.edges):
                if E["dead"] or E["bridge"] or e == eb or b in (E["a"], E["b"]) or len(E["pts"]) < 9:
                    continue
                d = np.linalg.norm(E["pts"] - pb, axis=1)
                i = int(np.argmin(d))
                if d[i] > ATTACH_MAX or i < 4 or i > len(E["pts"]) - 5:
                    continue
                g = E["pts"][i] - pb
                cosb = float(np.dot(tb, g / (np.linalg.norm(g) or 1)))
                if d[i] > 4 and cosb < 0.7:
                    continue
                if best is None or d[i] < best[0]:
                    best = (d[i], e, i, cosb)
            if best is not None:
                d, e, i, cosb = best
                cb, ce = G.edges[eb]["color"], G.edges[e]["color"]
                m = G.split_edge(e, i)
                if cb == WHITE and ce == GREEN:
                    bridge(b, m, GREEN, BRIDGE_COST * d + 2.0, catch=True)
                else:
                    bridge(b, m, cb, BRIDGE_COST * d + (1 - max(cosb, 0.0)) * 20.0 + 3.0)
    # TD rings: each is a node a completed route can end on
    G.ring_cap = {}
    for x, y, w, area in rings:
        r = G.add_node((x, y), "ring")
        c = np.array([x, y])
        reach = max(RING_REACH, w)
        for n, N in enumerate(G.nodes):
            if n == r or N["kind"] in ("ring", "color") or not G.live_edges(n):
                continue
            g = c - N["pos"]
            d = float(np.linalg.norm(g))
            if d <= reach:
                color = G.edges[G.real_edges(n)[0]]["color"] if G.real_edges(n) else WHITE
                bridge(n, r, color, BRIDGE_COST * d)
            elif G.loose(n) and d <= RING_REACH_HIDDEN and hide is not None:
                e0 = G.real_edges(n)[0]
                cosb = float(np.dot(-G.tangent(e0, n), g / d))
                if cosb >= 0.8 and _hidden_frac(N["pos"], c, hide) >= HIDDEN_FRAC:
                    bridge(n, r, G.edges[e0]["color"], BRIDGE_COST * d + (1 - cosb) * 20.0)
        # two touchdowns to the same spot draw overlapping rings: a ring holds as many as line ends reach it
        G.ring_cap[r] = int(np.clip(sum(1 for e in G.nodes[r]["edges"] if G.nodes[G.other(e, r)]["kind"] == "end"), 1, 3))
    # T-gaps: a loose end that stops just short of a crossing it points at
    for b in ends:
        if not G.real_edges(b):
            continue
        eb = G.real_edges(b)[0]
        tb = -G.tangent(eb, b)
        for a, N in enumerate(G.nodes):
            if a == b or N["kind"] in ("end", "ring") or not G.live_edges(a):
                continue
            gap = N["pos"] - G.nodes[b]["pos"]
            d = float(np.linalg.norm(gap))
            if d < 1e-6 or d > BRIDGE_HIDDEN_MAX:
                continue
            cosb = float(np.dot(tb, gap / d))
            if cosb < cos45:
                continue
            if d > BRIDGE_MAX and (hide is None or _hidden_frac(G.nodes[b]["pos"], N["pos"], hide) < HIDDEN_FRAC):
                continue
            bridge(b, a, G.edges[eb]["color"], BRIDGE_COST * d + (1 - cosb) * 20.0 + 2.0)
    # catch points where the white end sits in a crossing: bridge from that node to a nearby green loose end
    for b in ends:
        if not G.real_edges(b) or G.edges[G.real_edges(b)[0]]["color"] != GREEN:
            continue
        for a, N in enumerate(G.nodes):
            if a == b or G.loose(a) or N["kind"] in ("color", "ring") or not G.live_edges(a):
                continue
            if not any(G.edges[e]["color"] == WHITE for e in G.real_edges(a)):
                continue
            d = float(np.linalg.norm(N["pos"] - G.nodes[b]["pos"]))
            hidden = hide is not None and d <= BRIDGE_CATCH_HIDDEN and \
                _hidden_frac(N["pos"], G.nodes[b]["pos"], hide) >= HIDDEN_FRAC
            if 1e-6 < d <= (BRIDGE_CATCH_HIDDEN if hidden else BRIDGE_CATCH_MAX):
                bridge(a, b, GREEN, BRIDGE_COST * d + 2.0, catch=True)
    # loose end <-> loose end: same colour across a hidden gap, or white end -> green start (the catch seam)
    for i in range(len(ends)):
        for j in range(len(ends)):
            if i == j:
                continue
            a, b = ends[i], ends[j]
            if not G.real_edges(a) or not G.real_edges(b):
                continue
            ea, eb = G.real_edges(a)[0], G.real_edges(b)[0]
            ca, cb = G.edges[ea]["color"], G.edges[eb]["color"]
            gap = G.nodes[b]["pos"] - G.nodes[a]["pos"]
            d = float(np.linalg.norm(gap))
            if d < 1e-6:
                continue
            mid = (G.nodes[a]["pos"] + G.nodes[b]["pos"]) / 2
            over_ring = any(np.hypot(mid[0] - rx, mid[1] - ry) < max(14, rw) for rx, ry, rw, _ in rings)
            if ca == cb:
                if i > j:
                    continue                          # same-colour bridges are symmetric: make once
                lim = BRIDGE_RING_MAX if over_ring else BRIDGE_MAX
                if d > lim and hide is not None and d <= BRIDGE_HIDDEN_MAX and \
                        _hidden_frac(G.nodes[a]["pos"], G.nodes[b]["pos"], hide) >= HIDDEN_FRAC:
                    lim = BRIDGE_HIDDEN_MAX               # hidden under another line / a ring / the TD label
            elif ca == WHITE and cb == GREEN:
                lim = BRIDGE_CATCH_MAX                # catch point: white route end -> green run start
                if d > lim and hide is not None and d <= BRIDGE_CATCH_HIDDEN and \
                        _hidden_frac(G.nodes[a]["pos"], G.nodes[b]["pos"], hide) >= HIDDEN_FRAC:
                    lim = BRIDGE_CATCH_HIDDEN
            else:
                continue
            if d > lim:
                continue
            if ca != cb:                                # catch point: any direction change is allowed
                bridge(a, b, GREEN, BRIDGE_COST * d, catch=True)
                continue
            ta, tb = -G.tangent(ea, a), -G.tangent(eb, b)      # outward directions at the loose ends
            u = gap / d
            ca_, cb_ = float(np.dot(ta, u)), float(np.dot(tb, -u))
            if ca_ < cos45 or cb_ < cos45:
                continue
            bridge(a, b, ca, BRIDGE_COST * d + (2 - ca_ - cb_) * 20.0)
    G._tan.clear()
    return G


# ----------------------------------------------------------------------------------------------- paths
def _edge_step(G, e, rtype, phase):
    """(cost, new phase) of travelling edge e. Colour rules are soft: running over another colour costs
    MISMATCH per px for crossings / shared stems near the LOS and MISMATCH_LONG per px otherwise."""
    E = G.edges[e]
    c, L = E["color"], E["length"]
    crossing = L <= CROSS_LEN * G.width
    per = MISMATCH if (crossing or E["near_start"]) else MISMATCH_LONG
    if E["bridge"]:
        extra = E["extra"]
        if E["catch_bridge"]:
            return (extra, "after") if (rtype == "COMPLETE" and phase == "route") else None
        if rtype == "INCOMPLETE":
            return extra + (0 if c == GRAY else MISMATCH_LONG * L), phase
        if phase == "route":
            return (extra, "after") if c == GREEN else (extra + (0 if c == WHITE else MISMATCH_LONG * L), "route")
        return extra + (0 if c == GREEN else MISMATCH_LONG * L), "after"
    if rtype == "INCOMPLETE":
        return (0.0 if c == GRAY else per * L), phase
    if phase == "route":
        if c == WHITE:
            return 0.0, "route"
        if c == GREEN:
            return 0.0, "after"
        return per * L, "route"
    return (0.0 if c == GREEN else per * L), "after"


def _relax(G, heap, best, prev, rtype, penalty):
    while heap:
        c, (e, n, ph) = heapq.heappop(heap)
        if c > best.get((e, n, ph), np.inf):
            continue
        kind = G.nodes[n]["kind"]
        if kind == "ring":
            continue                                   # a route ends on its ring
        d_in = G.arrive(e, n) if (kind != "color" and not G.edges[e].get("jump")) else None
        for f in G.live_edges(n):
            if f == e:
                continue
            st = _edge_step(G, f, rtype, ph)
            if st is None:
                continue
            tc = 0.0
            if d_in is not None and not G.edges[f].get("jump"):
                turn = _angle(d_in, G.tangent(f, n))
                if (ph == "route" and st[1] == "after") or G.edges[e]["catch_bridge"]:
                    tc = CATCH_TURN * turn                # the catch: receivers often turn sharply upfield
                    if ph == "route" and st[1] == "after":
                        tc += _hint_cost(G, n)
                elif turn > MAX_TURN:
                    continue
                else:
                    tc = turn_cost(turn)
            state = (f, G.other(f, n), st[1])
            c2 = c + tc + st[0] + penalty.get(f, 0.0)
            if c2 < best.get(state, np.inf):
                best[state], prev[state] = c2, (e, n, ph)
                heapq.heappush(heap, (c2, state))
    return best, prev


def _shortest(G, s, s_cost, rtype, penalty):
    """Dijkstra over (edge, node heading to, phase) states from start node s."""
    best, prev, heap = {}, {}, []
    for e in G.live_edges(s):
        st = _edge_step(G, e, rtype, "route")
        if st is None or G.edges[e]["catch_bridge"]:
            continue                                      # a route runs before it is caught: never starts on the catch
        extra = 0.0
        if rtype == "COMPLETE" and st[1] == "after":      # white part too short to see: a catch at/behind the LOS
            if G.start_y.get(s, 0.0) > GREEN_START_MAX_Y:
                continue
            extra = GREEN_START
        state = (e, G.other(e, s), st[1])
        c = s_cost + st[0] + extra + penalty.get(e, 0.0)
        if c < best.get(state, np.inf):
            best[state], prev[state] = c, None
            heapq.heappush(heap, (c, state))
    return _relax(G, heap, best, prev, rtype, penalty)


def _rebuild(prev, state, s):
    edges, nodes, phases = [], [], []
    while state is not None:
        e, n, ph = state
        edges.append(e)
        nodes.append(n)
        phases.append(ph)
        state = prev[state]
    return [s] + nodes[::-1], edges[::-1], phases[::-1]


def _end_cost(G, m):
    """Cost of a route ending at node m, or None if it can't end there."""
    kind = G.nodes[m]["kind"]
    if kind == "ring" or G.loose(m):
        return 0.0
    if kind in ("junction", "split"):
        return HIDDEN_END
    return None


def _path(rtype, nodes, edges, phases, cost, G):
    catch_i = next((i for i, p in enumerate(phases) if p == "after"), None) if rtype == "COMPLETE" else None
    on_ring = rtype == "COMPLETE" and G.nodes[nodes[-1]]["kind"] == "ring"
    return (rtype, nodes, edges, cost, catch_i, on_ring)


def candidate_paths(G, lay, alts=3):
    """The cheapest few routes from every possible start to every possible end, for both route types:
    tuples (type, nodes, edges, cost, catch index, ends on a TD ring)."""
    starts = []
    for n, N in enumerate(G.nodes):
        if N["kind"] in ("color", "ring") or not G.live_edges(n):
            continue
        y = float(lay.img_to_field(N["pos"][None])[0, 1])
        if START_MIN_Y <= y <= START_MAX_Y:
            far = START_PER_YD * max(0.0, y - START_FREE_Y)
            if G.loose(n):
                starts.append((n, far, y))
            elif N["kind"] == "junction":
                starts.append((n, HIDDEN_START + far, y))
    G.starts = [(n, c) for n, c, _ in starts]
    G.start_y = {n: y for n, _, y in starts}
    G.fwd = {}
    out, seen, per_pair = [], set(), {}
    for s, s_cost, _ in starts:
        for rtype in ("COMPLETE", "INCOMPLETE"):
            penalty = {}
            for k in range(alts):
                best, prev = _shortest(G, s, s_cost, rtype, penalty)
                if k == 0:
                    G.fwd[(s, rtype)] = (best, prev)
                by_end = {}
                for (e, n, ph), c in best.items():
                    if n == s or (G.edges[e]["bridge"] and G.nodes[n]["kind"] != "ring"):
                        continue                      # a route ends on a drawn stroke (or its ring), not mid-bridge
                    ec = _end_cost(G, n)
                    if ec is None or (ec > 0 and k > 0):
                        continue                      # hidden ends: first pass only
                    if rtype == "COMPLETE" and ph == "route":
                        ec += _hint_cost(G, n)        # caught at the end of the white line (no run after)
                    if c + ec < by_end.get(n, (np.inf,))[0]:
                        by_end[n] = (c + ec, (e, n, ph))
                got, snap, used = 0, dict(penalty), []
                for n, (tot, state) in by_end.items():
                    nodes, edges, phases = _rebuild(prev, state, s)
                    key = (rtype, tuple(edges))
                    if key in seen:
                        continue
                    seen.add(key)
                    cost = tot - sum(snap.get(e, 0.0) for e in edges) + PATH_BASE
                    hidden_end = (_end_cost(G, n) or 0.0) > 0
                    if cost > CAND_MAX_COST or (hidden_end and cost > HIDDEN_END_MAX_COST):
                        continue
                    if per_pair.get((s, n, rtype), 0) >= CAND_PER_PAIR:
                        continue
                    per_pair[(s, n, rtype)] = per_pair.get((s, n, rtype), 0) + 1
                    out.append(_path(rtype, nodes, edges, phases, cost, G))
                    got += 1
                    used += edges
                for e in used:                        # discourage these edges in the next, alternative pass
                    penalty[e] = penalty.get(e, 0.0) + 15.0
                if not got:
                    break
    return out


def through_paths(G, targets):
    """For each stroke in `targets`, the cheapest full route (start -> stroke -> end) of each type."""
    out = []
    for rtype in ("COMPLETE", "INCOMPLETE"):
        fwd = {}
        for s, _ in G.starts:
            best, prev = G.fwd.get((s, rtype), ({}, {}))
            for st, c in best.items():
                if c < fwd.get(st, (np.inf,))[0]:
                    fwd[st] = (c, s, prev)
        for e in targets:
            for n in (G.edges[e]["a"], G.edges[e]["b"]):
                for ph in ("route", "after"):
                    st = (e, n, ph)
                    if st not in fwd:
                        continue
                    c0, s, prev0 = fwd[st]
                    best, prev = _relax(G, [(c0, st)], {st: c0}, {st: None}, rtype, {})
                    by_end = None
                    for (f, m, ph2), c in best.items():
                        if m == s or (G.edges[f]["bridge"] and G.nodes[m]["kind"] != "ring"):
                            continue
                        ec = _end_cost(G, m)
                        if ec is not None and (by_end is None or c + ec < by_end[0]):
                            by_end = (c + ec, (f, m, ph2))
                    if by_end is None:
                        continue
                    tot, end_state = by_end
                    tail_nodes, tail_edges, tail_ph = _rebuild(prev, end_state, None)
                    head_nodes, head_edges, head_ph = _rebuild(prev0, st, s)
                    out.append(_path(rtype, head_nodes + tail_nodes[2:], head_edges + tail_edges[1:],
                                     head_ph + tail_ph[1:], tot + PATH_BASE, G))
    return out


# ----------------------------------------------------------------------------------------------- choose
def choose(G, cands, rec, inc, td, min_cover):
    P = len(cands)
    if P == 0:
        return [], {}
    end_nodes = [n for n in range(len(G.nodes)) if G.loose(n)]
    rings = [n for n in range(len(G.nodes)) if G.nodes[n]["kind"] == "ring"]
    V = len(end_nodes)
    cover = [e for e, E in enumerate(G.edges) if not E["dead"] and not E["bridge"] and E["length"] >= min_cover]
    C = len(cover)
    # x_p (P) | start flags y_v (V) | count slacks (6) | unexplained-end slacks (V) | uncovered slacks (C)
    o_y, o_cs, o_ue, o_uc = P, P + V, P + V + 6, P + V + 6 + V
    nvar = o_uc + C
    c = np.zeros(nvar)
    c[:P] = [p[3] for p in cands]
    c[o_cs:o_cs + 6] = SLACK
    c[o_ue:o_ue + V] = SLACK / 4
    c[o_uc:] = SLACK / 4
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
    for k, e in enumerate(cover):                      # every stroke covered
        add([(j, 1) for j in by_edge.get(e, [])] + [(o_uc + k, 1)], 1, np.inf)
    for i, n in enumerate(end_nodes):                  # loose ends: start of >=1 route / end of 1 / passed through
        ends = ends_at.get(n, [])
        add([(j, 1) for j in ends], 0, 1)
        add([(j, 1) for j in ends] + [(j, 1) for j in through.get(n, []) if j not in ends]
            + [(o_y + i, 1), (o_ue + i, 1)], 1, np.inf)
        add([(j, 1) for j in starts_at.get(n, [])] + [(o_y + i, -1)], 0, np.inf)
    for r in rings:                                    # one touchdown per ring (two if rings overlap)
        js = [j for j in ends_at.get(r, []) if cands[j][5]]
        if js:
            add([(j, 1) for j in js], 0, G.ring_cap.get(r, 1))

    def count(pred, k, target):
        add([(j, 1) for j, p in enumerate(cands) if pred(p)] + [(o_cs + k, -1), (o_cs + k + 1, 1)], target, target)
    if rec is not None:
        count(lambda p: p[0] == "COMPLETE", 0, rec)
    if inc is not None:
        count(lambda p: p[0] == "INCOMPLETE", 2, inc)
    if td is not None and rings:
        count(lambda p: p[5], 4, td)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(lo), nvar))
    integ = np.zeros(nvar)
    integ[:o_cs] = 1
    ub = np.full(nvar, np.inf)
    ub[:o_cs] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")               # "threads" is passed through to HiGHS verbatim
        # one thread: charts are read many at a time, and multi-threaded solves running side by side stall
        res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(np.zeros(nvar), ub),
                   options={"time_limit": MILP_SECONDS, "mip_rel_gap": 0.01, "threads": 1})
    if res.x is None:
        return [], dict(status=res.status)
    x = res.x
    dbg = dict(uncovered=[cover[k] for k in range(C) if x[o_uc + k] > 0.5],
               unexplained=[end_nodes[i] for i in range(V) if x[o_ue + i] > 0.5],
               count_slack=np.round(x[o_cs:o_cs + 6], 2).tolist(), obj=float(res.fun))
    return [cands[j] for j in range(P) if x[j] > 0.5], dbg


# ----------------------------------------------------------------------------------------------- read
def _path_pixels(G, nodes, edges):
    pts = []
    for k, e in enumerate(edges):
        p = G.oriented(e, nodes[k])
        pts.append(p if not pts else p[1:])
    return np.vstack(pts) if pts else np.zeros((0, 2))


def _add_jumps(G, lay, rings):
    """
    Some TD rings have no route line leading to them: NGS drew the ring (and maybe a short piece of the run
    into it) but not the line connecting it to the route. For each such ring, let any completed route's line
    end jump - in a straight line, for JUMP_COST - to the loose end of the piece attached to the ring (or to the
    ring itself when nothing is drawn there). Used only when a TD has to be explained; returns the number added.
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
    fy = lambda n: float(lay.img_to_field(G.nodes[n]["pos"][None])[0, 1])
    loose = [n for n in range(len(G.nodes)) if G.loose(n)]
    added = 0
    for r in rings:
        attached = [n for n in loose if f(n) == f(r)]
        targets = [max(attached, key=lambda n: float(np.hypot(*(G.nodes[n]["pos"] - G.nodes[r]["pos"]))))] \
            if attached else [r]                     # the far end of the piece drawn into the ring
        for t in targets:
            ty = fy(t)
            ct = G.edges[G.real_edges(t)[0]]["color"] if t != r else None
            for a in loose:
                if f(a) == f(r):
                    continue
                ca = G.edges[G.real_edges(a)[0]]["color"]
                if ca == GRAY or not (0.0 <= fy(a) <= ty + 5.0):
                    continue
                d = float(np.hypot(*(G.nodes[t]["pos"] - G.nodes[a]["pos"])))
                color = ca if ct is None else ct
                G.add_edge(a, t, np.vstack([G.nodes[a]["pos"], G.nodes[t]["pos"]]), color, bridge=True, jump=True,
                           catch_bridge=(ca == WHITE and ct == GREEN), extra=JUMP_COST + 0.02 * d)
                added += 1
    G._tan.clear()
    return added


def _jump_to_ring(G, cands):
    """A jump exists only to reach a TD ring: drop candidates that take one and end anywhere else."""
    return [p for p in cands if p[5] or not any(G.edges[e].get("jump") for e in p[2])]


HINT_COST = 12.0          # per yd a completed route's catch point lies from the nearest catch hint (past HINT_FREE)
HINT_FREE, HINT_CAP = 1.5, 4.0           # (capped so a catch the hints don't cover - no dot - stays affordable)


def _hint_cost(G, n):
    """Cost of a catch at node n: by how far it is from the nearest catch hint (G.hint_xy, image px per yd scaled
    by the field calibration). 0 when there are no hints."""
    H = getattr(G, "hint_xy", None)
    if H is None or not len(H):
        return 0.0
    hc = G._hint_cache.get(n)
    if hc is None:
        xy = G.hint_lay.img_to_field(np.asarray(G.nodes[n]["pos"], float)[None])[0]
        d = float(np.min(np.hypot(H[:, 0] - xy[0], H[:, 1] - xy[1])))
        hc = G._hint_cache[n] = HINT_COST * min(max(d - HINT_FREE, 0.0), HINT_CAP)
    return hc


def read_routes(image, expected, mode="union", lay=None, pix=None):
    """
    expected: receptions, touchdowns, targets (None = unknown), max_yards, and optionally catches: field (x, y) of
    each completion (e.g. the QB chart's dots for this receiver's catches): the path search charges HINT_COST per yd
    a catch lands away from the nearest one, so tangles are untangled where the ball really went. mode: "union" or "layers" (see the
    module docstring). lay / pix (the line_colors() result) may be passed in to avoid recomputing them.
    Returns (routes, info) with routes in detect_routes() format: route_type, td, pts (field yd), seg
    ('route' / 'after_catch'), start_ok - plus px (the path in image pixels) and catch_px.
    """
    im = K.read_image(image)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    lay = lay or K.calibrate(im)
    rings = find_rings(hsv, lay, im.shape[:2])
    col, union, hide, layers = pix or line_colors(im, hsv, lay, rings)
    width = stroke_width(union)
    if mode == "layers":
        G = Graph()
        for colour, mask in layers.items():
            if mask.any():
                build_graph(None, mask, G=G, width=width, layer=colour)
        G.width = width
        G = finish_graph(G, lay, rings, hide, attach=True)
    else:
        G = finish_graph(build_graph(col, union, width=width), lay, rings, hide, attach=True)
    hints = expected.get("catches")
    G.hint_xy = np.asarray(hints, float).reshape(-1, 2) if hints is not None and len(hints) else None
    G.hint_lay, G._hint_cache = lay, {}
    cands = candidate_paths(G, lay)
    if expected.get("touchdowns"):
        reached = {p[1][-1] for p in cands}
        orphans = [n for n, N in enumerate(G.nodes) if N["kind"] == "ring" and n not in reached]
        if orphans and _add_jumps(G, lay, orphans):
            cands = _jump_to_ring(G, candidate_paths(G, lay))
    rec, td, tg = expected.get("receptions"), expected.get("touchdowns"), expected.get("targets")
    inc = (tg - rec) if (tg is not None and rec is not None) else None
    min_cover = 2.5 * G.width
    chosen, dbg = choose(G, cands, rec, inc, td, min_cover)
    seen = {(p[0], tuple(p[2])) for p in cands}
    for _ in range(2):                                 # column generation: routes through what's left uncovered
        todo = dbg.get("uncovered", [])
        if not todo:
            break
        new = _jump_to_ring(G, [p for p in through_paths(G, todo) if (p[0], tuple(p[2])) not in seen])
        if not new:
            break
        seen |= {(p[0], tuple(p[2])) for p in new}
        cands += new
        chosen, dbg = choose(G, cands, rec, inc, td, min_cover)
    out = []
    for rtype, nodes, edges, cost, catch_i, on_ring in chosen:
        px = _path_pixels(G, nodes, edges)
        if len(px) < 2:
            continue
        pts, truncated = _R().resample_field(lay, px, ceiling=expected.get("max_yards"))
        if len(pts) < 2:
            continue
        seg = np.array(["route"] * len(pts), dtype=object)
        catch_px = None
        if rtype == "COMPLETE" and catch_i is not None:
            n_route_px = len(_path_pixels(G, nodes[:catch_i + 1], edges[:catch_i])) if catch_i > 0 else 1
            catch_px = px[min(max(n_route_px - 1, 0), len(px) - 1)]
            split = lay.img_to_field(catch_px[None])[0]
            seg[int(np.argmin(np.linalg.norm(pts - split, axis=1))):] = "after_catch"
        out.append(dict(route_type=rtype, td=bool(on_ring), pts=pts, seg=seg,
                        start_ok=bool(pts[0][1] <= _R().MAX_START_Y), catch_px=catch_px, px=px,
                        truncated=bool(truncated), gap=any(G.edges[e].get("jump") for e in edges)))
    out.sort(key=lambda r: (r["pts"][0][1], r["pts"][0][0]))
    return out, dict(graph=G, cands=len(cands), cand_list=cands, rings=[(r[0], r[1]) for r in rings],
                     lay=lay, dbg=dbg, mode=mode)
