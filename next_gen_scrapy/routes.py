"""
Extract full route paths from a current-format Next Gen Stats *route chart* image.

Drawn on the field: white line = route of a completed target (up to the catch), green = yards after
catch, gray = route of an incomplete target, blue ring = touchdown (with a 'TD' text label beside it).
Each route is returned as an ordered list of field coordinates (x, y in yards) from the start
of the route (near the line of scrimmage) to its end.
"""
from collections import Counter

import cv2
import numpy as np

from . import calib as K
from . import paths as NP
from .passes import detect_blue_rings

RESAMPLE_YD = 0.5
MAX_START_Y = 1.0            # a route starts at the line of scrimmage (or behind it), never ahead of it

# Overlay text printed outside the sidelines ('LOS', '+10', '+20' ...) is masked in calib.text_mask,
# but a clipped glyph can still leave a small closed contour behind. A real route is never a tiny
# closed loop parked on the sideline, so drop those: checked against 51.5k scraped routes, this
# removes 134 glyph fragments and zero genuine routes (no real route's centroid reaches the sideline).
GLYPH_MAX_SPAN_YD = 6.0      # bounding box of the fragment
GLYPH_MAX_GAP_YD = 2.5       # end returns to the start => closed loop
GLYPH_MIN_ABS_X_YD = 24.0    # centred at/outside the sideline (26.67)


def is_label_glyph(pts):
    """A closed, tiny contour sitting on the sideline is overlay text, not a route."""
    x, y = pts[:, 0], pts[:, 1]
    span = max(x.max() - x.min(), y.max() - y.min())
    gap = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    centre_x = abs((x.min() + x.max()) / 2)
    return span < GLYPH_MAX_SPAN_YD and gap < GLYPH_MAX_GAP_YD and centre_x >= GLYPH_MIN_ABS_X_YD


def resample_field(lay, px_path, step=RESAMPLE_YD, smooth=5, ceiling=None):
    """
    Pixel polyline -> evenly spaced (by arc length) field-coordinate polyline.

    Truncated at min(lay.max_reliable_yard, ceiling): past the deepest yard line the chart's own
    calibration could verify, row_to_yard is extrapolating the perspective curve past any real
    gridline evidence, and a small pixel wobble in a traced path gets amplified into yards the
    further out it goes (see calib.calibrate). `ceiling`, when given, is a tighter box-score bound
    (a single route/carry/pass can never be deeper than the player's own game total for that stat) -
    a second, independent check, since the gridline-detection noise that max_reliable_yard guards
    against can itself occasionally produce a confident-looking but wrong deep anchor (a handful of
    spuriously-aligned candidate rows can vote past real gridlines with no nearby corroboration).
    Returns (pts, truncated) - truncated is True if points were cut.
    """
    p = np.asarray(px_path, float)
    if len(p) > smooth * 2:                               # light moving-average, endpoints kept
        k = np.ones(smooth) / smooth
        core = np.column_stack([np.convolve(p[:, c], k, mode="valid") for c in (0, 1)])
        p = np.vstack([p[:1], core, p[-1:]])
    f = lay.img_to_field(p)
    seg = np.linalg.norm(np.diff(f, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return f[:1], False
    grid = np.unique(np.concatenate([np.arange(0, s[-1], step), [s[-1]]]))
    pts = np.column_stack([np.interp(grid, s, f[:, 0]), np.interp(grid, s, f[:, 1])])
    ceiling = lay.max_reliable_yard if ceiling is None else min(ceiling, lay.max_reliable_yard)
    over = np.where(pts[:, 1] > ceiling)[0]
    if len(over):
        # over[0] can be 0: a line traced entirely beyond the calibrated grid (not a real route at
        # all, just some other artifact in the deep, ungridded part of the chart art) truncates to
        # nothing. The caller drops anything left with fewer than 2 points - not enough for a path.
        return pts[:over[0]], True
    return pts, False


def _endpoints(p):
    return p[0], p[-1]


def _closest_end(p, pt):
    d0, d1 = np.linalg.norm(p[0] - pt), np.linalg.norm(p[-1] - pt)
    return (0, d0) if d0 <= d1 else (1, d1)


def td_label_mask(shape, rings):
    """Box where the 'TD' text is drawn: up and to the right of each touchdown ring."""
    m = np.zeros(shape, np.uint8)
    for cx, cy in rings:
        cv2.rectangle(m, (int(cx + 12), int(cy - 48)), (int(cx + 85), int(cy - 8)), 1, -1)
    return m.astype(bool)


def line_masks(im, rings, lay):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    allowed = lay.on_card_field(im.shape[:2]) & ~td_label_mask(im.shape[:2], rings)
    white = (s < 30) & (v > 215) & allowed
    green = (h >= 38) & (h <= 70) & (s > 120) & (v > 100) & allowed
    gray = (s < 18) & (v > 95) & (v < 200) & allowed
    gray &= ~cv2.dilate(white.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)   # AA fringe
    gray = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    return white, gray, green


def detect_routes(image, expected=None):
    """
    `image` is a file path or an already-decoded BGR array (e.g. from cv2.imdecode).
    `expected` (optional): {"receptions": n, "touchdowns": n} from the chart metadata. If more white
    fragments than receptions are found, the best-fitting fragments are merged until the counts agree.
    `expected["max_yards"]` (optional): the player's total receivingYards for the game, from the same
    chart metadata - a single route can never be deeper than that, which is a useful independent check
    on top of the chart's own calibration (see resample_field).
    Returns (routes, info). routes: list of dict(route_type, td, pts (Nx2 field yd), seg (N labels)).
    route_type is COMPLETE (white line, maybe with after-catch) or INCOMPLETE (gray line).
    """
    ctx = _prepare(image, expected)
    return _finalize(ctx, ctx["reconcile"](ctx["white_pieces"], ctx["receptions"]))


def _prepare(image, expected):
    """Calibrate the chart and trace its raw line pieces (white / grey / green), not yet reconciled."""
    im = K.read_image(image)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    lay = K.calibrate(im)
    ceiling = None if expected is None else expected.get("max_yards")
    rings = detect_blue_rings(hsv, lay, None if expected is None else expected.get("touchdowns"))
    white_m, gray_m, green_m = line_masks(im, rings, lay)

    grow = lambda m: cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    ring_m = np.zeros(white_m.shape, np.uint8)
    for cx, cy in rings:
        cv2.circle(ring_m, (int(cx), int(cy)), 18, 1, -1)

    def trace(mask, min_len, others):
        occ = grow(others[0] | others[1] | ring_m.astype(bool))
        ps = NP.join_collinear(NP.extract_paths(mask, min_len=6), max_gap=90, max_turn_deg=40, occluder=occ)
        return [p for p in ps if NP._length(p) >= min_len]

    def reconcile(paths, want, sliver=45):
        """
        Bring a set of traced lines down to `want` of them: first re-join pieces of one line that the
        tracer split (progressively looser), then drop leftover slivers. Used with the counts from the
        chart metadata, which say exactly how many lines the chart drew.
        """
        paths = list(paths)
        if want is None or want < 1:
            return paths
        for gap, turn in ((30, 45), (60, 70), (90, 120)):
            if len(paths) <= want:
                break
            paths = NP.join_collinear(paths, max_gap=gap, max_turn_deg=turn,
                                      max_merges=len(paths) - want)
        paths.sort(key=NP._length)
        while len(paths) > want and NP._length(paths[0]) < sliver:
            paths.pop(0)
        return paths

    targets = None if expected is None else expected.get("targets")
    return dict(lay=lay, rings=rings, expected=expected, ceiling=ceiling, targets=targets,
                receptions=None if expected is None else expected.get("receptions"),
                white_pieces=trace(white_m, 25, (gray_m, green_m)),
                gray_pieces=trace(gray_m, 25, (white_m, green_m)),
                green=trace(green_m, 6, (white_m, gray_m)), reconcile=reconcile)


def _finalize(ctx, white):
    """Everything after the white (completed) route lines are fixed: incomplete lines, after-catch
    runs, touchdown rings, orientation, shared stems, resampling to field yards."""
    lay, rings, expected, ceiling, targets = ctx["lay"], ctx["rings"], ctx["expected"], ctx["ceiling"], ctx["targets"]
    green = ctx["green"]
    gray = ctx["gray_pieces"]
    # targets = every line the chart draws, so the incomplete (grey) ones are targets - receptions
    if targets is not None and expected.get("receptions") is not None:
        gray = ctx["reconcile"](gray, targets - len(white))

    routes = [dict(route_type="COMPLETE", px=w, seg=np.array(["route"] * len(w)), td=False) for w in white]
    routes += [dict(route_type="INCOMPLETE", px=g, seg=np.array(["route"] * len(g)), td=False) for g in gray]

    # after-catch segments: join to the white route end that they continue from. Nearness alone is
    # not enough (several routes can end at the same touchdown ring), so the two ends must also line
    # up: their outward directions must be opposed.
    def end_tan(p, end):
        return NP._end_tangent(p if end == 1 else p[::-1])

    orphans = 0
    for g in green:
        best = None
        for r in routes:
            if r["route_type"] != "COMPLETE" or (r["seg"] == "after_catch").any():
                continue
            for end in (0, 1):
                for gend in (0, 1):
                    a = r["px"][0] if end == 0 else r["px"][-1]
                    b = g[0] if gend == 0 else g[-1]
                    d = np.linalg.norm(a - b)
                    if d >= 25:
                        continue
                    cost = d + 25 * (1 + float(np.dot(end_tan(r["px"], end), end_tan(g, gend))))
                    if best is None or cost < best[0]:
                        best = (cost, r, end, gend)
        if best is not None:
            _, r, end, gend = best
            base = r["px"] if end == 1 else r["px"][::-1]
            bseg = r["seg"] if end == 1 else r["seg"][::-1]
            gg = g if gend == 0 else g[::-1]
            r["px"] = np.vstack([base, gg])
            r["seg"] = np.concatenate([bseg, np.array(["after_catch"] * len(gg))])
        else:
            orphans += 1

    # touchdown rings: extend the nearest route end to the ring centre
    ring_used = 0
    for cx, cy in rings:
        best = None
        for r in routes:
            for end in (0, 1):
                a = r["px"][0] if end == 0 else r["px"][-1]
                d = np.linalg.norm(a - np.array([cx, cy]))
                if best is None or d < best[0]:
                    best = (d, r, end)
        if best is not None and best[0] < 40:
            _, r, end = best
            pt = np.array([[cx, cy]])
            if end == 1:
                r["px"] = np.vstack([r["px"], pt]); r["seg"] = np.concatenate([r["seg"], ["route"]])
            else:
                r["px"] = np.vstack([pt, r["px"]]); r["seg"] = np.concatenate([["route"], r["seg"]])
            r["td"] = True
            r["td_end"] = end
            ring_used += 1

    oriented = []
    for r in routes:
        px, seg = r["px"], r["seg"]
        # orientation: away from the after-catch / touchdown end, else start at the end nearest the LOS
        ends_after = seg[-1] == "after_catch", seg[0] == "after_catch"
        if ends_after[0] or (r["td"] and r.get("td_end") == 1):
            pass                                            # already start -> end
        elif ends_after[1] or (r["td"] and r.get("td_end") == 0):
            px, seg = px[::-1], seg[::-1]
        else:
            fy = lay.img_to_field(np.array([px[0], px[-1]]))[:, 1]
            if fy[0] > fy[1]:
                px, seg = px[::-1], seg[::-1]
        oriented.append(dict(route_type=r["route_type"], kind=r["route_type"], px=px, seg=seg))

    # Routes that leave the same spot on the line of scrimmage share their first few yards, so the tracer
    # can hand that stretch to one of them and leave the others "starting" mid-route; a route that runs
    # under another line can lose its tail. Cut routes traced fused at their start, then give every route
    # that does not start at the line of scrimmage the stretch it is missing.
    # Only split a V where the chart says a line is still missing. A route may legitimately come back
    # towards the line of scrimmage (a comeback, a screen), so splitting on shape alone invents routes.
    want = {"COMPLETE": None if expected is None else expected.get("receptions")}
    want["INCOMPLETE"] = (None if (targets is None or want["COMPLETE"] is None)
                          else targets - want["COMPLETE"])
    counts = Counter(o["route_type"] for o in oriented)
    split = []
    for o in oriented:
        pieces = NP.split_v([o], lay.img_to_field)
        w = want.get(o["route_type"])
        if len(pieces) > 1 and w is not None and counts[o["route_type"]] + len(pieces) - 1 > w:
            pieces = [o]                       # splitting would overshoot the count the chart gives
        else:
            counts[o["route_type"]] += len(pieces) - 1
        split += pieces
    oriented = split
    NP.attach_trunks(oriented, lay.img_to_field, MAX_START_Y)
    NP.attach_trunks(oriented, lay.img_to_field, MAX_START_Y, max_dist=70.0, min_align=0.0, passes=2)

    if expected is not None:                    # still too many completed routes: drop tiny leftover slivers
        def yd_len(o):
            f = lay.img_to_field(o["px"])
            return float(np.linalg.norm(np.diff(f, axis=0), axis=1).sum())
        while True:
            comp = sorted((o for o in oriented if o["route_type"] == "COMPLETE"), key=yd_len)
            if len(comp) <= expected["receptions"] or yd_len(comp[0]) >= 6.0:
                break
            oriented = [o for o in oriented if o is not comp[0]]   # identity: dicts hold arrays

    # each touchdown ring marks the end of exactly one route: the one whose end is nearest to it
    for o in oriented:
        o["td"] = False
    for cx, cy in rings:
        dists = [np.linalg.norm(o["px"][-1] - np.array([cx, cy])) for o in oriented]
        if dists and min(dists) < 45:
            oriented[int(np.argmin(dists))]["td"] = True

    out = []
    n_glyphs = 0
    n_deep_truncated = 0
    n_deep_dropped = 0
    for o in oriented:
        px, seg = o["px"], o["seg"]
        pts, truncated = resample_field(lay, px, ceiling=ceiling)
        if truncated:
            n_deep_truncated += 1
            if len(pts) < 2:          # the whole line lived beyond the calibrated grid
                n_deep_dropped += 1
                continue
        if is_label_glyph(pts):
            n_glyphs += 1
            continue
        # per-point segment label via nearest original pixel
        seg_f = np.array(["route"] * len(pts), dtype=object)
        if (seg == "after_catch").any():
            first_ac = np.argmax(seg == "after_catch")
            split = lay.img_to_field(px[first_ac:first_ac + 1])[0]
            d = np.linalg.norm(pts - split, axis=1)
            seg_f[int(np.argmin(d)):] = "after_catch"
        out.append(dict(route_type=o["route_type"], td=o["td"], pts=pts, seg=seg_f,
                        start_ok=bool(pts[0][1] <= MAX_START_Y)))
    out.sort(key=lambda r: (r["pts"][0][1], r["pts"][0][0]))
    return out, dict(depth_yd=round(lay.max_reliable_yard, 1), n_white=len(white), n_gray=len(gray), n_green=len(green), orphan_green=orphans,
                     n_td_rings=len(rings), n_label_glyphs=n_glyphs, n_deep_truncated=n_deep_truncated,
                     n_deep_dropped=n_deep_dropped)


# ---------------------------------------------------------------------------------------------------
# Alternative readings of a tangled chart
#
# When the tracer finds more white pieces than the chart has catches, some pieces belong to the same
# route and must be joined - but where routes cross, several joinings can look about equally good to
# geometry alone. route_alternatives() enumerates the most plausible ones, so a later step can pick
# the one that fits the real plays best (see next_gen_scrapy.retrace).
# ---------------------------------------------------------------------------------------------------

ALT_MAX_GAP_PX = 90.0
ALT_MAX_TURN_DEG = 120.0


def _join_candidates(pieces, max_gap=ALT_MAX_GAP_PX, max_turn_deg=ALT_MAX_TURN_DEG):
    """Every plausible end-to-end join: (score, i, end_i, j, end_j), lower score = more natural."""
    cos_t = np.cos(np.radians(max_turn_deg))
    out = []
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            for ei in (0, 1):
                for ej in (0, 1):
                    a = pieces[i] if ei else pieces[i][::-1]
                    b = pieces[j] if ej else pieces[j][::-1]
                    gap_v = b[-1] - a[-1]
                    gap = float(np.linalg.norm(gap_v))
                    if gap > max_gap:
                        continue
                    ta, tb = NP._end_tangent(a), NP._end_tangent(b)
                    if np.dot(ta, tb) > -cos_t:
                        continue
                    if gap > 1e-6:
                        u = gap_v / gap
                        if np.dot(ta, u) < cos_t or np.dot(tb, -u) < cos_t:
                            continue
                    out.append((gap + 20 * (1 + float(np.dot(ta, tb))), i, ei, j, ej))
    return sorted(out)


def _assemble(pieces, joins):
    """Chain pieces through the chosen joins (each end used at most once, no cycles) into paths."""
    partner = {}
    for _, i, ei, j, ej in joins:
        partner[(i, ei)] = (j, ej)
        partner[(j, ej)] = (i, ei)
    seen, paths = set(), []
    for start in range(len(pieces)):
        if start in seen:
            continue
        # walk to one free end of this chain first
        cur, end_in = start, 0
        visited = {start}
        while (cur, end_in) in partner and partner[(cur, end_in)][0] not in visited:
            cur, e = partner[(cur, end_in)]
            visited.add(cur)
            end_in = 1 - e
        # now walk forward from that free end, building the path
        seq, i, free_end = [], cur, end_in
        while True:
            seen.add(i)
            p = pieces[i] if free_end == 0 else pieces[i][::-1]        # piece runs away from free_end
            seq.append(p)
            out_end = 1 - free_end
            nxt = partner.get((i, out_end))
            if nxt is None or nxt[0] in seen:
                break
            i, free_end = nxt
        paths.append(np.vstack(seq))
    return paths


def _white_decompositions(pieces, want, max_alts=40, max_cands=24, max_results=3000):
    """The `max_alts` lowest-cost ways to join `pieces` down to exactly `want` lines."""
    if want is None or len(pieces) <= want:
        return []
    k = len(pieces) - want
    cands = _join_candidates(pieces)[:max_cands]
    results = []

    def root(par, x):
        while par[x] != x:
            x = par[x]
        return x

    def dfs(start, chosen, used, par, total):
        if len(results) >= max_results:
            return
        if len(chosen) == k:
            results.append((total, list(chosen)))
            return
        for idx in range(start, len(cands)):
            s, i, ei, j, ej = cands[idx]
            if (i, ei) in used or (j, ej) in used:
                continue
            ri, rj = root(par, i), root(par, j)
            if ri == rj:
                continue                                   # would close a loop
            par2 = dict(par)
            par2[ri] = rj
            dfs(idx + 1, chosen + [cands[idx]], used | {(i, ei), (j, ej)}, par2, total + s)

    dfs(0, [], frozenset(), {i: i for i in range(len(pieces))}, 0.0)
    results.sort(key=lambda t: t[0])
    return [(total, _assemble(pieces, joins)) for total, joins in results[:max_alts]]


def route_alternatives(image, expected, max_alts=40):
    """
    Several complete readings of one route chart, as [(geometry_cost, routes, info), ...], lowest
    geometry cost first. The first is always detect_routes()' own reading (cost 0 relative to the
    others' best), so picking alternatives can only ever add options, never lose the default.
    """
    ctx = _prepare(image, expected)
    default = ctx["reconcile"](ctx["white_pieces"], ctx["receptions"])
    alts = _white_decompositions(ctx["white_pieces"], ctx["receptions"], max_alts=max_alts)
    base = alts[0][0] if alts else 0.0
    out = [(0.0,) + _finalize(ctx, default)]
    sigs = set()
    for total, white in alts:
        sig = tuple(sorted(tuple(np.round(w[[0, -1]].ravel()).astype(int)) for w in white))
        if sig in sigs:
            continue
        sigs.add(sig)
        routes, info = _finalize(ctx, white)
        out.append((total - base,) + (routes, info))
    return out
