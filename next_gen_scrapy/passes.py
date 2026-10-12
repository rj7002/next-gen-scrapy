"""
Extract pass locations from a current-format Next Gen Stats pass chart image.

Each target is drawn as a flat ring on the field:  green = complete, blue = touchdown,
red = interception, white = incomplete. Touchdowns also get a blue ball-flight arc that we ignore.
Ring centres are located in image space, then mapped to field coordinates with the chart's own calibration (calib).
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans

from . import calib as K

# colour classes: name -> function(hsv) -> boolean mask
def _masks(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return {
        # hue up to ~90: a ring seen through a touchdown arc is tinted teal
        "COMPLETE": (h >= 38) & (h <= 90) & (s > 110) & (v > 60),
        "INTERCEPTION": ((h <= 6) | (h >= 172)) & (s > 150) & (v > 60),
        # only the bright core of a touchdown ring: the blue flight arc is dimmer (V < ~190)
        "TOUCHDOWN": (h >= 100) & (h <= 116) & (s > 200) & (v > 190),
        "INCOMPLETE": (s < 40) & (v > 200),
    }


def _fill_holes(mask):
    """Fill the hollow centre of a ring so its centroid is the centre of the ellipse."""
    m = mask.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    flood = m.copy()
    ff = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    return m | cv2.bitwise_not(flood)


# A ring is a fixed-size disc on the field (~1.4 yd wide), so its size in pixels follows the local
# scale of the field->image homography. K_* were fit as medians over ~900 isolated rings.
K_W, K_H, K_AREA = 1.46, 1.43, 1.64


def _disc_area(lay, row, cx=600.0):
    sx, sy = lay.scale(cx, row)
    return K_AREA * sx * sy


def _disc_w(lay, row, cx=600.0):
    return K_W * lay.scale(cx, row)[0]


def _disc_h(lay, row, cx=600.0):
    return K_H * lay.scale(cx, row)[1]


def _is_disc_like(lay, c):
    """Reject line-of-scrimmage bar pieces, arc fragments and other non-ring blobs."""
    row = c["cy"]
    if c["w"] > 2.6 * _disc_w(lay, row, c["cx"]) and c["h"] < 1.2 * _disc_h(lay, row, c["cx"]):
        return False                      # long thin bar (LOS)
    if c["w"] < 0.5 * _disc_w(lay, row, c["cx"]) or c["h"] < 0.45 * _disc_h(lay, row, c["cx"]):
        return False                      # fragment
    return True


def _split(mask_filled, lab, i, k):
    ys, xs = np.nonzero(lab == i)
    pts = np.column_stack([xs, ys]).astype(float)
    km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(pts)
    return [tuple(c) for c in km.cluster_centers_]


def _td_candidates(mask, lay, min_area, split_pairs=False):
    """Blobs of a blue-core mask that could be a touchdown ring, best-looking first. With split_pairs, a compact
    blob about two rings big - two touchdown rings drawn over each other - gives both centres."""
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 5)))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(closed, connectivity=8)
    out = []
    for i in range(1, n):
        a, w, h = stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = cent[i]
        if a < min_area:
            continue
        dw, dh = _disc_w(lay, cy, cx), _disc_h(lay, cy, cx)
        # a ring core is compact and no wider than the ring itself; arc pieces are long and thin
        if w <= 1.3 * dw and h <= 1.6 * dh and w <= 60:
            out.append(dict(cx=float(cx), cy=float(cy), area=int(a), fill=a / float(max(w * h, 1))))
        elif split_pairs and w <= 2.3 * dw and h <= 2.6 * dh and a / float(w * h) > 0.45 \
                and 1.5 <= a / _disc_area(lay, cy, cx) <= 2.6:
            ys, xs = np.nonzero(lab == i)
            km = KMeans(n_clusters=2, n_init=5, random_state=0).fit(np.column_stack([xs, ys]).astype(float))
            for (x, y) in km.cluster_centers_:
                out.append(dict(cx=float(x), cy=float(y), area=int(a) // 2, fill=0.5))
    out.sort(key=lambda c: (-c["area"], -c["fill"]))
    return out


def _touchdown_rings(strict, hsv, allowed, lay, want):
    """
    Touchdown ring centres. `want` (from the chart metadata) is exact, so if the bright cores alone do
    not yield that many, look again with a mask that tolerates a ring dimmed by the arc drawn over it.
    """
    cands = _td_candidates(strict, lay, min_area=40)
    if want is not None and len(cands) != want:
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        loose = (((h >= 98) & (h <= 120) & (s > 170) & (v > 150)) | strict) & allowed
        pool = _td_candidates(loose, lay, min_area=12, split_pairs=True)
        # keep the strict hits, then top up from the looser pool (ignoring ones already taken) - with spots that
        # really are a ring: fully saturated blue (arcs, and hash marks seen through them, are paler), off the LOS
        # bar, and near the line - where arcs cross each other and the yard markings - ring-shaped as well
        sb = cv2.blur(hsv[..., 1].astype(np.float32), (3, 3))
        vb = cv2.blur(hsv[..., 2].astype(np.float32), (3, 3))
        bar = cv2.dilate(lay.los_bar_mask(hsv.shape[:2]).astype(np.uint8), np.ones((9, 3), np.uint8)).astype(bool)
        shaped, ends = None, None

        def at_arc_end(x, y):
            # 2018-19 charts draw touchdowns as solid discs, less saturated than later rings - like an arc. But a
            # touchdown marker always sits at the far end of its arc (arcs cross each other mid-way, and meet at the
            # QB's end, behind the line)
            nonlocal ends
            if ends is None:
                ends = [e for e in _arc_ends(hsv, lay)[0] if lay.img_to_field([e])[0][1] > -2]
            for ex, ey in ends:
                if (x - ex) ** 2 + (y - ey) ** 2 < (1.5 * _disc_w(lay, ey, ex)) ** 2:
                    c = _refine_td(hsv, lay, ex, ey)
                    if c is not None and (x - c[0]) ** 2 + (y - c[1]) ** 2 < (0.8 * _disc_w(lay, ey, ex)) ** 2:
                        return True
            return False

        def real(c):
            nonlocal shaped
            x, y = c["cx"], c["cy"]
            r = max(2, int(0.4 * _disc_w(lay, y, x)))
            y0, y1, x0, x1 = max(0, int(y) - r), int(y) + r + 1, max(0, int(x) - r), int(x) + r + 1
            if bar[int(y), int(x)]:
                return False
            if sb[y0:y1, x0:x1].max() < 251 or vb[y0:y1, x0:x1].max() < 196:
                return at_arc_end(x, y)
            if lay.img_to_field([[x, y]])[0][1] >= 15:
                return True
            if shaped is None:
                shaped = [(a, b) for sc, a, b in _ring_candidates(hsv, lay, "TOUCHDOWN", []) if sc > 5] + \
                         [(a, b) for sc, a, b in _td_under_arcs(hsv, lay, []) if sc >= 30]
            return any((x - a) ** 2 + (y - b) ** 2 < (0.6 * _disc_w(lay, b, a)) ** 2 for a, b in shaped)

        taken = list(cands)
        for c in pool:
            if len(taken) >= want:
                break
            if all((c["cx"] - t["cx"]) ** 2 + (c["cy"] - t["cy"]) ** 2 > 400 for t in taken) and real(c):
                taken.append(c)
        cands = taken[:want] if len(taken) >= want else taken
    elif want is not None:
        cands = cands[:want]
    return [dict(pass_type="TOUCHDOWN", cx=c["cx"], cy=c["cy"]) for c in cands]


# ---------------------------------------------------------------------------------------------------------------
# Recovery: the stat box says exactly how many completions / touchdowns / interceptions are drawn, so when the pass
# above comes up short the rest are on the chart somewhere it doesn't look or can't separate: on the LOS bar (a
# touchdown ring there is the bar's own blue), on a sideline or under a yard label, in the logo corner, or under a
# touchdown arc. Two ways to find them:
#  - by shape: a ring is a bright centre, a dark gap (~0.27 of its width out) and a bright band (~0.4) all the way
#    round. Arcs, bars and labels have no such gap. Searched over the whole field, masks ignored.
#  - touchdowns, by their arc: every touchdown ring sits at the far end of its blue ball-flight arc. An arc that
#    runs off the top of the chart ends beyond it - that ring isn't drawn, and nothing is added for it.
_ANG = np.linspace(0, 2 * np.pi, 24, endpoint=False)
_MIN_SCORE = {"COMPLETE": 30.0, "INTERCEPTION": 8.0, "TOUCHDOWN": 30.0}


def _loose_mask(hsv, ptype):
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    return {"COMPLETE": (h >= 35) & (h <= 95) & (s > 70) & (v > 70),
            "INTERCEPTION": ((h <= 10) | (h >= 160)) & (s > 90) & (v > 70),
            "TOUCHDOWN": (h >= 95) & (h <= 125) & (s > 120) & (v > 80)}[ptype]


def _ring_candidates(hsv, lay, ptype, taken, step=2):
    """Ring-shaped spots of a colour anywhere on the field, best first: [(score, x, y)], none within ~0.6 ring
    widths of each other or of `taken`."""
    cm = _loose_mask(hsv, ptype)
    cm[K.FIELD_ROWS:] = False
    ys, xs = np.nonzero(cm[::step, ::step])
    if not len(xs):
        return []
    xs, ys = (xs * step).astype(float), (ys * step).astype(float)
    V = hsv[..., 2].astype(float)
    C = cm.astype(float)
    H, W = V.shape
    rows = np.unique(ys)
    scale = {y: lay.scale(600.0, y) for y in rows}
    w = K_W * np.array([scale[y][0] for y in ys])
    h = K_H * np.array([scale[y][1] for y in ys])

    def ring(r, img):
        px = np.clip((xs[:, None] + r * w[:, None] * np.cos(_ANG)).round().astype(int), 0, W - 1)
        py = np.clip((ys[:, None] + r * h[:, None] * np.sin(_ANG)).round().astype(int), 0, H - 1)
        return img[py, px]

    centre = np.mean([ring(r, V).mean(1) for r in (0.0, 0.06, 0.11)], 0)
    gap = np.percentile(np.concatenate([ring(r, V) for r in (0.26, 0.29)], 1), 75, axis=1)
    band = np.percentile(np.concatenate([ring(r, V) for r in (0.37, 0.41)], 1), 25, axis=1)
    coloured = (np.mean([ring(r, C).mean(1) for r in (0.0, 0.08)], 0) > 0.6) & \
               (np.mean([ring(r, C).mean(1) for r in (0.37, 0.41)], 0) > 0.45)
    score = np.minimum(centre - gap, band - gap)
    keep = coloured & (score > 0)
    out = []
    for i in np.argsort(-score[keep]):
        x, y, sc = xs[keep][i], ys[keep][i], score[keep][i]
        r2 = (0.6 * _disc_w(lay, y, x)) ** 2
        if any((x - a) ** 2 + (y - b) ** 2 < r2 for _, a, b in out) or any((x - a) ** 2 + (y - b) ** 2 < r2 for a, b in taken):
            continue
        out.append((float(sc), float(x), float(y)))
    return out


def _td_under_arcs(hsv, lay, taken, step=2):
    """Touchdown rings with arcs crossing over them: where two see-through arcs overlap they're as bright as a ring
    and merge with it, and the all-round gap test fails along the arcs. Here the gap only has to be dark for most of
    the way round - but the ring must be the deep, bright blue of a touchdown ring (a white or green ring, a yard
    label or the hash marks seen through an arc are paler). On the LOS bar - the same blue - a ring must also cut the
    bar with its dark gap, left and right of its centre; a bar with an arc over it doesn't. Best first:
    [(score, x, y)]."""
    cm = _loose_mask(hsv, "TOUCHDOWN")
    cm[K.FIELD_ROWS:] = False
    bar = cv2.dilate(lay.los_bar_mask(hsv.shape[:2]).astype(np.uint8), np.ones((9, 3), np.uint8)).astype(bool)
    ys, xs = np.nonzero(cm[::step, ::step])
    if not len(xs):
        return []
    xs, ys = (xs * step).astype(float), (ys * step).astype(float)
    V, S = hsv[..., 2].astype(float), hsv[..., 1].astype(float)
    H, W = V.shape
    scale = {y: lay.scale(600.0, y) for y in np.unique(ys)}
    w = K_W * np.array([scale[y][0] for y in ys])
    h = K_H * np.array([scale[y][1] for y in ys])

    def ring(r, img):
        px = np.clip((xs[:, None] + r * w[:, None] * np.cos(_ANG)).round().astype(int), 0, W - 1)
        py = np.clip((ys[:, None] + r * h[:, None] * np.sin(_ANG)).round().astype(int), 0, H - 1)
        return img[py, px]

    centre = np.mean([ring(r, V).mean(1) for r in (0.0, 0.06)], 0)
    gap = np.median(np.concatenate([ring(r, V) for r in (0.26, 0.29)], 1), 1)
    band = np.median(np.concatenate([ring(r, V) for r in (0.37, 0.41)], 1), 1)
    sat_c = np.mean([ring(r, S).mean(1) for r in (0.0, 0.06)], 0)
    sat_b = np.median(np.concatenate([ring(r, S) for r in (0.37, 0.41)], 1), 1)
    score = np.minimum(centre - gap, band - gap)
    side = np.abs(np.sin(_ANG)) < 0.35                     # gap points level with the centre: on the bar, if it's there
    gap_side = np.median(np.concatenate([ring(r, V)[:, side] for r in (0.26, 0.29)], 1), 1)
    on_bar = bar[ys.astype(int), xs.astype(int)]
    score = np.where(on_bar, np.minimum(score, centre - gap_side - 60), score)
    ok = (centre >= 200) & (sat_c >= 180) & (sat_b >= 160) & (score > 0)
    out = []
    for i in np.argsort(-np.where(ok, score, -1)):
        if not ok[i]:
            break
        x, y = xs[i], ys[i]
        r2 = (0.9 * _disc_w(lay, y, x)) ** 2
        if any((x - a) ** 2 + (y - b) ** 2 < r2 for _, a, b in out) or any((x - a) ** 2 + (y - b) ** 2 < r2 for a, b in taken):
            continue
        out.append((float(score[i]), float(x), float(y)))
    return out


_PEEK_ANG = np.linspace(0, 2 * np.pi, 36, endpoint=False)


def _white_peeks(hsv, lay, found):
    """White rings mostly under another ring (drawn first, then a completion / TD / INT on top): their centre is
    covered, so the shape test can't see them. Around each found coloured ring, look for a centre whose white band
    (0.40 of a ring width) is white wherever that ring doesn't cover it - at least 30% of the way round - and whose
    gap is dark there. Checked on 2024: every hit is a real ring. [(x, y)]"""
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    white = (s < 50) & (v > 185)
    H, W = white.shape
    whites = [(f["cx"], f["cy"]) for f in found if f["pass_type"] == "INCOMPLETE"]
    cos, sin = np.cos(_PEEK_ANG), np.sin(_PEEK_ANG)
    out = []
    for f in found:
        if f["pass_type"] == "INCOMPLETE":
            continue
        fx, fy = f["cx"], f["cy"]
        dw, dh = _disc_w(lay, fy, fx), _disc_h(lay, fy, fx)
        best = None
        for oy in np.arange(-1.0, 1.01, 0.1):
            for ox in np.arange(-1.0, 1.01, 0.1):
                d2 = ox ** 2 + oy ** 2
                if d2 < 0.15 ** 2 or d2 > 1.0:
                    continue
                x, y = fx + ox * dw, fy + oy * dh
                if any((x - a) ** 2 + (y - b) ** 2 < (0.6 * dw) ** 2 for a, b in whites + [p[1:3] for p in out]):
                    continue
                bx = np.clip((x + 0.40 * dw * cos).round().astype(int), 0, W - 1)
                by = np.clip((y + 0.40 * dh * sin).round().astype(int), 0, H - 1)
                gx = np.clip((x + 0.27 * dw * cos).round().astype(int), 0, W - 1)
                gy = np.clip((y + 0.27 * dh * sin).round().astype(int), 0, H - 1)
                cov = lambda px, py: ((px - fx) / (0.52 * dw)) ** 2 + ((py - fy) / (0.52 * dh)) ** 2 <= 1
                vis = ~cov(bx, by)
                if vis.sum() < 8 or vis.mean() < 0.3:
                    continue
                frac = white[by, bx][vis].mean()
                gap_ok = (~white[gy, gx] | cov(gx, gy)).mean()
                if frac >= 0.75 and gap_ok >= 0.9:
                    sc = frac * vis.mean()
                    if best is None or sc > best[0]:
                        best = (sc, float(x), float(y))
        if best is not None:
            out.append(best)
    return [(x, y) for _, x, y in out]


def _small_whites(hsv, lay, allowed, taken):
    """Small white rings the colour pass discards as fragments: on 2018-19 charts rings are drawn smaller than the size
    model, and far downfield only their white centre (a few px) survives the colour test. On the field nothing else is
    a compact white blob (yard labels and the LOS label are masked; hash marks are grey lines). Pieces closer than a
    ring width are one ring. [(x, y)]"""
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    raw = ((s < 45) & (v > 190)).astype(np.uint8)
    raw[K.FIELD_ROWS:] = 0
    raw[:80, :250] = 0                                     # logo
    labels = ~allowed                                       # yard numbers / LOS label beside the sidelines
    m = raw
    n, lab, st, cent = cv2.connectedComponentsWithStats(_fill_holes(m.astype(bool)), connectivity=8)
    out = []
    for i in np.argsort(-st[:, cv2.CC_STAT_AREA])[:-1] if n > 1 else []:
        if i == 0:
            continue
        a, w, hh = st[i, cv2.CC_STAT_AREA], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        x, y = cent[i]
        dw, dh = _disc_w(lay, y, x), _disc_h(lay, y, x)
        if a < 12 or w < 0.3 * dw or hh < 0.25 * dh or w > 1.4 * dw or hh > 1.6 * dh or a / float(w * hh) < 0.5:
            continue
        x0, y0 = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
        if labels[y0:y0 + hh, x0:x0 + w].any():
            # beside a yard number: only a real ring - a white centre inside a dark gap inside a white band (a "0"
            # has the band and the hole, not the centre), and wider than tall like every ring on the tilted field
            sub = raw[y0:y0 + hh, x0:x0 + w] & (lab[y0:y0 + hh, x0:x0 + w] == i)
            k, kl, kst, kc = cv2.connectedComponentsWithStats(sub.astype(np.uint8), connectivity=4)

            def inside(j, o):          # component j's box strictly within component o's
                return (kst[j, 0] > kst[o, 0] and kst[j, 1] > kst[o, 1]
                        and kst[j, 0] + kst[j, 2] < kst[o, 0] + kst[o, 2] and kst[j, 1] + kst[j, 3] < kst[o, 1] + kst[o, 3])

            ring_like = any(kst[j, cv2.CC_STAT_AREA] >= 3 and abs(kc[j][0] - (x - x0)) < 0.2 * w
                            and abs(kc[j][1] - (y - y0)) < 0.25 * hh and inside(j, o)
                            and kst[o, cv2.CC_STAT_WIDTH] >= 0.8 * w
                            for j in range(1, k) for o in range(1, k) if o != j)
            los = float(lay.yard_to_row(0.0))
            beside_los = y0 <= los + 6 and y0 + hh >= los - 30           # the "LOS" labels
            flat = (not beside_los and w >= 1.6 * hh and hh >= 0.45 * dh and w <= 1.3 * dw
                    and a / float(w * hh) >= 0.6)
            if not ((w >= 1.2 * hh and ring_like) or flat):
                continue
        if any((x - p) ** 2 + (y - q) ** 2 < (0.9 * dw) ** 2 for p, q in taken + out):
            continue
        out.append((float(x), float(y)))
    return out


def _white_rings(hsv, lay, taken, step=2):
    """
    Incomplete (white) rings the colour pass misses: one seen through a touchdown arc (tinted light blue), on a
    sideline, by a yard label or the LOS label, or half under another ring. Found by shape - bright centre, dark gap,
    bright band - either all the way round or, for one peeking out, in at least 60% of directions. Incompletions have
    no count to aim for (NGS doesn't draw every one), so this must not invent rings: checked on 2024 charts whose white
    rings already matched their targeted incompletions, it adds none. [(x, y)]
    """
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    whitish = (v >= 185) & ((s <= 60) | ((h >= 90) & (h <= 125) & (s <= 150)))
    whitish[K.FIELD_ROWS:] = False
    ys, xs = np.nonzero(whitish[::step, ::step])
    if not len(xs):
        return []
    xs, ys = (xs * step).astype(float), (ys * step).astype(float)
    V, Wm = v.astype(float), whitish.astype(float)
    H, W = V.shape
    scale = {y: lay.scale(600.0, y) for y in np.unique(ys)}
    w = K_W * np.array([scale[y][0] for y in ys])
    hh = K_H * np.array([scale[y][1] for y in ys])

    def ring(r, img, k):
        px = np.clip((xs[:, None] + k * r * w[:, None] * np.cos(_ANG)).round().astype(int), 0, W - 1)
        py = np.clip((ys[:, None] + k * r * hh[:, None] * np.sin(_ANG)).round().astype(int), 0, H - 1)
        return img[py, px]

    # rings are drawn a little smaller on some charts (2018-19, deep downfield) than the size model says: try a few
    # sizes and keep each spot's best
    n = len(_ANG)
    best_full = np.full(len(xs), -np.inf)
    best_part = np.zeros(len(xs))
    best_keep = np.zeros(len(xs), bool)
    # two ring styles: the current one (gap ~0.27 of the width out, band ~0.39) and 2018-19's (a bigger white centre,
    # a thin gap ~0.33 out, band ~0.46)
    for k, (g_r, b_r) in [(k, prof) for k in (1.0, 0.8, 0.65) for prof in (((0.26, 0.29), (0.37, 0.41)), ((0.32, 0.35), (0.45, 0.48)))]:
        centre = np.mean([ring(r, V, k).mean(1) for r in (0.0, 0.06, 0.11)], 0)
        gap = np.concatenate([ring(r, V, k) for r in g_r], 1)
        band = np.concatenate([ring(r, V, k) for r in b_r], 1)
        band_w = np.concatenate([ring(r, Wm, k) for r in b_r], 1)
        full = np.minimum(centre - np.percentile(gap, 75, 1), np.percentile(band, 25, 1) - np.percentile(gap, 75, 1))
        part = ((band[:, :n] - gap[:, :n] > 60) & (band_w[:, :n] > 0.5)).mean(1)
        centre_w = np.mean([ring(r, Wm, k).mean(1) for r in (0.0, 0.06)], 0)
        keep = (centre_w >= 0.6) & ((full >= 20) | (part >= 0.6))
        better = keep & (full + 100 * part > np.where(best_keep, best_full + 100 * best_part, -np.inf))
        best_full = np.where(better, full, best_full)
        best_part = np.where(better, part, best_part)
        best_keep |= keep
    full, part, keep = best_full, best_part, best_keep
    out = []
    for i in np.argsort(-(full + 100 * part)):
        if not keep[i]:
            continue
        x, y = xs[i], ys[i]
        r2 = (0.6 * _disc_w(lay, y, x)) ** 2
        if any((x - a) ** 2 + (y - b) ** 2 < r2 for a, b in out) or any((x - a) ** 2 + (y - b) ** 2 < r2 for a, b in taken):
            continue
        out.append((float(x), float(y)))
    return out


def _arc_mask(hsv, lay):
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    m = (h >= 95) & (h <= 125) & (s > 80) & (v > 55)
    m[K.FIELD_ROWS:] = False
    m &= ~lay.los_bar_mask(hsv.shape[:2])
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool)


def _arc_discs(hsv, lay, claimed):
    """The marker on each arc piece downfield that has no touchdown ring found on it: the most saturated, brightest
    compact spot on the arc. On 2018-19 charts a deep throw's arc is a narrow hairpin (up and back down over itself)
    with a small solid disc at the bottom of its down leg - no skeleton end and no ring shape to find. [(x, y)]"""
    m = _arc_mask(hsv, lay)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    core = ((h >= 95) & (h <= 125) & (s >= 232) & (v >= 165) & ~lay.los_bar_mask(hsv.shape[:2])).astype(np.uint8)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 150 or st[i, cv2.CC_STAT_TOP] < 20:
            continue                                   # a fragment, or an arc leaving the top of the chart
        comp = cv2.dilate((lab == i).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        ys, xs = np.nonzero(comp)
        if any(((xs - a) ** 2 + (ys - b) ** 2).min() < (1.5 * _disc_w(lay, b, a)) ** 2 for a, b in claimed):
            continue                                   # its ring is found
        k, kl, kst, kc = cv2.connectedComponentsWithStats(core & comp.astype(np.uint8), connectivity=8)
        best = None
        for j in range(1, k):
            x, y = kc[j]
            if lay.img_to_field([[x, y]])[0][1] <= -2:
                continue                                   # the QB's end
            a = kst[j, cv2.CC_STAT_AREA] / _disc_area(lay, y, x)
            if a >= 0.15 and (best is None or a > best[0]):
                best = (a, float(x), float(y))
        if best is not None:
            out.append(best[1:])
    return out


def _arc_ends(hsv, lay):
    """(far ends of the touchdown arcs [(x, y)], number of arcs that run off the top of the chart)."""
    from skimage.morphology import skeletonize
    m = _arc_mask(hsv, lay)
    sk = skeletonize(m)
    nb = cv2.filter2D(sk.astype(np.float32), -1, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32),
                      borderType=cv2.BORDER_CONSTANT)
    junction = sk & (nb >= 3)
    ends, off_top = [], 0
    for y, x in zip(*np.nonzero(sk & (nb == 1))):
        # an endpoint at the end of a real branch (not a spur): walk 25 px back along the skeleton
        px, py, prev, n = x, y, None, 0
        while n < 25:
            nxt = next(((px + dx, py + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                        if (dx or dy) and 0 <= py + dy < sk.shape[0] and 0 <= px + dx < sk.shape[1]
                        and sk[py + dy, px + dx] and (px + dx, py + dy) != prev), None)
            if nxt is None or junction[nxt[1], nxt[0]]:
                break
            prev, (px, py), n = (px, py), nxt, n + 1
        if n < 24:
            continue
        if y < 6:
            off_top += 1                                  # runs off the top: its ring is beyond the chart
        else:                                             # (pieces cut off by the LOS bar and the QB's end of the arc
            ends.append((float(x), float(y)))             #  included: _refine_td finds no ring there)
    return ends, off_top


def _refine_td(hsv, lay, x, y):
    c = _refine_td_once(hsv, lay, x, y, s_min=170, v_min=120)
    if c is None:
        # a 2018-19 disc is joined to its arc in that mask; its own pixels are the most saturated and bright
        c = _refine_td_once(hsv, lay, x, y, s_min=232, v_min=165, small=True)
    return c


def _refine_td_once(hsv, lay, x, y, s_min, v_min, small=False):
    """Centre of the touchdown ring at an arc's end - the strongly saturated blue near it (the arc is see-through
    and pale) - or None when there is no ring there (e.g. the narrow top of an arc, which also skeletonises to an
    end)."""
    r = 1.6 * _disc_w(lay, y, x)
    x0, x1, y0, y1 = int(max(0, x - r)), int(x + r) + 1, int(max(0, y - r)), int(min(K.FIELD_ROWS, y + r)) + 1
    h, s, v = (hsv[y0:y1, x0:x1, i].astype(int) for i in range(3))
    core = ((h >= 95) & (h <= 125) & (s > s_min) & (v > v_min)).astype(np.uint8)
    core[lay.los_bar_mask(hsv.shape[:2])[y0:y1, x0:x1]] = 0          # the bar is the same blue
    n, lab, st, cent = cv2.connectedComponentsWithStats(core, connectivity=8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    a, bw, bh = st[i, cv2.CC_STAT_AREA], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
    dw, dh = _disc_w(lay, y, x), _disc_h(lay, y, x)
    area, rw, rh = a / _disc_area(lay, y, x), bw / dw, bh / dh
    # a ring (or a 2018-style solid disc, which merges with the arc's last stretch) is about ring-sized and round;
    # what else ends an arc - its narrow top, a piece cut off by the LOS bar - is long and thin or flat
    lo = 0.25 if small else 0.7
    if not (lo <= area <= 3.2 and rw <= 2.0 and rh <= 2.4 and 0.45 <= rh / rw <= 1.35 and a / float(bw * bh) >= 0.5):
        return None
    # ...and has a ring's bright core (v ~190+, a 2018 disc ~170+); an arc on its own is dimmer
    vv = cv2.blur(v.astype(np.float32), (3, 3))
    if vv[lab == i].max() < 172:
        return None
    # centre: the brightest ring-sized spot of the blob - a ring's core (or a 2018 disc) is brighter than its arc
    blob = (lab == i).astype(np.float32) * np.clip(v.astype(np.float32) - 120.0, 0, None)
    k = np.zeros((max(3, int(dh) | 1), max(3, int(dw) | 1)), np.float32)
    cv2.ellipse(k, (k.shape[1] // 2, k.shape[0] // 2), (k.shape[1] // 2, k.shape[0] // 2), 0, 0, 360, 1.0, -1)
    resp = cv2.filter2D(blob, -1, k, borderType=cv2.BORDER_CONSTANT)
    gy, gx = np.mgrid[0:resp.shape[0], 0:resp.shape[1]]
    resp[((gx + x0 - x) / dw) ** 2 + ((gy + y0 - y) / dh) ** 2 > 0.9 ** 2] = -1   # the ring is at the arc's end
    yy, xx = np.unravel_index(int(np.argmax(resp)), resp.shape)
    return float(x0 + xx), float(y0 + yy)


def _colour_blobs(hsv, lay, taken, ptype="INTERCEPTION"):
    """Rings by colour alone: on the field, red is only ever an interception ring and green a completion (the logo
    aside). Catches the ones the ring shape can't: tiny far ones, ones on a sideline or under a yard label, one tinted
    by an arc drawn over it, one cut by the bottom of the field. Biggest first: [(x, y)]."""
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    if ptype == "INTERCEPTION":
        m = (((h <= 10) | (h >= 160)) & (s > 90) & (v > 70)).astype(np.uint8)
        m[:70, :70] = 0                         # NFL shield
    else:
        m = ((h >= 35) & (h <= 95) & (s > 90) & (v > 70)).astype(np.uint8)
        m[:80, :250] = 0                        # the green NEXT GEN STATS lettering
    m[K.FIELD_ROWS + 25:] = 0                   # the legend's own rings sit well below the field
    m = _fill_holes(m.astype(bool))
    n, lab, st, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y = cent[i]
        a = st[i, cv2.CC_STAT_AREA]
        ratio = a / _disc_area(lay, min(y, K.FIELD_ROWS - 1), x)
        if ratio < 0.15 or ratio > 3.0:
            continue
        r2 = (0.6 * _disc_w(lay, y, x)) ** 2
        if any((x - a_) ** 2 + (y - b) ** 2 < r2 for a_, b in taken):
            continue
        out.append((ratio, float(x), float(y)))
    out.sort(key=lambda t: -min(t[0], 1.0))
    return [(x, y) for _, x, y in out]


def _recover(hsv, lay, ptype, need, taken, others=()):
    """Up to `need` rings of `ptype` the main pass missed (see above)."""
    got, how = [], []
    cands = _ring_candidates(hsv, lay, ptype, taken) if ptype != "INTERCEPTION" else []
    if ptype == "INTERCEPTION":
        got = _colour_blobs(hsv, lay, taken, ptype)[:need]
        how = ["colour"] * len(got)
    for sc, x, y in cands:
        if len(got) >= need or sc < _MIN_SCORE[ptype]:
            break
        got.append((x, y))
        how.append(f"ring:{sc:.0f}")
    if ptype == "TOUCHDOWN" and len(got) < need:
        ends, _off = _arc_ends(hsv, lay)
        for x, y in ends:
            if len(got) >= need:
                break
            r2 = (2.0 * _disc_w(lay, y, x)) ** 2              # (an arc stops a little short of its own ring)
            if any((x - a) ** 2 + (y - b) ** 2 < r2 for a, b in taken + got):
                continue                                   # this arc's ring is already found
            c = _refine_td(hsv, lay, x, y)
            if c is not None:
                # snap to a ring shape close by, if there is one (a ring on the LOS bar has its gap; the bar doesn't)
                near = [(sc, a, b) for sc, a, b in cands if sc > 5 and (a - c[0]) ** 2 + (b - c[1]) ** 2 < (1.5 * _disc_w(lay, b, a)) ** 2]
                if near:
                    c = max(near)[1:]
            if c is not None and not any((c[0] - a) ** 2 + (c[1] - b) ** 2 < r2 / 4 for a, b in taken + got):
                got.append(c)
                how.append("arc")
    if ptype == "TOUCHDOWN" and len(got) < need:
        for sc, x, y in _td_under_arcs(hsv, lay, taken + got + list(others)):
            if len(got) >= need or sc < 30:
                break
            got.append((x, y))
            how.append(f"under-arc:{sc:.0f}")
    if ptype == "TOUCHDOWN" and len(got) < need:
        # last: an arc's far end is its catch point, even where the marker there can't be made out (a tiny one at
        # the top of the chart, one on the LOS bar, one under another arc). Downfield ends only (the QB's end is
        # behind the line), none already claimed, and not the two cut ends of an arc that crosses the LOS bar
        ends, _off = _arc_ends(hsv, lay)
        los = float(lay.yard_to_row(0.0))
        sb = cv2.blur(hsv[..., 1].astype(np.float32), (3, 3))
        pool = []
        for x, y in ends:
            if y < 20 or lay.img_to_field([[x, y]])[0][1] <= -2:
                continue                                   # (at the top edge an arc is leaving the chart)
            w = _disc_w(lay, y, x)
            if any((x - a) ** 2 + (y - b) ** 2 < (3.0 * w) ** 2 for a, b in taken + got) or \
                    any((x - a) ** 2 + (y - b) ** 2 < (2.0 * w) ** 2 for a, b in others):
                continue                                   # the end of an arc whose ring is already found
            if abs(y - los) < 45:
                continue                                   # at the bar an arc is cut in two: its ends there aren't tips
            if any((x - a) ** 2 + (y - b) ** 2 < (1.5 * w) ** 2 for a, b in others):
                continue                                   # nor where it passes under another ring
            r = max(2, int(0.6 * w))
            pool.append((float(sb[max(0, int(y) - r):int(y) + r + 1, max(0, int(x) - r):int(x) + r + 1].max()), x, y))
        for _sat, x, y in sorted(pool, reverse=True)[:need - len(got)]:
            got.append((x, y))
            how.append("arc-end")
        if len(got) < need:
            for x, y in _arc_discs(hsv, lay, taken + got):
                if len(got) >= need:
                    break
                if any((x - a) ** 2 + (y - b) ** 2 < (2.0 * _disc_w(lay, y, x)) ** 2 for a, b in others):
                    continue
                got.append((x, y))
                how.append("arc-disc")
    if ptype == "COMPLETE" and len(got) < need:
        for c in _colour_blobs(hsv, lay, taken + got, ptype)[:need - len(got)]:
            got.append(c)
            how.append("colour")
    return [dict(pass_type=ptype, cx=x, cy=y, recovered=h) for (x, y), h in zip(got, how)]


def detect_passes(image, expected=None):
    """
    `image` is a file path or an already-decoded BGR array (e.g. from cv2.imdecode).
    Returns (image, [dict(pass_type, cx, cy)]) with pixel centres of every ring found.
    `expected` optionally maps pass_type -> count (from the chart metadata); it is used to
    decide how many rings a merged blob holds and to drop the least ring-like extras.
    """
    im = K.read_image(image)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    lay = K.calibrate(im)
    allowed = lay.on_card_field(im.shape[:2])
    found = []
    for ptype, mask in _masks(hsv).items():
        mask = mask & allowed
        if ptype == "TOUCHDOWN":
            # A touchdown ring is blue like its own ball-flight arc, so it is found by its bright core
            # rather than by the ring size model. How much of that core survives depends on how much
            # of the ring the arc covers, so rank the candidates and keep as many as the metadata says.
            found += _touchdown_rings(mask & ~lay.los_bar_mask(im.shape[:2]), hsv, allowed, lay,
                                      None if expected is None else expected.get(ptype))
            continue
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
        filled = _fill_holes(mask)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(filled, connectivity=8)
        comps = []
        for i in range(1, n):
            c = dict(i=i, pass_type=ptype, cx=cent[i][0], cy=cent[i][1], area=stats[i, cv2.CC_STAT_AREA],
                     w=stats[i, cv2.CC_STAT_WIDTH], h=stats[i, cv2.CC_STAT_HEIGHT])
            if c["area"] >= 40 and _is_disc_like(lay, c):
                c["ratio"] = c["area"] / _disc_area(lay, c["cy"], c["cx"])
                comps.append(c)

        want = None if expected is None else expected.get(ptype)
        # How many rings does each blob hold? Start from its area, then - when the chart metadata says
        # how many there should be - move rings between blobs before discarding any. The metadata count
        # is exact for completions, touchdowns and interceptions (every one of them is drawn), so the
        # job here is to place that many rings, not to re-decide how many there are.
        for c in comps:
            wide = c["w"] / _disc_w(lay, c["cy"], c["cx"]) > 1.35
            c["k"] = min(4, int(round(c["ratio"]))) if c["ratio"] >= 1.8 else (2 if wide and c["ratio"] > 1.4 else 1)
        if want is not None:
            # crowding: how much area each blob carries per ring it has been given. The most crowded
            # blob is the best candidate to hold one more; the least crowded to give one up.
            crowd = lambda c: c["ratio"] / c["k"]
            while sum(c["k"] for c in comps) < want and comps:
                c = max(comps, key=crowd)
                if crowd(c) < 0.75:        # nothing left that plausibly hides another ring
                    break
                c["k"] += 1
            while sum(c["k"] for c in comps) > want and comps:
                multi = [c for c in comps if c["k"] > 1]
                if multi:
                    min(multi, key=crowd)["k"] -= 1
                else:                      # every blob holds one: drop the least ring-like ones
                    comps.remove(min(comps, key=lambda c: -abs(np.log(max(c["ratio"], 1e-3)))))
        for c in comps:
            if c["k"] == 1:
                found.append(dict(pass_type=ptype, cx=c["cx"], cy=c["cy"]))
            else:
                for (x, y) in _split(filled, lab, c["i"], c["k"]):
                    found.append(dict(pass_type=ptype, cx=x, cy=y))
    # white rings the colour pass can't see (no count for these: NGS doesn't draw every incompletion), capped by the
    # most incompletions the stat box allows
    extra = _white_rings(hsv, lay, [(f["cx"], f["cy"]) for f in found])
    extra += _small_whites(hsv, lay, allowed, [(f["cx"], f["cy"]) for f in found] + extra)
    extra += _white_peeks(hsv, lay, found + [dict(pass_type="INCOMPLETE", cx=x, cy=y) for x, y in extra])
    cap = None if expected is None else expected.get("INCOMPLETE_MAX")
    if cap is not None:
        extra = extra[:max(0, cap - sum(f["pass_type"] == "INCOMPLETE" for f in found))]
    found += [dict(pass_type="INCOMPLETE", cx=x, cy=y, recovered="shape") for x, y in extra]
    if expected is not None:
        for ptype in ("INTERCEPTION", "TOUCHDOWN", "COMPLETE"):
            need = (expected.get(ptype) or 0) - sum(f["pass_type"] == ptype for f in found)
            if need > 0:
                found += _recover(hsv, lay, ptype, need, [(f["cx"], f["cy"]) for f in found if f["pass_type"] == ptype],
                                  [(f["cx"], f["cy"]) for f in found if f["pass_type"] != ptype])
    return im, found, lay


def detect_blue_rings(hsv, lay, want=None):
    """
    Pixel centres (cx, cy) of blue touchdown rings (bright blue cores; excludes the LOS bar).
    `want` is the touchdown count from the chart metadata; given it, a ring whose core is mostly
    covered by another line is still recovered.
    """
    allowed = lay.on_card_field(hsv.shape[:2])
    strict = _masks(hsv)["TOUCHDOWN"] & allowed & ~lay.los_bar_mask(hsv.shape[:2])
    return [(r["cx"], r["cy"]) for r in _touchdown_rings(strict, hsv, allowed, lay, want)]
