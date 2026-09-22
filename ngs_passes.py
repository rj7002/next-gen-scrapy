"""
Extract pass locations from a current-format Next Gen Stats pass chart image.

Each target is drawn as a flat ring on the field:  green = complete, blue = touchdown,
red = interception, white = incomplete. Touchdowns also get a blue ball-flight arc that we ignore.
Ring centres are located in image space, then mapped to field coordinates with the chart's own calibration (ngs_calib).
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans

import ngs_calib as K

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


def _td_candidates(mask, lay, min_area):
    """Blobs of a blue-core mask that could be a touchdown ring, best-looking first."""
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 5)))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(closed, connectivity=8)
    out = []
    for i in range(1, n):
        a, w, h = stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = cent[i]
        if a < min_area or w > 60:
            continue
        # a ring core is compact and no wider than the ring itself; arc pieces are long and thin
        if w > 1.3 * _disc_w(lay, cy, cx) or h > 1.6 * _disc_h(lay, cy, cx):
            continue
        out.append(dict(cx=float(cx), cy=float(cy), area=int(a), fill=a / float(max(w * h, 1))))
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
        pool = _td_candidates(loose, lay, min_area=12)
        # keep the strict hits, then top up from the looser pool (ignoring ones already taken)
        taken = list(cands)
        for c in pool:
            if len(taken) >= want:
                break
            if all((c["cx"] - t["cx"]) ** 2 + (c["cy"] - t["cy"]) ** 2 > 400 for t in taken):
                taken.append(c)
        cands = taken[:want] if len(taken) >= want else taken
    elif want is not None:
        cands = cands[:want]
    return [dict(pass_type="TOUCHDOWN", cx=c["cx"], cy=c["cy"]) for c in cands]


def detect_passes(image_path, expected=None):
    """
    Returns (image, [dict(pass_type, cx, cy)]) with pixel centres of every ring found.
    `expected` optionally maps pass_type -> count (from the chart metadata); it is used to
    decide how many rings a merged blob holds and to drop the least ring-like extras.
    """
    im = cv2.imread(image_path)
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
