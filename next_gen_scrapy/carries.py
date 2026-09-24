"""
Extract full rush paths from a current-format Next Gen Stats *carry chart* image.

Every carry is one coloured line from the handoff point to where the runner was stopped:
red = tackled for loss, yellow = 0-5 yards gained, green = 5+ yards gained / touchdown.
A blue ring marks a touchdown and a red ring a lost fumble. Each carry is returned as an ordered list
of field coordinates (x, y in yards) from its start (in the backfield) to its end.
"""
import cv2
import numpy as np

from . import calib as K
from . import paths as NP
from .passes import detect_blue_rings
from .routes import resample_field

COLORS = ("LOSS", "SHORT", "LONG")          # red, yellow, green


def carry_masks(hsv, lay, rings):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    allowed = lay.on_card_field(hsv.shape[:2])
    ring_area = np.zeros(hsv.shape[:2], np.uint8)
    for cx, cy in rings:
        cv2.circle(ring_area, (int(cx), int(cy)), 22, 1, -1)
    allowed &= ~ring_area.astype(bool)
    red = (((h <= 8) | (h >= 170)) & (s > 150) & (v > 90))
    yellow = (h >= 14) & (h <= 32) & (s > 150) & (v > 120)
    green = (h >= 38) & (h <= 75) & (s > 110) & (v > 60)
    return {"LOSS": red & allowed, "SHORT": yellow & allowed, "LONG": green & allowed}


def detect_red_rings(red_mask, hsv):
    """
    Red discs (lost fumble). Thick overlapping red lines can also survive an opening, so a candidate
    must be a solid ellipse *and* have the white 'FL' label drawn up and to the right of it.
    """
    opened = cv2.morphologyEx(red_mask.astype(np.uint8), cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 11)))
    white = (hsv[..., 1] < 30) & (hsv[..., 2] > 215)
    n, lab, st, ce = cv2.connectedComponentsWithStats(opened, connectivity=8)
    rings = []
    for i in range(1, n):
        area, w, h = st[i, cv2.CC_STAT_AREA], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if area < 80 or area / float(w * h) < 0.66 or w > 45:
            continue
        cx, cy = ce[i]
        box = white[max(0, int(cy - 50)):int(cy - 6), int(cx + 8):int(cx + 90)]
        if box.sum() >= 40:
            rings.append((cx, cy))
    return rings


MAX_HANDOFF_Y = 0.5          # a handoff is always behind the line of scrimmage


def detect_carries(image, expected=None):
    """
    `image` is a file path or an already-decoded BGR array (e.g. from cv2.imdecode).
    `expected` (optional): {"carries": n, "touchdowns": n} from the chart metadata; used to merge
    fragments when more lines than carries are found.
    `expected["max_yards"]` (optional): the player's total rushingYards for the game, from the same
    chart metadata - an extra, independent cap on top of the chart's own calibration (see
    routes.resample_field). Looser than for routes/passes: an individual carry can in principle run
    longer than the game total if other carries lost yardage, but it still catches gross errors.
    Returns (carries, info); carries = list of dict(color, td, fumble, handoff_ok, pts (Nx2 field yd)).
    handoff_ok is False if the line still starts past the line of scrimmage (its start could not be traced).
    """
    im = K.read_image(image)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    ceiling = None if expected is None else expected.get("max_yards")
    lay = K.calibrate(im)
    blue_rings = detect_blue_rings(hsv, lay, None if expected is None else expected.get("touchdowns"))
    masks = carry_masks(hsv, lay, blue_rings)
    red_rings = detect_red_rings(masks["LOSS"], hsv)
    if red_rings:
        rr = np.zeros(hsv.shape[:2], np.uint8)
        for cx, cy in red_rings:
            cv2.circle(rr, (int(cx), int(cy)), 20, 1, -1)
        masks["LOSS"] = masks["LOSS"] & ~rr.astype(bool)

    allm = masks["LOSS"] | masks["SHORT"] | masks["LONG"]
    grow = lambda m: cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)

    lines = []
    for color, m in masks.items():
        others = allm & ~m
        ps = NP.join_collinear(NP.extract_paths(m, min_len=6), max_gap=90, max_turn_deg=40, occluder=grow(others))
        lines += [dict(color=color, px=p) for p in ps if NP._length(p) >= 25]

    lines = NP.split_v(lines, lay.img_to_field)          # cut carries that were traced fused at the handoff

    if expected is not None:
        n_target = expected["carries"]
        for gap, turn in ((30, 45), (60, 70)):
            if len(lines) <= n_target:
                break
            for color in COLORS:                           # merge only within a colour: a carry has one colour
                sel = [l for l in lines if l["color"] == color]
                over = len(lines) - n_target
                if len(sel) > 1 and over > 0:
                    merged = NP.join_collinear([l["px"] for l in sel], max_gap=gap, max_turn_deg=turn,
                                               max_merges=min(over, len(sel) - 1),
                                               reject=lambda p: NP.v_index(p, lay.img_to_field) is not None)
                    lines = [l for l in lines if l["color"] != color] + [dict(color=color, px=p) for p in merged]

    def yd_length(l):
        f = lay.img_to_field(l["px"])
        return float(np.linalg.norm(np.diff(f, axis=0), axis=1).sum())

    if expected is not None:                              # still too many: drop the tiniest leftover slivers
        lines.sort(key=yd_length)
        while len(lines) > expected["carries"] and yd_length(lines[0]) < 8.0:
            lines.pop(0)

    out = []
    for l in lines:
        px = l["px"]
        f = lay.img_to_field(np.array([px[0], px[-1]]))
        if f[0, 1] > f[1, 1]:                              # start in the backfield (lower y)
            px = px[::-1]
        out.append(dict(color=l["color"], td=False, fumble=False, px=px))
    # each ring marks the end of exactly one carry: the one whose end is nearest to it
    for rings, key in ((blue_rings, "td"), (red_rings, "fumble")):
        for r in rings:
            dists = [np.linalg.norm(o["px"][-1] - np.array(r)) for o in out]
            if dists and min(dists) < 45:
                out[int(np.argmin(dists))][key] = True
    for o in out:
        o["kind"] = o["color"]
    NP.attach_trunks(out, lay.img_to_field, MAX_HANDOFF_Y)                                  # strict: smooth continuation
    NP.attach_trunks(out, lay.img_to_field, MAX_HANDOFF_Y, max_dist=70.0, min_align=0.0, passes=2)   # looser, what is left
    n_deep_truncated = n_deep_dropped = 0
    kept = []
    for o in out:
        pts, truncated = resample_field(lay, o.pop("px"), ceiling=ceiling)
        if truncated:
            n_deep_truncated += 1
            if len(pts) < 2:          # the whole line lived beyond the calibrated grid
                n_deep_dropped += 1
                continue
        o["pts"] = pts
        o["handoff_ok"] = bool(pts[0][1] <= MAX_HANDOFF_Y)
        kept.append(o)
    out = kept
    out.sort(key=lambda c: (c["pts"][0][0], c["pts"][0][1]))
    return out, dict(depth_yd=round(lay.max_reliable_yard, 1), n_lines=len(lines), n_td_rings=len(blue_rings),
                     n_fumble_rings=len(red_rings), n_deep_truncated=n_deep_truncated, n_deep_dropped=n_deep_dropped)
