"""
Read the numbers printed in a route chart's stat box.

2023+ charts print TARGETS-REC / YARDS / TD / YAC under the field; older ones RECEPTIONS / REC YARDS /
TOUCHDOWNS. The chart metadata carries receptions, yards and TDs but not targets, and the route reader needs
targets to know how many incomplete (grey) routes to look for - so for 2023+ charts they are read off the image.

The digits are matched against templates learned from charts whose REC / YARDS / TD are known from the
metadata (statbox_digits.npz). Checked on 1,447 held-out charts: receptions 100%, yards 99.9%, TDs 99.9%.
"""
import os

import cv2
import numpy as np

BAND = (980, 1195)          # rows holding the stat box (its exact row varies a little by chart)
GLYPH = (20, 36)            # templates are compared at this size
_TEMPLATES = None


def _templates():
    global _TEMPLATES
    if _TEMPLATES is None:
        z = np.load(os.path.join(os.path.dirname(__file__), "statbox_digits.npz"))
        _TEMPLATES = {k: z[k] for k in z.files}
    return _TEMPLATES


def _numbers(img):
    """Big stat-box characters grouped into numbers, left to right: lists of {x, x1, kind, glyph}."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    band = (gray[BAND[0]:BAND[1], 250:1200] > 170).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(band, 8)
    digits, dashes = [], []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 25 <= h <= 50 and w >= 5 and a >= 40:
            digits.append((x, y, w, h, i))
        elif h <= 8 and w >= 8 and a >= 15:
            dashes.append((x, y, w, h, i))
    if not digits:
        return []
    cy = np.median([y + h / 2 for x, y, w, h, i in digits])          # the row of big digits
    digits = [d for d in digits if abs(d[1] + d[3] / 2 - cy) < 15]
    dashes = [d for d in dashes if abs(d[1] + d[3] / 2 - cy) < 10]
    comps = []
    for (x, y, w, h, i), kind in [(d, "digit") for d in digits] + [(d, "dash") for d in dashes]:
        g = (lab[y:y + h, x:x + w] == i).astype(np.float32)
        comps.append(dict(x=x + 250, x1=x + w + 250, kind=kind, glyph=cv2.resize(g, GLYPH, interpolation=cv2.INTER_AREA)))
    comps.sort(key=lambda c: c["x"])
    groups = [[comps[0]]]
    for c in comps[1:]:
        if c["x"] - groups[-1][-1]["x1"] < 45:
            groups[-1].append(c)
        else:
            groups.append([c])
    return groups


def _read(comps, T):
    digs = [c for c in comps if c["kind"] == "digit"]
    if not digs:
        return None
    s = "".join(min(T, key=lambda d: np.abs(T[d] - c["glyph"]).sum()) for c in digs)
    neg = any(c["kind"] == "dash" and c["x1"] <= digs[0]["x"] + 2 for c in comps)
    return -int(s) if neg else int(s)


def read_stats(image):
    """{'targets', 'receptions', 'yards', 'td', 'yac'} from a 2023+ chart's stat box, or None for older charts
    (or if the box can't be read). `image` is a decoded BGR array."""
    g = _numbers(image)
    if len(g) != 4:
        return None
    tr, yards, td, yac = g
    dash = [i for i, c in enumerate(tr) if c["kind"] == "dash"]
    if not dash:
        return None
    T = _templates()
    return dict(targets=_read(tr[:dash[0]], T), receptions=_read(tr[dash[0] + 1:], T),
                yards=_read(yards, T), td=_read(td, T), yac=_read(yac, T))
