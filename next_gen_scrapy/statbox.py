"""
Read the numbers printed in a chart's stat box.

Route charts: 2023+ print TARGETS-REC / YARDS / TD / YAC under the field; older ones RECEPTIONS / REC YARDS /
TOUCHDOWNS. The chart metadata carries receptions, yards and TDs but not targets, and the route reader needs
targets to know how many incomplete (grey) routes to look for - so for 2023+ charts they are read off the image.

Every chart type: NGS's listing carries no stats at all for some charts (mostly 2021-23 playoff games), and the
readers need them (how many completions / TDs / routes / carries to find) - read_pass_stats / read_route_stats /
read_carry_stats read them off the image instead:
  pass   COMP/ATT, YARDS, TOUCHDOWNS (older) | + TDS, INTS (2021-22) | TD-INT, CPOE (2023+)
  carry  CARRIES, RUSH YARDS, TOUCHDOWNS | CARRIES, YARDS, TD, RYOE (2023+)

The digits are matched against templates learned from charts whose REC / YARDS / TD are known from the
metadata (statbox_digits.npz). Checked on 1,447 held-out charts: receptions 100%, yards 99.9%, TDs 99.9%.
"""
import os

import cv2
import numpy as np

BAND = (980, 1195)          # rows holding the stat box (its exact row varies a little by chart)
GLYPH = (20, 36)            # templates are compared at this size
_TEMPLATES = None


def _all():
    global _TEMPLATES
    if _TEMPLATES is None:
        z = np.load(os.path.join(os.path.dirname(__file__), "statbox_digits.npz"))
        _TEMPLATES = {k: z[k] for k in z.files}
    return _TEMPLATES


def _templates(bold=False):
    """Digit templates '0'-'9' for the regular stat-box font, or the heavier one some charts use (stored 'b0'-'b9').
    The 'slash' of COMP/ATT is kept apart: _slash()."""
    t = _all()
    if bold:
        return {k[1:]: v for k, v in t.items() if k.startswith("b") and k[1:].isdigit()}
    return {k: v for k, v in t.items() if k.isdigit()}


def _font(groups):
    """The template set that fits this chart's digits best (regular or bold)."""
    glyphs = [c["glyph"] for g in groups for c in g if c["kind"] == "digit" and not _is_slash(c)]
    if not glyphs:
        return _templates()
    sets = [_templates(), _templates(bold=True)]
    cost = [np.median([min(np.abs(T[d] - x).sum() for d in T) for x in glyphs]) if T else np.inf for T in sets]
    return sets[int(np.argmin(cost))]


def _slash():
    return _all().get("slash")


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
    T = _font(g)
    return dict(targets=_read(tr[:dash[0]], T), receptions=_read(tr[dash[0] + 1:], T),
                yards=_read(yards, T), td=_read(td, T), yac=_read(yac, T))


def _is_slash(c):
    sl = _slash()
    if sl is None or c["kind"] != "digit":
        return False
    d = np.abs(sl - c["glyph"]).sum()
    return all(d < min(np.abs(T[k] - c["glyph"]).sum() for k in T) for T in (_templates(), _templates(bold=True)) if T)


def _split_at(comps, test):
    for i, c in enumerate(comps):
        if test(c):
            return comps[:i], comps[i + 1:]
    return None, None


def read_pass_stats(image):
    """{'completions', 'attempts', 'passingYards', 'touchdowns', 'interceptions'} from a pass chart's stat box
    (interceptions is None on the oldest layout, which doesn't print them), or None if it can't be read."""
    g = _numbers(image)
    if len(g) < 3:
        return None
    T = _font(g)
    comp, att = _split_at(g[0], _is_slash)
    if comp is None:
        # the oldest layout's "/" is taller than the digits and isn't picked up: split at the widest gap instead
        first = [c for c in g[0] if c["kind"] == "digit"]
        if len(first) < 2:
            return None
        gaps = [first[i + 1]["x"] - first[i]["x1"] for i in range(len(first) - 1)]
        i = int(np.argmax(gaps))
        if len(gaps) > 1 and gaps[i] < 1.6 * float(np.median(gaps)):
            return None
        comp, att = first[:i + 1], first[i + 1:]
    out = dict(completions=_read(comp, T), attempts=_read(att, T), passingYards=_read(g[1], T), interceptions=None)
    third = g[2]
    dash = [i for i, c in enumerate(third) if c["kind"] == "dash" and 0 < i < len(third) - 1]
    if dash:                                         # 2023+: "TD-INT"
        out["touchdowns"], out["interceptions"] = _read(third[:dash[0]], T), _read(third[dash[0] + 1:], T)
    else:
        out["touchdowns"] = _read(third, T)
        if len(g) >= 4:                              # 2021-22: separate INTS
            out["interceptions"] = _read(g[3], T)
    return out if None not in (out["completions"], out["attempts"], out["touchdowns"]) else None


def read_route_stats(image):
    """{'receptions', 'receivingYards', 'touchdowns', 'targets' (2023+ only)} from a route chart's stat box."""
    s = read_stats(image)
    if s:
        return dict(receptions=s["receptions"], receivingYards=s["yards"], touchdowns=s["td"], targets=s["targets"])
    g = _numbers(image)
    if len(g) != 3:
        return None
    T = _font(g)
    out = dict(receptions=_read(g[0], T), receivingYards=_read(g[1], T), touchdowns=_read(g[2], T), targets=None)
    return out if None not in (out["receptions"], out["touchdowns"]) else None


def read_carry_stats(image):
    """{'carries', 'rushingYards', 'touchdowns'} from a carry chart's stat box."""
    g = _numbers(image)
    if len(g) not in (3, 4):
        return None
    T = _font(g)
    out = dict(carries=_read(g[0], T), rushingYards=_read(g[1], T), touchdowns=_read(g[2], T))
    return out if None not in (out["carries"], out["touchdowns"]) else None
