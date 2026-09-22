"""
Per-chart calibration of a Next Gen Stats chart image.

The charts are drawn as a perspective view of the ground plane, but the zoom is chosen per chart (so a
deep play fits), so there is no fixed set of layouts to hard-code. Instead every chart is calibrated
from its own pixels, using three features that are always drawn:

  * the blue line-of-scrimmage bar          -> the image row of y = 0
  * the two grey sideline bands             -> x = -/+ 26.667 yd, and (where they meet) the horizon
  * the horizontal yard lines, every 5 yd   -> the one remaining degree of freedom, the scale

Geometry. For a ground plane seen in perspective, the row of the line d yards past the LOS is

        row(d) = (A d + B) / (C d + 1)

B is the LOS row and A/C is the horizon row V, so with B and V known only C is left. Inverting,

        u(y) = (B - y) / (y - V) = C * d

so the detected yard lines have u values in the ratio 1 : 2 : 3 ... and C follows from matching them
to multiples of 5 yards. Everything is then pinned without knowing which line is which.
"""
import os

import cv2
import numpy as np

FIELD_HALF_WIDTH = 160.0 / 6.0   # 26.667 yd
FIELD_ROWS = 660                 # rows of the image that show the field
IMG_SIZE = 1200
YARD_STEP = 5.0                  # yard lines are drawn every 5 yards


def read_image(image):
    """`image` may be a file path (str/Path) or an already-decoded BGR array; returns the array."""
    return cv2.imread(str(image)) if isinstance(image, (str, os.PathLike)) else image


class CalibrationError(ValueError):
    """The chart could not be calibrated (blank image, or the site changed its art)."""


def _los_rows(im):
    b, g, r = (im[:FIELD_ROWS, :, i].astype(int) for i in range(3))
    rows = np.where(((b > 150) & (r < 80) & (g < 120)).mean(axis=1) > 0.6)[0]
    if not len(rows):
        raise CalibrationError("no field drawn: line-of-scrimmage bar not found")
    return rows


def _sideline_edges(im, y0=70, y1=470, min_run=12):
    """
    Inner edge of the left and right sideline bands for each row. The bands are the only wide runs of
    low-saturation, mid-brightness pixels, which is what separates them from the drawn lines.
    """
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    band = (s < 70) & (v > 70) & (v < 200)
    left, right = [], []
    for y in range(y0, min(y1, FIELD_ROWS)):
        row = band[y].view(np.int8)
        d = np.diff(np.concatenate([[0], row, [0]]))
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        runs = [(a, b) for a, b in zip(st, en) if b - a > min_run]
        if not runs:
            continue
        if runs[0][0] < IMG_SIZE // 2:
            left.append((y, runs[0][1]))          # inner (right) edge of the left band
        if runs[-1][1] > IMG_SIZE // 2:
            right.append((y, runs[-1][0]))        # inner (left) edge of the right band
    return np.array(left, float), np.array(right, float)


def _fit_line(pts, iters=8):
    """Robust col = m*row + c fit; returns (m, c, rms, inlier_fraction)."""
    if len(pts) < 30:
        raise CalibrationError("not enough sideline pixels (%d)" % len(pts))
    m, c = np.polyfit(pts[:, 0], pts[:, 1], 1)
    keep = np.ones(len(pts), bool)
    for _ in range(iters):
        res = pts[:, 1] - (m * pts[:, 0] + c)
        keep = np.abs(res) < max(1.0, 2.0 * np.median(np.abs(res)) * 1.48)
        if keep.sum() < 20:
            break
        m, c = np.polyfit(pts[keep, 0], pts[keep, 1], 1)
    rms = float(np.sqrt(np.mean((pts[keep, 1] - (m * pts[keep, 0] + c)) ** 2)))
    return m, c, rms, keep.mean()


def _yard_rows(im, los_last):
    """Rows of the drawn horizontal yard lines, above the LOS bar (brighter than the turf)."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(float)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    drawn = hsv[..., 1] > 90                       # coloured route / carry lines: ignore them
    prof = []
    for y in range(FIELD_ROWS):
        band = g[y, 300:900]
        ok = ~drawn[y, 300:900]
        prof.append(np.median(band[ok]) if ok.sum() > 200 else np.nan)
    prof = np.array(prof)
    rows = []
    for y in range(4, int(los_last) - 4):
        w = prof[y - 4:y + 5]
        if np.isnan(w).any():
            continue
        if prof[y] >= prof[y - 3] + 7 and prof[y] >= prof[y + 3] + 7 and prof[y] == np.nanmax(w):
            rows.append(y)
    # merge rows that are adjacent (a line is 1-2 px thick)
    merged = []
    for y in rows:
        if merged and y - merged[-1][-1] <= 2:
            merged[-1].append(y)
        else:
            merged.append([y])
    return np.array([float(np.mean(m)) for m in merged])


class Chart:
    """Calibration of one chart image: converts between image pixels and field yards."""

    def __init__(self, B, V, C, left, right, quality):
        self.B, self.V, self.C = B, V, C
        self.left, self.right = left, right
        self.quality = quality
        self.A = V * C
        far = self.row_to_yard(0.0) if C else 100.0
        src = np.array([[side * FIELD_HALF_WIDTH, d] for d in (far * 0.9, -10.0) for side in (-1, 1)],
                       dtype=np.float32)
        dst = np.array([self._corner(side, d) for d in (far * 0.9, -10.0) for side in (-1, 1)],
                       dtype=np.float32)
        self.H_field_to_img = cv2.getPerspectiveTransform(src, dst)
        self.H_img_to_field = np.linalg.inv(self.H_field_to_img)

    # --- the perspective model -------------------------------------------------
    def yard_to_row(self, d):
        d = np.asarray(d, dtype=float)
        return (self.A * d + self.B) / (self.C * d + 1.0)

    def row_to_yard(self, y):
        y = np.asarray(y, dtype=float)
        return (self.B - y) / (self.C * (y - self.V))

    def _corner(self, side, d):
        row = float(self.yard_to_row(d))
        m, c = self.left if side < 0 else self.right
        return [m * row + c, row]

    # --- conversions -----------------------------------------------------------
    def img_to_field(self, pts):
        """Nx2 array of (col, row) pixels -> Nx2 array of (x_yd, y_yd)."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_img_to_field).reshape(-1, 2)

    def field_to_img(self, pts):
        """Nx2 array of (x_yd, y_yd) -> Nx2 array of (col, row) pixels."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_field_to_img).reshape(-1, 2)

    def scale(self, cx, row):
        """Local pixels per yard along x and along y at image position (cx, row)."""
        p = self.img_to_field([[cx, row]])[0]
        q = self.field_to_img([[p[0] + 0.5, p[1]], [p[0] - 0.5, p[1]],
                               [p[0], p[1] + 0.5], [p[0], p[1] - 0.5]])
        return np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3])

    # --- masks -----------------------------------------------------------------
    def sideline_mask(self, shape=(IMG_SIZE, IMG_SIZE), margin=0.0):
        """Pixels inside the two sidelines (the playing surface)."""
        rows = np.arange(shape[0], dtype=float)[:, None]
        cols = np.arange(shape[1], dtype=float)[None, :]
        inside = (cols >= self.left[0] * rows + self.left[1] + margin) & \
                 (cols <= self.right[0] * rows + self.right[1] - margin)
        inside[FIELD_ROWS:] = False
        return inside

    def on_card_field(self, shape=(IMG_SIZE, IMG_SIZE)):
        """Rows of the card that show the field, minus the static overlay text."""
        m = np.zeros(shape, dtype=bool)
        m[:FIELD_ROWS] = True
        return m & ~self.text_mask(shape)

    def text_mask(self, shape=(IMG_SIZE, IMG_SIZE)):
        """
        The static overlay: the logo, and the 'LOS' / '+10' / '+20' ... labels, which are drawn just
        outside the sidelines. Derived from this chart's own geometry, so it follows the zoom.
        """
        m = np.zeros(shape, dtype=bool)
        m[0:78, 0:245] = True                                  # NFL / Next Gen Stats logo
        rows = np.arange(shape[0], dtype=float)[:, None]
        cols = np.arange(shape[1], dtype=float)[None, :]
        lx = self.left[0] * rows + self.left[1]
        rx = self.right[0] * rows + self.right[1]
        band = 105.0                                           # '+10', '+20' ... sit within ~105 px of the edge
        outside = ((cols < lx) & (cols > lx - band)) | ((cols > rx) & (cols < rx + band))
        outside[FIELD_ROWS:] = False
        m |= outside
        r0, r1 = int(self.B) - 21, int(self.B) + 17            # the two 'LOS' labels, in the corners
        m[max(0, r0):r1, 0:78] = True
        m[max(0, r0):r1, shape[1] - 78:] = True
        return m

    def los_bar_mask(self, shape=(IMG_SIZE, IMG_SIZE), pad=4):
        """The drawn blue line-of-scrimmage bar."""
        m = np.zeros(shape, dtype=bool)
        m[max(0, int(self.los_first) - pad):int(self.los_last) + pad + 1] = True
        return m

    def __repr__(self):
        return "<Chart LOS row %.1f, horizon %.0f, %.1f yd deep, fit %.2f px>" % (
            self.B, self.V, self.row_to_yard(0.0), self.quality["yard_rms"])


def calibrate(im):
    """Calibrate one chart image. Raises CalibrationError if its geometry cannot be recovered."""
    bar = _los_rows(im)
    B = float(bar[-1]) + 1.0                      # y = 0 sits just under the drawn bar

    left_pts, right_pts = _sideline_edges(im)
    ml, cl, rms_l, frac_l = _fit_line(left_pts)
    mr, cr, rms_r, frac_r = _fit_line(right_pts)
    if ml >= 0 or mr <= 0:
        raise CalibrationError("sideline slopes are not a perspective view (%.3f, %.3f)" % (ml, mr))
    if abs(ml + mr) > 0.05:
        raise CalibrationError("sidelines are not symmetric (%.3f, %.3f)" % (ml, mr))
    V = float((cr - cl) / (ml - mr))              # horizon: where the two sidelines meet
    if V > B - 200:
        raise CalibrationError("horizon row %.0f is not above the field" % V)

    rows = _yard_rows(im, bar[0])
    u = (B - rows) / (rows - V)                   # = C * d, so these are in the ratio 1:2:3...
    u = np.sort(u[u > 1e-6])
    if len(u) < 4:
        raise CalibrationError("only %d yard lines found" % len(u))
    # Scale by vote: the detector also picks up shadows and drawn lines, so instead of trusting every
    # row, try "this row is the +5 / +10 / ... line", and keep the scale that puts the most rows on
    # the 5-yard grid. Spurious rows simply fail to vote.
    tol = 0.35
    best = None
    for ui in u:
        for d0 in np.arange(YARD_STEP, 65.0, YARD_STEP):
            C = float(ui / d0)
            if C <= 0:
                continue
            d = u / C
            err = np.abs(d - np.round(d / YARD_STEP) * YARD_STEP)
            hit = err < tol
            if hit.sum() < 4 or d.max() > 120:
                continue
            key = (-int(hit.sum()), float(err[hit].mean()))
            if best is None or key < best[0]:
                best = (key, C, err, hit)
    if best is None:
        raise CalibrationError("yard lines do not fit a 5-yard grid (%d rows)" % len(u))
    _, C, err, hit = best

    chart = Chart(B, V, C, (ml, cl), (mr, cr),
                  quality=dict(yard_rms=float(np.sqrt(np.mean(err[hit] ** 2))), n_yard_lines=int(hit.sum()),
                               sideline_rms=max(rms_l, rms_r), sideline_inliers=min(frac_l, frac_r)))
    chart.los_first, chart.los_last = float(bar[0]), float(bar[-1])
    return chart
