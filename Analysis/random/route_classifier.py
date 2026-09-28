"""Rule-based NFL route-tree classifier.

Unsupervised clustering can't recover named route types - the route tree is a human-defined
taxonomy, not a natural grouping. So this encodes the geometric definitions directly.

Everything is measured in the receiver's own frame:
  depth(t)  = yards past the line of scrimmage
  inside(t) = yards toward the middle of the field (away from his nearest sideline)
so a slant and a quick-out are no longer mirror images of each other.

All thresholds are constants below - tune them and re-run.
"""
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROUTES_CSV = '/Users/ryan/Downloads/FF-Python-Analytics-master/ALL DATA/route_coords.csv'
RESAMPLE_N = 40
BREAK_WINDOW = 4   # points each side; measures the cut angle over a span, not between adjacent
                   # samples - arc-length resampling smooths a sharp cut below any adjacent-point
                   # threshold, which is what buried the OUT/DIG/POST/CORNER routes.

# ---- tunable thresholds (yards / degrees) ----
SCREEN_MAX_DEPTH = 1.5     # never really crosses the LOS
SHALLOW_DEPTH = 8.0        # "shallow" ceiling: drags, slants, flats
DEEP_BREAK_DEPTH = 7.0     # a break at/after this depth is an intermediate-or-deeper break
GO_MAX_LATERAL = 6.0       # a go route stays in its lane
GO_MIN_DEPTH = 12.0
REVERSAL_MIN = 2.0         # yards lost off the peak = came back to the ball
COMEBACK_MIN_DEPTH = 10.0
CONTINUES_DEEP = 3.0       # depth gained after the break (post/corner keep climbing)
BREAK_MIN_ANGLE = 25.0     # degrees of direction change to count as a real break
CROSS_MIN_INSIDE = 8.0     # a drag crosses the formation
SLANT_MAX_DEPTH = 14.0
DIAGONAL_LATERAL = 6.0       # min drift for a breakless deep route to be a post/corner
DIAGONAL_MIN_HEADING = 22.0  # ...and it must still be angling this hard at the finish
# wheel = flat to the sideline first, THEN straight up it (strict, or it eats every deep fade)
WHEEL_SHALLOW_DEPTH = 6.0
WHEEL_MIN_OUTSIDE = 3.0
WHEEL_MIN_CLIMB = 12.0
WHEEL_MAX_FINAL_HEADING = 35.0

ROUTE_COLORS = {
    'GO': '#e34948', 'POST': '#eda100', 'CORNER': '#eb6834', 'DIG/IN': '#1baf7a',
    'OUT': '#2a78d6', 'COMEBACK': '#4a3aa7', 'CURL': '#9085e9', 'HITCH': '#86b6ef',
    'SLANT': '#e87ba4', 'DRAG': '#008300', 'FLAT': '#5598e7', 'SCREEN': '#b3b2ab',
    'WHEEL': '#d55181', 'SEAM': '#6da7ec', 'FADE': '#c98500', 'OTHER': '#52514e',
}


def draw_field(ax, ymin=-8, ymax=30):
    ymin, ymax = int(math.floor(ymin)), int(math.ceil(ymax))
    ax.set_facecolor('black')
    for y in range(ymin - ymin % 5, ymax + 1, 5):
        ax.axhline(y, color='white', alpha=0.18, lw=0.7)
    ax.axhline(0, color='white', lw=1.4, alpha=0.7, zorder=2)
    ax.axvline(0, color='white', alpha=0.18, lw=0.7, ls=':')
    ax.set_aspect('equal')


def resample(x, y, n=RESAMPLE_N):
    d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    arc = np.concatenate([[0], np.cumsum(d)])
    if arc[-1] == 0:
        return np.full(n, x[0]), np.full(n, y[0])
    t = np.linspace(0, arc[-1], n)
    return np.interp(t, arc, x), np.interp(t, arc, y)


def geometry(x, y):
    """Path -> features in the receiver's own (depth, inside) frame."""
    rx, ry = resample(x, y)
    side = 1.0 if rx[0] >= 0 else -1.0          # which half of the field he lined up on
    depth = ry - ry[0]
    inside = -side * (rx - rx[0])                # + = toward the middle, - = toward his sideline

    seg = np.sqrt(np.diff(inside) ** 2 + np.diff(depth) ** 2)
    path_len = seg.sum()

    # Cut angle at point i = heading over [i, i+w] minus heading over [i-w, i].
    # Measuring across a span (not adjacent samples) survives arc-length smoothing.
    w = BREAK_WINDOW
    bi, break_angle, break_dir = 0, 0.0, 0.0
    for i in range(w, len(depth) - w):
        before = math.degrees(math.atan2(inside[i] - inside[i - w], depth[i] - depth[i - w]))
        after = math.degrees(math.atan2(inside[i + w] - inside[i], depth[i + w] - depth[i]))
        turn = (after - before + 180) % 360 - 180
        if abs(turn) > abs(break_angle):
            bi, break_angle, break_dir = i, turn, float(np.sign(turn))
    break_angle = abs(break_angle)

    # final heading over the last quarter of the route (0 = straight downfield)
    q = max(len(depth) // 4, 1)
    final_heading = math.degrees(math.atan2(inside[-1] - inside[-q], max(depth[-1] - depth[-q], 1e-6)))

    # wheel signature: farthest-outside point reached while still shallow, then climb from there
    shallow = np.where(depth < WHEEL_SHALLOW_DEPTH)[0]
    if len(shallow):
        si = int(shallow[np.argmin(inside[shallow])])
        shallow_outside = float(-inside[si])
        climb_after = float(depth[-1] - depth[si])
    else:
        shallow_outside, climb_after = 0.0, 0.0

    return dict(
        side=side, depth=depth, inside=inside, path_len=path_len,
        max_depth=float(depth.max()), final_depth=float(depth[-1]),
        net_inside=float(inside[-1]), max_abs_lateral=float(np.abs(inside).max()),
        reversal=float(depth.max() - depth[-1]),
        break_angle=break_angle, break_depth=float(depth[bi]), break_dir=break_dir,
        break_frac=bi / max(len(depth) - 1, 1),
        post_break_depth=float(depth[-1] - depth[bi]),
        post_break_inside=float(inside[-1] - inside[bi]),
        final_heading=final_heading,
        shallow_outside=shallow_outside, climb_after=climb_after,
        rx=rx, ry=ry,
    )


LAST_RULE = {'v': ''}


def classify(g):
    """Route tree, first match wins. Order matters - specific patterns before generic ones."""
    md, fd = g['max_depth'], g['final_depth']
    ni, rev = g['net_inside'], g['reversal']
    bd, bdir, bang = g['break_depth'], g['break_dir'], g['break_angle']
    pbd = g['post_break_depth']
    has_break = bang >= BREAK_MIN_ANGLE

    # never gets past the line
    if md < SCREEN_MAX_DEPTH:
        return 'SCREEN'

    # wheel: gets outside while still shallow, THEN turns and climbs up the sideline
    if (g['shallow_outside'] >= WHEEL_MIN_OUTSIDE and g['climb_after'] >= WHEEL_MIN_CLIMB
            and ni <= -4.0 and abs(g['final_heading']) <= WHEEL_MAX_FINAL_HEADING):
        return 'WHEEL'

    # came back toward the ball off the peak
    if rev >= REVERSAL_MIN:
        if md >= COMEBACK_MIN_DEPTH:
            return 'COMEBACK' if ni < -1.0 else 'CURL'
        return 'HITCH'

    # a real cut at intermediate depth or deeper
    if has_break and bd >= DEEP_BREAK_DEPTH:
        LAST_RULE['v'] = 'break_deep'
        if bdir > 0:
            return 'POST' if pbd >= CONTINUES_DEEP else 'DIG/IN'
        return 'CORNER' if pbd >= CONTINUES_DEEP else 'OUT'

    # a cut close to the line
    if has_break and bd < DEEP_BREAK_DEPTH and md < GO_MIN_DEPTH:
        LAST_RULE['v'] = 'break_shallow'
        if ni >= CROSS_MIN_INSIDE:
            return 'DRAG'
        if ni > 1.0:
            return 'SLANT'
        if ni < -1.0:
            return 'FLAT'
        return 'HITCH'

    # No distinct cut, just a deep line. A post/corner REQUIRES a break, so these aren't posts
    # and corners - a gradually-bending vertical is a seam (inside) or a fade (to the sideline).
    if md >= GO_MIN_DEPTH:
        LAST_RULE['v'] = 'nobreak_deep'
        fh = g['final_heading']
        if abs(fh) >= DIAGONAL_MIN_HEADING and abs(ni) >= DIAGONAL_LATERAL:
            return 'SEAM' if fh > 0 else 'FADE'
        return 'GO'

    if ni >= CROSS_MIN_INSIDE:
        return 'DRAG'
    if ni > 1.5:
        return 'SLANT'
    if ni < -1.5:
        return 'FLAT'
    return 'HITCH'


def build(routes_csv):
    df = pd.read_csv(routes_csv)
    df = df[df['segment'] == 'route']

    rows, paths = [], {}
    for key, r in df.groupby(['game_id', 'esb_id', 'route_id']):
        r = r.sort_values('point')
        x, y = r['x_coord'].to_numpy(), r['y_coord'].to_numpy()
        if len(x) < 5:
            continue
        g = geometry(x, y)
        LAST_RULE['v'] = 'fallthrough'
        label = classify(g)
        rows.append(dict(
            game_id=key[0], esb_id=key[1], route_id=key[2],
            name=r['name'].iloc[0], position=r['position'].iloc[0], team=r['team'].iloc[0],
            season=r['season'].iloc[0], route_type=r['route_type'].iloc[0],
            rule=LAST_RULE['v'], label=label, side=g['side'], max_depth=g['max_depth'], final_depth=g['final_depth'],
            net_inside=g['net_inside'], reversal=g['reversal'], break_depth=g['break_depth'],
            break_angle=g['break_angle'], break_dir=g['break_dir'], post_break_depth=g['post_break_depth'],
        ))
        paths[key] = (g['depth'], g['inside'])
    return pd.DataFrame(rows), paths


def report(lab):
    n = len(lab)
    s = lab.groupby('label').agg(
        n=('label', 'size'),
        avg_depth=('max_depth', 'mean'),
        avg_break_depth=('break_depth', 'mean'),
        comp_pct=('route_type', lambda v: (v == 'COMPLETE').mean() * 100),
    ).round(1).sort_values('n', ascending=False)
    s['share_pct'] = (s['n'] / n * 100).round(1)
    print(f'\nClassified {n:,} routes:')
    print(s[['n', 'share_pct', 'avg_depth', 'avg_break_depth', 'comp_pct']])
    return s


def plot_exemplars(lab, paths, out, per=60):
    """Sampled routes per class, drawn in the receiver frame: up = downfield, right = inside."""
    labels = lab['label'].value_counts().index.tolist()
    ncol = 4
    nrow = math.ceil(len(labels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 5.0 * nrow))
    axes = np.atleast_1d(axes).flatten()

    for ax, label in zip(axes, labels):
        sub = lab[lab['label'] == label]
        sample = sub.sample(min(per, len(sub)), random_state=0)
        ymax, xabs = 12, 8
        curves = []
        for _, row in sample.iterrows():
            depth, inside = paths[(row['game_id'], row['esb_id'], row['route_id'])]
            curves.append((inside, depth))
            ymax = max(ymax, depth.max())
            xabs = max(xabs, np.abs(inside).max())

        draw_field(ax, ymin=-8, ymax=ymax + 2)
        for inside, depth in curves:
            ax.plot(inside, depth, color=ROUTE_COLORS.get(label, '#888'), alpha=0.45, lw=1.1, zorder=3)
        avg_i = np.mean([c[0] for c in curves], axis=0)
        avg_d = np.mean([c[1] for c in curves], axis=0)
        ax.plot(avg_i, avg_d, color='white', lw=2.6, zorder=5)
        ax.scatter([0], [0], color='#ffb300', s=22, zorder=6)
        ax.set_xlim(-xabs - 3, xabs + 3)
        ax.set_title(f'{label}  (n={len(sub):,}, {len(sub)/len(lab)*100:.1f}%)', fontsize=11)
        ax.set_xlabel('← outside    inside →', fontsize=8)

    for ax in axes[len(labels):]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out, dpi=170, facecolor='white')
    print('Saved:', out)


if __name__ == '__main__':
    routes_csv = sys.argv[1] if len(sys.argv) > 1 else ROUTES_CSV
    lab, paths = build(routes_csv)
    report(lab)
    lab.drop(columns=[]).to_csv('route_labels.csv', index=False)
    print('Saved: route_labels.csv')
    plot_exemplars(lab, paths, 'route_types.png')
