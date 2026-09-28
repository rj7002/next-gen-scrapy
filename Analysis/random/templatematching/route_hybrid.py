"""Hybrid route classifier: rules + template matching.

Templates are digitised from a standard receiver route tree (takeoff/post/dig/corner/...).
Matching follows Kinney (arXiv:2003.05428) - symmetric point-to-polyline distance - with two
deliberate departures:

  * ABSOLUTE SCALE. Kinney normalises size via the bounding box, which makes a 3-yard quick out
    and a 10-yard speed out the same shape. Templates here live in real yards and are only allowed
    to scale within a narrow band, so depth-ladder routes (quick out / smash / speed out,
    in / dig) stay distinguishable.
  * ASPECT RATIO LOCKED. Scaling is uniform in both axes, so break angles never distort. This is
    Kinney's key insight: angles are what separate routes, so a stretched template turns a dig
    into a post.

Routes that template distance provably cannot see are handled by rule FIRST:
  * screens (no movement at all)
  * the reversal family - hitch / curl / comeback. A doubling-back route's points all sit on top
    of its own outbound line, so point-to-line distance is blind to it. Kinney predicted ZERO
    curls and comebacks for exactly this reason and left it to future work.

Coordinates are (inside, depth) in yards: +inside = toward the middle of the field, +depth =
downfield. Edit TEMPLATES freely - that's the whole point of doing it this way.
"""
import hashlib
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROUTES_CSV = '/Users/ryan/Downloads/FF-Python-Analytics-master/ALL DATA/route_coords.csv'

# The spec file is the source of truth for templates and thresholds - edit route_spec.json, not this.
SPEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'route_spec.json')
with open(SPEC_PATH) as _f:
    _RAW = _f.read()
SPEC = json.loads(_RAW)
SPEC_HASH = hashlib.sha256(_RAW.encode()).hexdigest()[:10]   # stamped into the output CSV

RESAMPLE_N = SPEC['matching']['resample_points']

# ---- derived from route_spec.json ----
TEMPLATES = {k: [tuple(p) for p in v['points']] for k, v in SPEC['templates'].items()}
_DEFAULT_SCALES = tuple(SPEC['matching']['default_scales'])
TEMPLATE_SCALES = {k: tuple(v['scales']) for k, v in SPEC['templates'].items() if v.get('scales')}
GAMMA = SPEC['matching']['gamma']
OTHER_MAX_DIST = SPEC['matching']['other_max_dist']

# ---- rule thresholds (from route_spec.json) ----
_R = SPEC['rules']
SCREEN_MAX_DEPTH = _R['SCREEN']['screen_max_depth']
ANGLE_MIN_DIP = _R['ANGLE']['angle_min_dip']
ANGLE_MIN_RELEASE = _R['ANGLE']['angle_min_release']
REVERSAL_MIN = _R['REVERSAL_FAMILY']['reversal_min']
COMEBACK_MIN_DEPTH = _R['REVERSAL_FAMILY']['comeback_min_depth']
WHEEL_SHALLOW_DEPTH = _R['WHEEL']['wheel_shallow_depth']
WHEEL_MIN_OUTSIDE = _R['WHEEL']['wheel_min_outside']
WHEEL_MIN_CLIMB = _R['WHEEL']['wheel_min_climb']
WHEEL_MAX_NET_INSIDE = _R['WHEEL']['wheel_max_net_inside']
WHEEL_MAX_FINAL_HEADING = _R['WHEEL']['wheel_max_final_heading']
BACKFIELD_RB_MAX_X = 4.0
MIN_POINTS = SPEC['exclusions']['min_points']

COLORS = {
    'GO (9)': '#e34948', 'POST (8)': '#eda100', 'CORNER (7)': '#eb6834', 'DIG (6)': '#1baf7a',
    'OUT (5)': '#2a78d6', 'SLANT (2)': '#e87ba4', 'FLAT (1)': '#86b6ef', 'SCREEN': '#b3b2ab', 'HITCH (0)': '#9ec5f4', 'CURL (4)': '#9085e9',
    'COMEBACK (3)': '#4a3aa7', 'WHEEL': '#d55181', 'DRAG': '#008300', 'FADE': '#c98500',
    'CROSS': '#00c8c8', 'ANGLE': '#a0522d', 'OTHER': '#555555', 'SEAM': '#ffd700',
}


def resample(x, y, n=RESAMPLE_N):
    d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    arc = np.concatenate([[0], np.cumsum(d)])
    if arc[-1] == 0:
        return np.full(n, x[0]), np.full(n, y[0])
    t = np.linspace(0, arc[-1], n)
    return np.interp(t, arc, x), np.interp(t, arc, y)


def densify(pts, n=RESAMPLE_N):
    """Polyline -> n evenly spaced points along it (Kinney adds points so cardinalities match)."""
    p = np.asarray(pts, float)
    seg = np.sqrt((np.diff(p, axis=0) ** 2).sum(1))
    arc = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, arc[-1], n)
    return np.column_stack([np.interp(t, arc, p[:, 0]), np.interp(t, arc, p[:, 1])])


TEMPLATE_PTS = {name: densify(pts) for name, pts in TEMPLATES.items()}


def pt_to_polyline(P, V):
    """Min distance from each point in P (n,2) to polyline V (m,2). Vectorised over segments."""
    A, B = V[:-1], V[1:]
    AB = B - A
    denom = (AB ** 2).sum(1)
    denom[denom == 0] = 1e-9
    PA = P[:, None, :] - A[None, :, :]
    t = np.clip((PA * AB[None]).sum(2) / denom[None], 0, 1)
    closest = A[None] + t[..., None] * AB[None]
    return np.sqrt(((P[:, None, :] - closest) ** 2).sum(2)).min(1)


def match_template(route_pts):
    """Symmetric point-to-polyline distance against every template x scale. Lowest wins."""
    best, best_d, best_s = None, np.inf, 1.0
    for name, tpl in TEMPLATE_PTS.items():
        for s in TEMPLATE_SCALES.get(name, _DEFAULT_SCALES):
            T = tpl * s
            d = pt_to_polyline(route_pts, T).mean() + GAMMA * pt_to_polyline(T, route_pts).mean()
            if d < best_d:
                best, best_d, best_s = name, d, s
    return best, float(best_d), best_s


def geometry(x, y):
    rx, ry = resample(x, y)
    side = 1.0 if rx[0] >= 0 else -1.0
    depth = ry - ry[0]
    inside = -side * (rx - rx[0])
    q = max(len(depth) // 4, 1)
    final_heading = math.degrees(math.atan2(inside[-1] - inside[-q], max(depth[-1] - depth[-q], 1e-6)))
    shallow = np.where(depth < WHEEL_SHALLOW_DEPTH)[0]
    if len(shallow):
        si = int(shallow[np.argmin(inside[shallow])])
        shallow_outside, climb_after = float(-inside[si]), float(depth[-1] - depth[si])
    else:
        shallow_outside, climb_after = 0.0, 0.0
    return dict(
        pts=np.column_stack([inside, depth]), depth=depth, inside=inside,
        max_depth=float(depth.max()), min_depth=float(depth.min()), net_inside=float(inside[-1]),
        end_depth=float(depth[-1]), reversal=float(depth.max() - depth[-1]), final_heading=final_heading,
        shallow_outside=shallow_outside, climb_after=climb_after, start_x=float(rx[0]),
    )


def classify(g):
    """Rules for what templates can't see, then template matching for everything else."""
    if g['max_depth'] < SCREEN_MAX_DEPTH:
        return 'SCREEN', 'rule', np.nan, np.nan

    # dips behind the LOS before releasing upfield - its own outbound/return legs overlap the same
    # way the reversal family's do, so point-to-polyline distance can't see the dip either.
    if g['min_depth'] <= ANGLE_MIN_DIP and g['end_depth'] >= ANGLE_MIN_RELEASE:
        return 'ANGLE', 'rule', np.nan, np.nan

    if (g['shallow_outside'] >= WHEEL_MIN_OUTSIDE and g['climb_after'] >= WHEEL_MIN_CLIMB
            and g['net_inside'] <= WHEEL_MAX_NET_INSIDE and abs(g['final_heading']) <= WHEEL_MAX_FINAL_HEADING):
        return 'WHEEL', 'rule', np.nan, np.nan

    # reversal family - invisible to point-to-line distance
    if g['reversal'] >= REVERSAL_MIN:
        if g['max_depth'] >= COMEBACK_MIN_DEPTH:
            return ('COMEBACK (3)' if g['net_inside'] < -1.0 else 'CURL (4)'), 'rule', np.nan, np.nan
        return 'HITCH (0)', 'rule', np.nan, np.nan

    name, dist, scale = match_template(g['pts'])
    if dist > OTHER_MAX_DIST:      # closest template is still a bad fit - don't report a false-confident name
        return 'OTHER', 'other', dist, scale
    return name, 'template', dist, scale


def build(routes_csv, limit=None):
    df = pd.read_csv(routes_csv)
    df = df[df['segment'] == 'route']

    rows, paths = [], {}
    groups = df.groupby(['game_id', 'esb_id', 'route_id'])
    for i, (key, r) in enumerate(groups):
        if limit and i >= limit:
            break
        r = r.sort_values('point')
        x, y = r['x_coord'].to_numpy(), r['y_coord'].to_numpy()
        if len(x) < MIN_POINTS:
            continue
        g = geometry(x, y)
        label, how, dist, scale = classify(g)
        pos = r['position'].iloc[0]
        rows.append(dict(
            game_id=key[0], esb_id=key[1], route_id=key[2], name=r['name'].iloc[0],
            position=pos, season=r['season'].iloc[0], route_type=r['route_type'].iloc[0],
            label=label, how=how, match_dist=dist, match_scale=scale,
            max_depth=g['max_depth'], net_inside=g['net_inside'], reversal=g['reversal'],
            # Kinney excludes backfield RBs: they navigate the line, not the route tree
            spec_hash=SPEC_HASH,
            backfield_rb=bool(pos in ('RB', 'FB', 'HB') and abs(g['start_x']) < BACKFIELD_RB_MAX_X),
        ))
        paths[key] = (g['inside'], g['depth'])
    return pd.DataFrame(rows), paths


def report(lab):
    core = lab[~lab['backfield_rb']]
    s = core.groupby('label').agg(
        n=('label', 'size'), how=('how', lambda v: v.iloc[0]),
        avg_depth=('max_depth', 'mean'),
        comp_pct=('route_type', lambda v: (v == 'COMPLETE').mean() * 100),
        avg_dist=('match_dist', 'mean'),
    ).round(2).sort_values('n', ascending=False)
    s['share_pct'] = (s['n'] / len(core) * 100).round(1)
    print(f'spec {SPEC["spec_version"]} (hash {SPEC_HASH}) - {len(TEMPLATES)} templates, rules: {", ".join(SPEC["rules"]["order"])}')
    print(f'\nClassified {len(core):,} routes '
          f'({lab["backfield_rb"].sum():,} backfield-RB routes excluded):')
    print(s[['n', 'share_pct', 'how', 'avg_depth', 'comp_pct', 'avg_dist']])
    return s


def plot_exemplars(lab, paths, out, per=60):
    core = lab[~lab['backfield_rb']]
    labels = core['label'].value_counts().index.tolist()
    ncol, nrow = 4, math.ceil(len(labels) / 4)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 5.0 * nrow))
    axes = np.atleast_1d(axes).flatten()

    for ax, label in zip(axes, labels):
        sub = core[core['label'] == label]
        sample = sub.sample(min(per, len(sub)), random_state=0)
        ax.set_facecolor('black')
        curves, ymax, xabs = [], 12, 8
        for _, row in sample.iterrows():
            inside, depth = paths[(row['game_id'], row['esb_id'], row['route_id'])]
            curves.append((inside, depth))
            ymax, xabs = max(ymax, depth.max()), max(xabs, np.abs(inside).max())
        for inside, depth in curves:
            ax.plot(inside, depth, color=COLORS.get(label, '#888'), alpha=0.4, lw=1.1, zorder=3)
        if label in TEMPLATE_PTS:   # overlay the template it was matched against
            T = TEMPLATE_PTS[label]
            ax.plot(T[:, 0], T[:, 1], color='white', lw=2.8, ls='--', zorder=6)
        ax.axhline(0, color='white', lw=1.3, alpha=0.7)
        ax.axvline(0, color='white', lw=0.7, alpha=0.2, ls=':')
        for yy in range(0, int(ymax) + 1, 5):
            ax.axhline(yy, color='white', alpha=0.15, lw=0.6)
        ax.set_xlim(-xabs - 3, xabs + 3)
        ax.set_ylim(-8, ymax + 2)
        ax.set_aspect('equal')
        how = sub['how'].iloc[0]
        ax.set_title(f'{label}  ({how})  n={len(sub):,}, {len(sub)/len(core)*100:.1f}%', fontsize=10)
        ax.set_xlabel('← outside    inside →', fontsize=8)

    for ax in axes[len(labels):]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out, dpi=170, facecolor='white')
    print('Saved:', out)


if __name__ == '__main__':
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    lab, paths = build(ROUTES_CSV, limit=limit)
    report(lab)
    lab.to_csv('route_labels_tree.csv', index=False)
    print('Saved: route_labels_tree.csv')
    plot_exemplars(lab, paths, 'route_types_tree.png')
