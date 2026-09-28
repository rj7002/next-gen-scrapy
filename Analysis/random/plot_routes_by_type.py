"""Plot classified routes at their REAL field positions - no centering, no mirroring - one panel
per route type. Unlike route_types_tree.png (which draws everything in the receiver's own frame),
this shows where on the field each route type actually gets run."""
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Analysis.random.templatematching.route_hybrid import COLORS

ROUTES_CSV = '/Users/ryan/Downloads/FF-Python-Analytics-master/ALL DATA/route_coords.csv'
LABELS_CSV = 'route_labels_tree.csv'
N_PER_LABEL = 400
RESAMPLE_N = 20


def draw_field(ax, ymin=-12, ymax=55):
    ymin, ymax = int(math.floor(ymin)), int(math.ceil(ymax))
    ax.set_facecolor('black')
    for y in range(ymin - ymin % 5, ymax + 1, 5):
        ax.axhline(y, color='white', alpha=0.18, lw=0.7)
    ax.axhline(0, color='white', lw=1.6, alpha=0.75, zorder=2)
    for x in (-26.67, 26.67):
        ax.axvline(x, color='white', lw=1.4)
    for x in (-3.08, 3.08):
        ax.axvline(x, color='white', alpha=0.18, lw=0.7, ls=':')
    ax.set_xlim(-30, 30)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('x (yards from ball)', fontsize=8)


def resample(x, y, n=RESAMPLE_N):
    d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    arc = np.concatenate([[0], np.cumsum(d)])
    if arc[-1] == 0:
        return np.full(n, x[0]), np.full(n, y[0])
    t = np.linspace(0, arc[-1], n)
    return np.interp(t, arc, x), np.interp(t, arc, y)


def main():
    lab = pd.read_csv(LABELS_CSV)
    lab = lab[~lab['backfield_rb']]

    sampled = pd.concat([g.sample(min(N_PER_LABEL, len(g)), random_state=0)
                         for _, g in lab.groupby('label')])
    label_map = dict(zip(zip(sampled.game_id, sampled.esb_id, sampled.route_id), sampled.label))
    print(f'Sampling {len(sampled):,} routes across {lab.label.nunique()} types ...')

    routes = pd.read_csv(ROUTES_CSV)
    routes = routes[routes['segment'] == 'route']
    routes['key'] = list(zip(routes.game_id, routes.esb_id, routes.route_id))
    routes = routes[routes['key'].isin(set(label_map))]

    paths = {}
    for key, r in routes.groupby('key'):
        r = r.sort_values('point')
        x, y = r['x_coord'].to_numpy(), r['y_coord'].to_numpy()
        if len(x) < 2:
            continue
        paths.setdefault(label_map[key], []).append(resample(x, y))   # raw field coords

    order = lab['label'].value_counts().index.tolist()
    order = [o for o in order if o in paths]
    ncol = 4
    nrow = math.ceil(len(order) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 6.2 * nrow))
    axes = np.atleast_1d(axes).flatten()

    for ax, label in zip(axes, order):
        curves = paths[label]
        ymax = max(12, max(ry.max() for _, ry in curves))
        draw_field(ax, ymin=-12, ymax=ymax + 3)
        for rx, ry in curves:
            ax.plot(rx, ry, color=COLORS.get(label, '#888'), alpha=0.28, lw=0.8, zorder=3)
        share = (lab.label == label).mean() * 100
        ax.set_title(f'{label}   n={len(curves)} shown / {(lab.label == label).sum():,} '
                     f'({share:.1f}%)', fontsize=10)

    for ax in axes[len(order):]:
        ax.axis('off')
    plt.tight_layout()
    out = 'route_types_on_field.png'
    plt.savefig(out, dpi=165, facecolor='white')
    print('Saved:', out)


if __name__ == '__main__':
    main()
