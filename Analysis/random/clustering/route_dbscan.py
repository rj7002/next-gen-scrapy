"""
DBSCAN-based route shape discovery.

Unlike route_clustering.py (KMeans on interpretable shape ratios, k chosen by silhouette), this
clusters directly on the resampled (x, y) point sequence itself, after translating each route to
start at (0, 0) and mirroring it so its overall break always points the same way. DBSCAN needs no
k up front and - the useful part for this - leaves anything that doesn't sit in a dense group as
noise (-1) instead of forcing it into the nearest cluster, so a diffuse population (no dominant
shape) shows up as mostly noise rather than a false cluster.

Absolute scale is kept (no depth normalization): a 5-yd out and a 50-yd corner should NOT cluster
together just because they're both "a break to the sideline" - depth is part of the shape, same
principle as the template scales in route_spec.json.

Usage:
    python3 route_dbscan.py OTHER_ROUTES.CSV --eps ...           # cluster
    python3 route_dbscan.py OTHER_ROUTES.CSV                     # no --eps: scan candidates first
"""
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from Analysis.random.clustering.route_clustering import draw_field

TARGET_POINTS = 20


def normalize_and_mirror_route(route_points, target_points=TARGET_POINTS):
    """Translate to (0,0), mirror so the route breaks right, resample to target_points by arc length."""
    route = np.asarray(route_points, dtype=float)
    route = route - route[0]
    if route[-1, 0] < 0:
        route[:, 0] = -route[:, 0]

    d = np.sqrt(np.diff(route[:, 0]) ** 2 + np.diff(route[:, 1]) ** 2)
    arc = np.concatenate([[0], np.cumsum(d)])
    total = arc[-1]
    if total == 0:
        resampled = np.tile(route[0], (target_points, 1))
    else:
        targets = np.linspace(0, total, target_points)
        rx = np.interp(targets, arc, route[:, 0])
        ry = np.interp(targets, arc, route[:, 1])
        resampled = np.column_stack([rx, ry])
    return resampled.flatten()


def load_routes(csv_path, labels_csv=None, label_filter=None):
    """
    Reads a route_coords.csv-shaped file (game_id, esb_id, route_id, point, segment, x_coord,
    y_coord, ...). If labels_csv + label_filter are given (e.g. route_labels_tree.csv,
    ["OTHER"]), restricts to just those already-classified routes.
    """
    df = pd.read_csv(csv_path)
    df = df[df['segment'] == 'route']

    keys = None
    if labels_csv and label_filter:
        lab = pd.read_csv(labels_csv)
        lab = lab[(~lab['backfield_rb']) & (lab['label'].isin(label_filter))]
        keys = set(zip(lab.game_id, lab.esb_id, lab.route_id))

    raw_routes, meta = [], []
    for (game_id, esb_id, route_id), r in df.groupby(['game_id', 'esb_id', 'route_id']):
        if keys is not None and (game_id, esb_id, route_id) not in keys:
            continue
        r = r.sort_values('point')
        x, y = r['x_coord'].to_numpy(), r['y_coord'].to_numpy()
        if len(x) < 5:
            continue
        raw_routes.append(np.column_stack([x, y]))
        meta.append(dict(game_id=game_id, esb_id=esb_id, route_id=route_id, name=r['name'].iloc[0]))
    return raw_routes, pd.DataFrame(meta)


def scan_eps(X, min_samples, candidates):
    """DBSCAN is sensitive to eps - scan a range and report cluster count + noise fraction for each,
    rather than guessing a single value blindly the way the mock example's eps=1.5 does."""
    print(f'{"eps":>6} {"n_clusters":>11} {"noise_pct":>10}')
    for eps in candidates:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct = (labels == -1).mean() * 100
        print(f'{eps:6.2f} {n_clusters:11d} {noise_pct:9.1f}%')


def main(csv_path, labels_csv=None, label_filter=None, eps=None, min_samples=8, out_prefix='dbscan_'):
    raw_routes, meta = load_routes(csv_path, labels_csv, label_filter)
    print(f'Loaded {len(raw_routes)} routes'
          + (f' (filtered to {label_filter} from {labels_csv})' if label_filter else ''))

    features = np.array([normalize_and_mirror_route(r) for r in raw_routes])
    X = StandardScaler().fit_transform(features)

    if eps is None:
        print('No --eps given - scanning candidates (pick one from below, then re-run with --eps):')
        scan_eps(X, min_samples, np.arange(1.0, 12.5, 0.5))
        return

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean() * 100
    print(f'eps={eps}, min_samples={min_samples}: {n_clusters} clusters, {noise_pct:.1f}% noise '
          f'({(labels == -1).sum()} / {len(labels)} routes)')

    meta = meta.copy()
    meta['cluster'] = labels
    meta.to_csv(f'{out_prefix}clusters.csv', index=False)
    print(f'Saved: {out_prefix}clusters.csv')

    if n_clusters == 0:
        print('No clusters found at this eps - try a larger value from the scan.')
        return

    order = meta.loc[meta['cluster'] != -1, 'cluster'].value_counts().index.tolist()
    ncol = 4
    nrow = math.ceil(len(order) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 5.6 * nrow))
    axes = np.atleast_1d(axes).flatten()

    rng = np.random.default_rng(0)
    for ax, c in zip(axes, order):
        idx = np.where(labels == c)[0]
        sample_idx = rng.choice(idx, size=min(40, len(idx)), replace=False)
        pts_all = features[idx].reshape(len(idx), TARGET_POINTS, 2)
        pts_sample = features[sample_idx].reshape(len(sample_idx), TARGET_POINTS, 2)
        ymax = max(5, pts_all[:, :, 1].max())

        draw_field(ax, ymin=-5, ymax=ymax + 3)
        for pts in pts_sample:
            ax.plot(pts[:, 0], pts[:, 1], color='#9aa0a6', alpha=0.35, lw=1, zorder=3)
        avg = pts_all.mean(axis=0)
        ax.plot(avg[:, 0], avg[:, 1], color='#ffb300', lw=3, zorder=5)
        ax.scatter([0], [0], color='#2f6bff', s=30, zorder=6)
        ax.set_title(f'Cluster {c}  (n={len(idx)})', fontsize=11)

    for ax in axes[len(order):]:
        ax.axis('off')
    plt.tight_layout()
    out = f'{out_prefix}clusters.png'
    plt.savefig(out, dpi=180, facecolor='white')
    print('Saved:', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csv_path', nargs='?', default='route_coords.csv')
    ap.add_argument('--labels-csv', default=None, help='route_labels_tree.csv, to restrict to specific label(s)')
    ap.add_argument('--label-filter', nargs='+', default=None, help='e.g. --label-filter OTHER')
    ap.add_argument('--eps', type=float, default=None)
    ap.add_argument('--min-samples', type=int, default=8)
    ap.add_argument('--out-prefix', default='dbscan_')
    args = ap.parse_args()
    main(args.csv_path, args.labels_csv, args.label_filter, args.eps, args.min_samples, args.out_prefix)
