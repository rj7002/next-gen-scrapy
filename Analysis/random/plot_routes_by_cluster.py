import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_PER_CLUSTER = 50  # sampled per cluster, plotting all 51k would be an unreadable blob
RESAMPLE_N = 15

CLUSTER_COLORS = {0: '#eb6834', 1: '#2a78d6', 2: '#1baf7a', 3: '#e34948'}  # categorical palette
CLUSTER_NAMES = {0: 'Out / In / Dig', 1: 'Comeback / Curl', 2: 'Drag / Flat / Crosser', 3: 'Go / Vertical / Seam'}


def draw_field(ax, ymin=-10, ymax=55):
    ymin, ymax = int(math.floor(ymin)), int(math.ceil(ymax))
    ax.set_facecolor('black')
    for y in range(ymin - ymin % 5, ymax + 1, 5):
        ax.axhline(y, color='white', alpha=0.2, lw=0.7)
    ax.axhline(0, color='#ffffff', lw=1.5, alpha=0.6, zorder=1)
    for x in (-26.67, 26.67):
        ax.axvline(x, color='white', lw=1.5)
    for x in (-3.08, 3.08):
        ax.axvline(x, color='white', alpha=0.2, lw=0.7, ls=':')
    ax.set_xlim(-30, 30)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('x (yards)')
    ax.set_ylabel('yards past line of scrimmage')


def resample_route(x, y, n=RESAMPLE_N):
    d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    arc = np.concatenate([[0], np.cumsum(d)])
    total = arc[-1]
    if total == 0:
        return np.full(n, x[0]), np.full(n, y[0])
    targets = np.linspace(0, total, n)
    return np.interp(targets, arc, x), np.interp(targets, arc, y)


def load_sampled_paths(clusters_csv, routes_csv):
    """Returns {cluster: [(rx, ry), ...]} of resampled, real-field-position route paths."""
    clusters = pd.read_csv(clusters_csv)

    sampled = pd.concat([
        g.sample(min(N_PER_CLUSTER, len(g)), random_state=0)
        for _, g in clusters.groupby('cluster')
    ])

    cluster_map = dict(zip(zip(sampled['game_id'], sampled['esb_id'], sampled['route_id']), sampled['cluster']))
    keys = set(cluster_map)
    print(f'Sampling {len(sampled)} routes across {clusters["cluster"].nunique()} clusters, '
          f'pulling raw coordinates from {routes_csv} ...')

    routes = pd.read_csv(routes_csv)
    routes = routes[routes['segment'] == 'route']
    routes['key'] = list(zip(routes['game_id'], routes['esb_id'], routes['route_id']))
    routes = routes[routes['key'].isin(keys)]

    paths_by_cluster = {c: [] for c in cluster_map.values()}
    for key, r in routes.groupby('key'):
        r = r.sort_values('point')
        x, y = r['x_coord'].to_numpy(), r['y_coord'].to_numpy()
        if len(x) < 2:
            continue
        rx, ry = resample_route(x, y)
        paths_by_cluster[cluster_map[key]].append((rx, ry))

    return paths_by_cluster


def plot_combined(paths_by_cluster, out_path):
    fig, ax = plt.subplots(figsize=(11, 13))
    draw_field(ax)

    ymax = 5
    for cluster, paths in paths_by_cluster.items():
        for rx, ry in paths:
            ax.plot(rx, ry, color=CLUSTER_COLORS[cluster], alpha=0.35, lw=0.9, zorder=3)
            ymax = max(ymax, ry.max())

    ax.set_ylim(-10, ymax + 3)
    n_total = sum(len(p) for p in paths_by_cluster.values())
    ax.set_title(f'Routes on the field, colored by cluster (n={n_total} sampled)')

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=CLUSTER_COLORS[c], lw=2, label=f'{CLUSTER_NAMES[c]} (cluster {c})')
               for c in sorted(CLUSTER_COLORS)]
    ax.legend(handles=handles, loc='upper right', fontsize=9, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor='white')
    print('Saved:', out_path)


def plot_grid(paths_by_cluster, out_path):
    clusters = sorted(paths_by_cluster)
    n_cols = 2
    n_rows = math.ceil(len(clusters) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 9 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, cluster in zip(axes, clusters):
        paths = paths_by_cluster[cluster]
        ymax = max((ry.max() for _, ry in paths), default=5)
        draw_field(ax, ymin=-10, ymax=ymax + 3)
        for rx, ry in paths:
            ax.plot(rx, ry, color=CLUSTER_COLORS[cluster], alpha=0.4, lw=1, zorder=3)
        ax.set_title(f'{CLUSTER_NAMES[cluster]}  (cluster {cluster}, n={len(paths)})', fontsize=12)

    for ax in axes[len(clusters):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor='white')
    print('Saved:', out_path)


if __name__ == '__main__':
    import sys
    clusters_csv = sys.argv[1] if len(sys.argv) > 1 else '/Users/ryan/Downloads/FF-Python-Analytics-master/Analysis/random/route_clusters.csv'
    routes_csv = sys.argv[2] if len(sys.argv) > 2 else '/Users/ryan/Downloads/FF-Python-Analytics-master/ALL DATA/route_coords.csv'
    out_prefix = sys.argv[3] if len(sys.argv) > 3 else ''

    paths_by_cluster = load_sampled_paths(clusters_csv, routes_csv)
    plot_combined(paths_by_cluster, f'{out_prefix}routes_by_cluster_on_field.png')
    plot_grid(paths_by_cluster, f'{out_prefix}routes_by_cluster_grid.png')
