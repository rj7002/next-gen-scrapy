import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RESAMPLE_N = 15  # points per route after arc-length resampling


def draw_field(ax, ymin=-10, ymax=40):
    ymin, ymax = int(math.floor(ymin)), int(math.ceil(ymax))
    ax.set_facecolor('black')
    for y in range(ymin - ymin % 5, ymax + 1, 5):
        ax.axhline(y, color='white', alpha=0.25, lw=0.8)
    ax.axhline(0, color='#2f6bff', lw=2.5, zorder=1)
    for x in (-26.67, 26.67):
        ax.axvline(x, color='white', lw=1.5)
    for x in (-3.08, 3.08):
        ax.axvline(x, color='white', alpha=0.25, lw=0.8, ls=':')
    ax.set_xlim(-30, 30)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')


def resample_route(x, y, n=RESAMPLE_N):
    """Resample a route to n points evenly spaced by arc length (removes frame-rate/duration noise)."""
    d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    arc = np.concatenate([[0], np.cumsum(d)])
    total = arc[-1]
    if total == 0:
        return np.full(n, x[0]), np.full(n, y[0])
    targets = np.linspace(0, total, n)
    rx = np.interp(targets, arc, x)
    ry = np.interp(targets, arc, y)
    return rx, ry


def route_features(rx, ry):
    """Interpretable shape features for one resampled route. Mirrored so lateral break is always >= 0."""
    if (rx[-1] - rx[0]) < 0:
        rx = -rx  # mirror left/right breaks onto the same side so shape (not direction) drives clustering

    net_depth = ry[-1] - ry[0]
    max_depth = ry.max() - ry[0]
    net_lateral = rx[-1] - rx[0]
    max_lateral_dev = np.abs(rx - rx[0]).max()

    seg_d = np.sqrt(np.diff(rx) ** 2 + np.diff(ry) ** 2)
    path_length = seg_d.sum()
    end_to_end = math.hypot(rx[-1] - rx[0], ry[-1] - ry[0])
    straightness = end_to_end / path_length if path_length > 0 else 1.0

    heading = np.degrees(np.arctan2(np.diff(rx), np.diff(ry)))  # 0 = straight downfield
    dh = np.diff(heading)
    dh = (dh + 180) % 360 - 180  # wrap to [-180, 180]
    break_idx = np.argmax(np.abs(dh)) if len(dh) else 0
    break_angle = np.abs(dh).max() if len(dh) else 0.0
    break_frac = break_idx / max(len(dh) - 1, 1)
    depth_at_break = ry[break_idx + 1] - ry[0]

    end_heading = np.abs(heading[-3:].mean()) if len(heading) >= 3 else np.abs(heading.mean())
    reversal = max(0.0, max_depth - net_depth)  # comeback/curl signal

    return dict(net_depth=net_depth, max_depth=max_depth, net_lateral=net_lateral,
                max_lateral_dev=max_lateral_dev, path_length=path_length, straightness=straightness,
                break_angle=break_angle, break_frac=break_frac, depth_at_break=depth_at_break,
                end_heading=end_heading, reversal=reversal)


def main(csv_path='route_coords.csv', out_prefix=''):
    df = pd.read_csv(csv_path)

    rows = []
    resampled = {}
    # route_id is only unique WITHIN a player's own routes in a game (it restarts at 1 for each
    # receiver), so grouping by (game_id, route_id) alone silently merges different receivers'
    # routes together. esb_id is needed to scope it correctly.
    for (game_id, esb_id, route_id), r in df.groupby(['game_id', 'esb_id', 'route_id']):
        r = r[r['segment'] == 'route'].sort_values('point')  # exclude after-catch/YAC points
        x, y = r['x_coord'].to_numpy(), r['y_coord'].to_numpy()
        if len(x) < 5:
            continue
        rx, ry = resample_route(x, y)
        feats = route_features(rx, ry)
        feats.update(game_id=game_id, esb_id=esb_id, route_id=route_id, name=r['name'].iloc[0],
                      team=r['team'].iloc[0], route_type=r['route_type'].iloc[0])
        rows.append(feats)
        resampled[(game_id, esb_id, route_id)] = (rx, ry)

    feat_df = pd.DataFrame(rows)

    # Scale-invariant shape ratios: dividing by path_length means a 5-yd out and a 15-yd out
    # (or a slant run from different depths) land in the same "shape family" instead of being
    # split apart by how far downfield they happened to go.
    feat_df['lateral_ratio'] = feat_df['max_lateral_dev'] / feat_df['path_length']
    feat_df['net_lateral_ratio'] = feat_df['net_lateral'] / feat_df['path_length']
    feat_df['depth_ratio'] = feat_df['net_depth'] / feat_df['path_length']
    feat_df['reversal_ratio'] = feat_df['reversal'] / feat_df['path_length']

    feature_cols = ['straightness', 'break_angle', 'break_frac', 'end_heading',
                     'lateral_ratio', 'net_lateral_ratio', 'depth_ratio', 'reversal_ratio']
    # kept for description/interpretation only, not used to drive clustering:
    descriptive_cols = ['net_depth', 'max_depth', 'path_length', 'depth_at_break']

    X = StandardScaler().fit_transform(feat_df[feature_cols])

    # ---- choose k via elbow (inertia) + silhouette ----
    ks = range(2, 13)
    inertias, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, km.labels_))

    best_k = list(ks)[int(np.argmax(sils))]
    print('Silhouette scores by k:', dict(zip(ks, [round(s, 3) for s in sils])))
    print('Chosen k (max silhouette):', best_k)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(list(ks), inertias, marker='o', color='#2a78d6')
    axes[0].set_title('Elbow (inertia)')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('inertia')
    axes[1].plot(list(ks), sils, marker='o', color='#eb6834')
    axes[1].axvline(best_k, color='#9a9990', ls='--', lw=1)
    axes[1].set_title('Silhouette score')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('avg silhouette')
    plt.tight_layout()
    plt.savefig(f'{out_prefix}route_cluster_k_selection.png', dpi=200, facecolor='white')
    print(f'Saved: {out_prefix}route_cluster_k_selection.png')

    # ---- final clustering ----
    km = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(X)
    feat_df['cluster'] = km.labels_

    print()
    print('Cluster sizes:')
    print(feat_df['cluster'].value_counts().sort_index())
    print()
    print('Cluster feature means (shape features used for clustering):')
    print(feat_df.groupby('cluster')[feature_cols].mean().round(2))
    print()
    print('Cluster feature means (descriptive only, not used for clustering):')
    print(feat_df.groupby('cluster')[descriptive_cols].mean().round(2))

    # ---- visualize: sample routes + average path per cluster ----
    n_cols = 3
    n_rows = math.ceil(best_k / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 6 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    rng = np.random.default_rng(0)
    for c in range(best_k):
        ax = axes[c]
        sub = feat_df[feat_df['cluster'] == c]
        ymax = 5
        sample_idx = rng.choice(sub.index, size=min(40, len(sub)), replace=False)
        paths = []
        for idx in sample_idx:
            row = feat_df.loc[idx]
            rx, ry = resampled[(row['game_id'], row['esb_id'], row['route_id'])]
            if (rx[-1] - rx[0]) < 0:
                rx = -rx
            rx, ry = rx - rx[0], ry - ry[0]  # recenter every sample route to a shared start point
            paths.append((rx, ry))
            ymax = max(ymax, ry.max())

        draw_field(ax, ymin=-5, ymax=ymax + 3)
        for rx, ry in paths:
            ax.plot(rx, ry, color='#9aa0a6', alpha=0.35, lw=1, zorder=3)

        avg_rx = np.mean([p[0] - p[0][0] for p in paths], axis=0)
        avg_ry = np.mean([p[1] - p[1][0] for p in paths], axis=0)
        ax.plot(avg_rx, avg_ry, color='#ffb300', lw=3, zorder=5)
        ax.scatter([0], [0], color='#2f6bff', s=30, zorder=6)

        ax.set_title(f'Cluster {c}  (n={len(sub)})', color='black', fontsize=11)

    for c in range(best_k, len(axes)):
        axes[c].axis('off')

    plt.tight_layout()
    plt.savefig(f'{out_prefix}route_clusters.png', dpi=200, facecolor='white')
    print(f'Saved: {out_prefix}route_clusters.png')

    feat_df.to_csv(f'{out_prefix}route_clusters.csv', index=False)
    print(f'Saved: {out_prefix}route_clusters.csv')


if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'route_coords.csv'
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else ''
    main(csv_path, out_prefix)
