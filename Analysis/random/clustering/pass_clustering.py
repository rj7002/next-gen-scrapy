import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

PASS_CSV = '/Users/ryan/Downloads/FF-Python-Analytics-master/ALL DATA/pass_locations.csv'


def draw_field(ax, ymin=-15, ymax=60):
    ymin, ymax = int(math.floor(ymin)), int(math.ceil(ymax))
    ax.set_facecolor('black')
    for y in range(ymin - ymin % 5, ymax + 1, 5):
        ax.axhline(y, color='white', alpha=0.18, lw=0.7)
    ax.axhline(0, color='#ffffff', lw=1.6, alpha=0.7, zorder=2)
    for x in (-26.67, 26.67):
        ax.axvline(x, color='white', lw=1.5)
    for x in (-3.08, 3.08):
        ax.axvline(x, color='white', alpha=0.18, lw=0.7, ls=':')
    ax.set_xlim(-30, 30)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('x (yards)')
    ax.set_ylabel('yards past line of scrimmage')


def load():
    df = pd.read_csv(PASS_CSV)
    df = df[df['located'] == True].dropna(subset=['x_coord', 'y_coord'])
    return df


def plot_density(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(17, 11))

    draw_field(axes[0], ymax=60)
    hb = axes[0].hexbin(df['x_coord'], df['y_coord'], gridsize=60, cmap='inferno',
                        bins='log', mincnt=1, extent=(-30, 30, -15, 60), zorder=3)
    axes[0].set_title(f'Pass location density (n={len(df):,})')
    fig.colorbar(hb, ax=axes[0], shrink=0.6, label='log10(passes)')

    # marginal distributions - the real test of "are there distinct modes?"
    axes[1].hist(df['y_coord'], bins=140, range=(-15, 60), color='#2a78d6', alpha=0.85)
    axes[1].set_xlabel('yards past line of scrimmage (depth)')
    axes[1].set_ylabel('passes')
    axes[1].set_title('Depth distribution')
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out, dpi=180, facecolor='white')
    print('Saved:', out)


def select_k(X, out):
    ks = range(2, 13)
    inertias, sils, bics = [], [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        inertias.append(km.inertia_)
        # silhouette on 130k points is O(n^2); subsample
        sils.append(silhouette_score(X, km.labels_, sample_size=10000, random_state=0))
        gm = GaussianMixture(n_components=k, covariance_type='full', random_state=0).fit(X)
        bics.append(gm.bic(X))
        print(f'  k={k:2d}  inertia={km.inertia_:12.0f}  silhouette={sils[-1]:.3f}  gmm_bic={bics[-1]:12.0f}')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(list(ks), inertias, marker='o', color='#2a78d6')
    axes[0].set_title('KMeans elbow (inertia)')
    axes[1].plot(list(ks), sils, marker='o', color='#eb6834')
    axes[1].set_title('Silhouette (higher = more separated)')
    axes[2].plot(list(ks), bics, marker='o', color='#1baf7a')
    axes[2].set_title('GMM BIC (lower = better fit)')
    for ax in axes:
        ax.set_xlabel('k')
        ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=180, facecolor='white')
    print('Saved:', out)

    return list(ks), inertias, sils, bics


ZONE_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7', '#e34948', '#008300']


def fit_zones(df, k, out_plot, out_csv):
    """GMM soft-assigns each pass to a zone. Reported as zones, not 'clusters' - the underlying
    density is continuous, so these are partitions of a smooth surface, not separated groups."""
    X = df[['x_coord', 'y_coord']].to_numpy()
    gm = GaussianMixture(n_components=k, covariance_type='full', random_state=0).fit(X)
    df = df.copy()
    df['zone'] = gm.predict(X)
    df['zone_conf'] = gm.predict_proba(X).max(axis=1)

    # order zones by mean depth so numbering is interpretable
    order = df.groupby('zone')['y_coord'].mean().sort_values().index
    remap = {old: new for new, old in enumerate(order)}
    df['zone'] = df['zone'].map(remap)

    summary = df.groupby('zone').agg(
        n=('zone', 'size'),
        mean_x=('x_coord', 'mean'),
        mean_depth=('y_coord', 'mean'),
        comp_pct=('pass_type', lambda s: (s == 'COMPLETE').mean() * 100),
        td_pct=('pass_type', lambda s: (s == 'TOUCHDOWN').mean() * 100),
        int_pct=('pass_type', lambda s: (s == 'INTERCEPTION').mean() * 100),
        mean_conf=('zone_conf', 'mean'),
    ).round(2)
    summary['share_pct'] = (summary['n'] / len(df) * 100).round(1)
    print()
    print(f'GMM zones (k={k}):')
    print(summary)

    fig, ax = plt.subplots(figsize=(11, 13))
    draw_field(ax, ymax=60)
    plot_df = df.sample(min(40000, len(df)), random_state=0)
    for z in sorted(df['zone'].unique()):
        d = plot_df[plot_df['zone'] == z]
        ax.scatter(d['x_coord'], d['y_coord'], s=3, alpha=0.30,
                   color=ZONE_COLORS[z % len(ZONE_COLORS)], zorder=3,
                   label=f'zone {z}: {summary.loc[z, "share_pct"]}% of passes, '
                         f'{summary.loc[z, "comp_pct"]:.0f}% comp')
    ax.set_title(f'Pass locations by GMM zone (k={k}, {len(plot_df):,} of {len(df):,} plotted)')
    leg = ax.legend(loc='upper right', fontsize=8, facecolor='white', framealpha=0.92, markerscale=4)
    for h in leg.legend_handles:
        h.set_alpha(1)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=180, facecolor='white')
    print('Saved:', out_plot)

    df.to_csv(out_csv, index=False)
    print('Saved:', out_csv)
    return df


if __name__ == '__main__':
    df = load()
    if 'zones-only' not in sys.argv:
        plot_density(df, 'pass_density.png')
        X = df[['x_coord', 'y_coord']].to_numpy()
        print('Model selection sweep:')
        select_k(X, 'pass_cluster_k_selection.png')
    # BIC's last meaningful drop is at k=6; past that it's noise-level improvement
    fit_zones(df, 6, 'pass_zones.png', 'pass_zones.csv')
