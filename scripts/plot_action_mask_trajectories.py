#!/usr/bin/env python3
"""
Per-seed training trajectory plot: action_mask vs preshield_soft.

The headline figure for the gradient-signal contribution. Shows:
  - action_mask: each seed peaks and then collapses (weak teaching signal)
  - preshield_soft: each seed climbs and plateaus (gradient through shielded
    distribution teaches the policy)

Reads per-seed train_metrics_run<seed>.csv from
<base_dir>/<env>/{ppo_action_mask, ppo_preshield_soft}/run_<seed>/.
"""

import argparse
import os
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_COLORS = {
    'ppo_action_mask': '#d62728',   # red
    'ppo_preshield_soft': '#1f77b4',  # blue
}

METHOD_DISPLAY = {
    'ppo_action_mask': 'PPO + Action Mask',
    'ppo_preshield_soft': 'PPO + Pre-emptive (Soft)',
}


def load_seeds(method_dir):
    files = sorted(glob(os.path.join(method_dir, 'run_*', 'train_metrics_run*.csv')))
    seeds = []
    for f in files:
        seed = int(os.path.basename(os.path.dirname(f)).removeprefix('run_'))
        df = pd.read_csv(f)
        seeds.append((seed, df))
    return seeds


def smoothed(values, window=15):
    if len(values) < window:
        return values
    return pd.Series(values).rolling(window, min_periods=1, center=True).mean().values


def plot_pair(env_dir, env_display, out_path, window=15):
    methods = ['ppo_action_mask', 'ppo_preshield_soft']
    data = {m: load_seeds(os.path.join(env_dir, m)) for m in methods}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, m in zip(axes, methods):
        color = METHOD_COLORS[m]
        seeds = data[m]
        if not seeds:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
            continue

        # Per-seed faded curves (smoothed)
        for seed, df in seeds:
            x = df['episode'].values
            y_smooth = smoothed(df['reward'].values, window=window)
            ax.plot(x, y_smooth, color=color, alpha=0.30, linewidth=1.2)

        # Build a common x-grid for median across seeds (interpolate to longest episode count)
        max_ep = max(df['episode'].max() for _, df in seeds)
        grid = np.arange(1, max_ep + 1)
        per_seed_on_grid = []
        for seed, df in seeds:
            x = df['episode'].values
            y = smoothed(df['reward'].values, window=window)
            y_grid = np.interp(grid, x, y, left=np.nan, right=np.nan)
            per_seed_on_grid.append(y_grid)
        per_seed_on_grid = np.array(per_seed_on_grid)
        # Median across seeds where data exists
        with np.errstate(invalid='ignore'):
            median = np.nanmedian(per_seed_on_grid, axis=0)
            q1 = np.nanpercentile(per_seed_on_grid, 25, axis=0)
            q3 = np.nanpercentile(per_seed_on_grid, 75, axis=0)

        ax.fill_between(grid, q1, q3, color=color, alpha=0.18, linewidth=0)
        ax.plot(grid, median, color=color, linewidth=2.6,
                label=f"{METHOD_DISPLAY[m]} (median across {len(seeds)} seeds)")

        ax.set_title(METHOD_DISPLAY[m], fontsize=13)
        ax.set_xlabel('Training episode', fontsize=12)
        ax.grid(alpha=0.25, linestyle='--')
        ax.legend(loc='lower right', fontsize=10, framealpha=0.85)

    axes[0].set_ylabel('Reward (smoothed)', fontsize=12)
    fig.suptitle(f'{env_display}: per-seed training trajectories', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'wrote {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='results/nesy_experiments')
    parser.add_argument('--env', default='CartPole-v1')
    parser.add_argument('--window', type=int, default=15,
                        help='rolling-mean window for smoothing (default 15 episodes)')
    args = parser.parse_args()

    env_safe = args.env.replace('/', '_')
    env_dir = os.path.join(args.base_dir, env_safe)
    if not os.path.isdir(env_dir):
        raise SystemExit(f'env dir not found: {env_dir}')

    plots_dir = os.path.join(env_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    out = os.path.join(plots_dir, f'{env_safe}_action_mask_vs_preshield_soft_trajectories.png')
    plot_pair(env_dir, args.env, out, window=args.window)


if __name__ == '__main__':
    main()
