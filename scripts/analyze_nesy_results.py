#!/usr/bin/env python3
"""
Statistical analysis for NeSy submission.

Reads per-seed results.json from <base_dir>/<env>/<method>/run_<seed>/
and produces, for each environment:

  1. Per-method stats: mean, std, median, IQR, min, max, bootstrap 95% CI
     (separate rows for reward, viol_rate, mod_rate).
  2. All-pairs comparisons: Welch's t-test and Mann-Whitney U on reward and
     viol_rate, with Benjamini-Hochberg FDR correction across the comparison
     family.

Outputs:
  - <base_dir>/<env_safe>_per_method_stats.csv
  - <base_dir>/<env_safe>_pairwise_reward.csv
  - <base_dir>/<env_safe>_pairwise_viol_rate.csv
  - Printed tables to stdout

Usage:
  python scripts/analyze_nesy_results.py --env CartPole-v1
  python scripts/analyze_nesy_results.py --env CartPole-v1 --base_dir results/nesy_experiments
"""

import argparse
import csv
import json
import os
from glob import glob
from itertools import combinations

import numpy as np
from scipy import stats


METHOD_ORDER = [
    'ppo_unshielded',
    'ppo_reward_shaping',
    'ppo_semantic_loss',
    'ppo_action_mask',
    'ppo_preshield_soft',
    'ppo_preshield_hard',
    'ppo_layer_soft',
    'ppo_layer_hard',
    'cppo',
]

METHOD_DISPLAY = {
    'ppo_unshielded': 'PPO (Unshielded)',
    'ppo_reward_shaping': 'PPO + Reward Shaping',
    'ppo_semantic_loss': 'PPO + Semantic Loss',
    'ppo_action_mask': 'PPO + Action Mask',
    'ppo_preshield_soft': 'PPO + Pre-emptive (Soft)',
    'ppo_preshield_hard': 'PPO + Pre-emptive (Hard)',
    'ppo_layer_soft': 'PPO + Layer (Soft)',
    'ppo_layer_hard': 'PPO + Layer (Hard)',
    'cppo': 'CMDP',
}

METRICS = [
    ('avg_reward', 'reward'),
    ('avg_violations_per_step', 'viol_rate'),
    ('avg_shield_mod_rate', 'mod_rate'),
]


def load_method_results(method_dir):
    """Returns dict[metric_key] -> list of per-seed values."""
    out = {key: [] for key, _ in METRICS}
    files = sorted(glob(os.path.join(method_dir, 'run_*', 'results.json')))
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        for key, _ in METRICS:
            if key in d:
                out[key].append(float(d[key]))
    return out


def bootstrap_ci(values, n_resamples=10000, alpha=0.05, statistic=np.mean, seed=0):
    """Percentile-method bootstrap CI."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    samples = arr[idx]
    stat_vals = statistic(samples, axis=1)
    lo, hi = np.quantile(stat_vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def per_method_stats(values):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return None
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    ci_lo, ci_hi = bootstrap_ci(arr)
    return {
        'n': n,
        'mean': float(arr.mean()),
        'std': float(arr.std(ddof=1)) if n > 1 else float('nan'),
        'median': float(np.median(arr)),
        'iqr': float(q3 - q1),
        'q1': float(q1),
        'q3': float(q3),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'ci_lo_95': ci_lo,
        'ci_hi_95': ci_hi,
    }


def benjamini_hochberg(pvals):
    """Return BH-FDR adjusted p-values for a family of raw p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    # Working backwards from largest p-value, take the min of (p * n / rank)
    factors = n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate((ranked * factors)[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def pairwise_tests(by_method, metric_key):
    methods = [m for m in METHOD_ORDER if m in by_method and by_method[m][metric_key]]
    rows = []
    for a, b in combinations(methods, 2):
        x = np.asarray(by_method[a][metric_key], dtype=float)
        y = np.asarray(by_method[b][metric_key], dtype=float)
        if len(x) < 2 or len(y) < 2:
            continue
        # Welch's t-test: unequal variances
        t_stat, t_p = stats.ttest_ind(x, y, equal_var=False)
        # Mann-Whitney U: non-parametric, robust to non-normality
        u_stat, u_p = stats.mannwhitneyu(x, y, alternative='two-sided')
        rows.append({
            'a': a,
            'b': b,
            'mean_a': float(x.mean()),
            'mean_b': float(y.mean()),
            'median_a': float(np.median(x)),
            'median_b': float(np.median(y)),
            't_stat': float(t_stat),
            't_p_raw': float(t_p),
            'u_stat': float(u_stat),
            'u_p_raw': float(u_p),
        })
    if rows:
        t_adj = benjamini_hochberg([r['t_p_raw'] for r in rows])
        u_adj = benjamini_hochberg([r['u_p_raw'] for r in rows])
        for i, r in enumerate(rows):
            r['t_p_fdr'] = float(t_adj[i])
            r['u_p_fdr'] = float(u_adj[i])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base_dir', default='results/nesy_experiments',
                        help='Root results directory (default: results/nesy_experiments)')
    parser.add_argument('--env', default='CartPole-v1',
                        help='Environment to analyse (e.g. CartPole-v1, ALE/Seaquest-v5)')
    args = parser.parse_args()

    env_safe = args.env.replace('/', '_')
    env_dir = os.path.join(args.base_dir, env_safe)
    if not os.path.isdir(env_dir):
        raise SystemExit(f'env dir not found: {env_dir}')

    by_method = {}
    for m in METHOD_ORDER:
        method_dir = os.path.join(env_dir, m)
        if not os.path.isdir(method_dir):
            continue
        results = load_method_results(method_dir)
        if results['avg_reward']:
            by_method[m] = results

    if not by_method:
        raise SystemExit(f'no method results found under {env_dir}')

    # ------------------------------------------------------------------
    # Per-method stats (printed + CSV)
    # ------------------------------------------------------------------
    stats_rows = []
    print(f"\n=== {args.env}: per-method stats (reward) ===\n")
    print(f"{'method':<26s} {'n':>3} {'mean':>8} {'std':>8} "
          f"{'median':>8} {'IQR':>8} {'CI_95_lo':>10} {'CI_95_hi':>10}")
    print('-' * 92)
    for m in METHOD_ORDER:
        if m not in by_method:
            continue
        for metric_key, metric_label in METRICS:
            s = per_method_stats(by_method[m][metric_key])
            if s is None:
                continue
            if metric_label == 'reward':
                print(f"{METHOD_DISPLAY[m]:<26s} {s['n']:>3} "
                      f"{s['mean']:>8.2f} {s['std']:>8.2f} "
                      f"{s['median']:>8.2f} {s['iqr']:>8.2f} "
                      f"{s['ci_lo_95']:>10.2f} {s['ci_hi_95']:>10.2f}")
            stats_rows.append({
                'method': m,
                'display_name': METHOD_DISPLAY[m],
                'metric': metric_label,
                **s,
            })

    print(f"\n=== {args.env}: per-method stats (viol_rate) ===\n")
    print(f"{'method':<26s} {'n':>3} {'mean':>8} {'std':>8} "
          f"{'median':>8} {'IQR':>8} {'CI_95_lo':>10} {'CI_95_hi':>10}")
    print('-' * 92)
    for row in stats_rows:
        if row['metric'] == 'viol_rate':
            print(f"{row['display_name']:<26s} {row['n']:>3} "
                  f"{row['mean']:>8.4f} {row['std']:>8.4f} "
                  f"{row['median']:>8.4f} {row['iqr']:>8.4f} "
                  f"{row['ci_lo_95']:>10.4f} {row['ci_hi_95']:>10.4f}")

    out_per_method = os.path.join(args.base_dir, f'{env_safe}_per_method_stats.csv')
    if stats_rows:
        with open(out_per_method, 'w', newline='') as f:
            fieldnames = ['method', 'display_name', 'metric', 'n', 'mean', 'std',
                          'median', 'iqr', 'q1', 'q3', 'min', 'max',
                          'ci_lo_95', 'ci_hi_95']
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(stats_rows)
        print(f"\nWrote per-method stats to {out_per_method}")

    # ------------------------------------------------------------------
    # Pairwise tests
    # ------------------------------------------------------------------
    for metric_key, metric_label in [('avg_reward', 'reward'),
                                       ('avg_violations_per_step', 'viol_rate')]:
        rows = pairwise_tests(by_method, metric_key)
        if not rows:
            continue
        print(f"\n=== {args.env}: pairwise tests on {metric_label} (FDR-corrected) ===")
        print(f"{'a':<26s} {'b':<26s} {'mean_a':>9} {'mean_b':>9} "
              f"{'Welch_p':>9} {'Welch_fdr':>10} {'MWU_p':>9} {'MWU_fdr':>10}")
        print('-' * 122)
        fmt = '.4f' if metric_label == 'viol_rate' else '.2f'
        for r in rows:
            sig_t = '*' if r['t_p_fdr'] < 0.05 else ' '
            print(f"{METHOD_DISPLAY[r['a']]:<26s} {METHOD_DISPLAY[r['b']]:<26s} "
                  f"{r['mean_a']:>9{fmt}} {r['mean_b']:>9{fmt}} "
                  f"{r['t_p_raw']:>9.4f} {r['t_p_fdr']:>10.4f}{sig_t} "
                  f"{r['u_p_raw']:>9.4f} {r['u_p_fdr']:>10.4f}")

        out_pairwise = os.path.join(args.base_dir, f'{env_safe}_pairwise_{metric_label}.csv')
        with open(out_pairwise, 'w', newline='') as f:
            fieldnames = ['a', 'b', 'mean_a', 'mean_b', 'median_a', 'median_b',
                          't_stat', 't_p_raw', 't_p_fdr',
                          'u_stat', 'u_p_raw', 'u_p_fdr']
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in fieldnames})
        print(f"Wrote pairwise tests to {out_pairwise}")


if __name__ == '__main__':
    main()
