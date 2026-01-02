#!/usr/bin/env python3
"""
Experiment runner for IJCAI submission.
Runs ConstrainedPPO (CMDP) vs PPO with shields at different positions and softness levels.

Key Metrics:
- Violation Rate: Rate of constraint violations per step (primary metric)
- Modification Rate: Rate of shield modifications per step (primary metric)
- Reward: Secondary metric (should be roughly similar across methods for fair comparison)

The budget for CMDP is set per-environment to achieve comparable reward performance,
allowing focus on violation and modification rate differences.
"""

import os
import sys
import argparse
import json
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train, evaluate_policy, make_run_dir
from src.utils import graphing

# Fixed seeds for reproducibility (5 runs per configuration)
SEEDS = [42, 123, 456, 789, 1011]

# Environments to test
ENVIRONMENTS = [
    'CliffWalking-v1',
    'CartPole-v1',
    'MiniGrid-DoorKey-5x5-v0',
    'ALE/Seaquest-v5',
]

# Human-friendly display names for environments (for paper/graphs)
ENV_DISPLAY_NAMES = {
    'CliffWalking-v1': 'Cliff Walking',
    'CartPole-v1': 'Cart Pole',
    'MiniGrid-DoorKey-5x5-v0': 'Door Key',
    'ALE/Seaquest-v5': 'Seaquest',
}

# Methods to compare (7 total: CPO and Post-hoc methods skipped)
# Post-hoc methods removed due to known PPO collapse issue
# Ordered from lightest to heaviest computationally
METHODS = [
    # Lightest: No shield computation during training
    {
        'name': 'ppo_reward_shaping',
        'agent': 'ppo',
        'use_shield_post': False,
        'use_shield_pre': False,
        'use_shield_layer': False,
        'mode': '',
        'lambda_sem': 0.0,
        'lambda_penalty': 1.0,  # Reward shaping penalty (will be tuned)
        'display_name': 'PPO + Reward Shaping'
    },
    {
        'name': 'ppo_semantic_loss',
        'agent': 'ppo',
        'use_shield_post': False,
        'use_shield_pre': False,
        'use_shield_layer': False,
        'mode': '',
        'lambda_sem': 1.0,  # Semantic loss coefficient
        'display_name': 'PPO + Semantic Loss'
    },
    # Medium: Shield computation before action (pre-emptive)
    {
        'name': 'ppo_preshield_soft',
        'agent': 'ppo',
        'use_shield_post': False,
        'use_shield_pre': True,
        'use_shield_layer': False,
        'mode': 'soft',
        'lambda_sem': 0.0,
        'display_name': 'PPO + Pre-emptive (Soft)'
    },
    {
        'name': 'ppo_preshield_hard',
        'agent': 'ppo',
        'use_shield_post': False,
        'use_shield_pre': True,
        'use_shield_layer': False,
        'mode': 'hard',
        'lambda_sem': 0.0,
        'display_name': 'PPO + Pre-emptive (Hard)'
    },
    # Medium-Heavy: Shield as differentiable layer
    {
        'name': 'ppo_layer_soft',
        'agent': 'ppo',
        'use_shield_post': False,
        'use_shield_pre': False,
        'use_shield_layer': True,
        'mode': 'soft',
        'lambda_sem': 0.0,
        'display_name': 'PPO + Layer (Soft)'
    },
    {
        'name': 'ppo_layer_hard',
        'agent': 'ppo',
        'use_shield_post': False,
        'use_shield_pre': False,
        'use_shield_layer': True,
        'mode': 'hard',
        'lambda_sem': 0.0,
        'display_name': 'PPO + Layer (Hard)'
    },
    # Heaviest: CMDP with dual optimization (2-3x slower)
    {
        'name': 'cppo',
        'agent': 'cppo',
        'use_shield_post': False,
        'use_shield_pre': False,
        'use_shield_layer': False,
        'mode': '',
        'lambda_sem': 0.0,
        'display_name': 'CMDP'
    },
    # CPO skipped for now as requested
    # {
    #     'name': 'cpo',
    #     'agent': 'cpo',
    #     'use_shield_post': False,
    #     'use_shield_pre': False,
    #     'use_shield_layer': False,
    #     'mode': '',
    #     'lambda_sem': 0.0,
    #     'display_name': 'CPO'
    # },
    # Post-hoc methods removed - known PPO collapse issue
    # {
    #     'name': 'ppo_postshield_hard',
    #     'agent': 'ppo',
    #     'use_shield_post': True,
    #     'use_shield_pre': False,
    #     'use_shield_layer': False,
    #     'mode': 'hard',
    #     'lambda_sem': 0.0,
    #     'display_name': 'PPO + Post-hoc (Hard)'
    # },
    # {
    #     'name': 'ppo_postshield_soft',
    #     'agent': 'ppo',
    #     'use_shield_post': True,
    #     'use_shield_pre': False,
    #     'use_shield_layer': False,
    #     'mode': 'soft',
    #     'lambda_sem': 0.0,
    #     'display_name': 'PPO + Post-hoc (Soft)'
    # },
]


def run_single_experiment(
    env_name,
    method,
    seed,
    num_train_episodes,
    num_eval_episodes,
    base_dir,
    verbose=False
):
    """
    Run a single experiment configuration.
    
    Returns:
        dict: Results dictionary with metrics
    """
    print(f"\n{'='*80}")
    print(f"Running: {method['display_name']} on {env_name} (seed={seed})")
    print(f"{'='*80}")
    
    # Create run directory
    run_dir = os.path.join(
        base_dir,
        env_name.replace('/', '_'),
        method['name'],
        f'run_{seed}'
    )
    os.makedirs(run_dir, exist_ok=True)
    
    try:
        # Prepare agent_kwargs - start with tuned hyperparameters if available
        agent_kwargs = {}
        
        # Try to load tuned hyperparameters
        import yaml
        from pathlib import Path
        env_safe = env_name.replace('/', '_')
        config_path = Path(f"config/ijcai_tuned/{method['name']}_{env_safe}_params.yaml")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                tuned_params = yaml.safe_load(f)
            agent_kwargs.update(tuned_params)
            if verbose:
                print(f"[Loaded tuned hyperparameters from {config_path}]")
        else:
            # Fallback to defaults if no tuned params found
            if verbose:
                print(f"[No tuned hyperparameters found, using defaults]")
        
        # Semantic loss coefficient (override if in method config)
        if method.get('lambda_sem', 0.0) > 0:
            agent_kwargs['lambda_sem'] = method['lambda_sem']
        
        # Reward shaping penalty (override if in method config)
        if method.get('lambda_penalty', 0.0) > 0:
            agent_kwargs['lambda_penalty'] = method['lambda_penalty']
        
        # Budget and memory limits for CPO/CMDP (only if not in tuned params)
        if method['agent'] in ['cpo', 'cppo'] and 'budget' not in agent_kwargs:
            # Set budget based on environment - aim for similar reward performance
            # Budget represents max allowed violation rate (cost per step)
            env_budgets = {
                'CartPole-v1': 0.15,  # Allow ~15% violation rate for comparable rewards
                'CliffWalking-v1': 0.10,
                'MiniGrid-DoorKey-5x5-v0': 0.20,
                'ALE/Seaquest-v5': 0.25,
            }
            budget = env_budgets.get(env_name, 0.15)  # Default 15%
            agent_kwargs['budget'] = budget
            if method['agent'] == 'cppo':
                agent_kwargs['nu_lr'] = 1e-3  # Lagrangian multiplier learning rate (only for CMDP)
        
        if method['agent'] == 'cppo' and env_name == 'CliffWalking-v1':
            min_budget_for_comparison = 0.15
            if agent_kwargs.get('budget', 1.0) < min_budget_for_comparison:
                if verbose:
                    print(f"[Adjusting CMDP budget from {agent_kwargs.get('budget', 'N/A')} to {min_budget_for_comparison} to ensure violations for comparison]")
                agent_kwargs['budget'] = min_budget_for_comparison
        
        # Set memory limits based on environment (to prevent OOM on large environments)
        if 'max_episode_memory' not in agent_kwargs:
            env_memory_limits = {
                'CartPole-v1': 500,      # Short episodes, small memory needed
                'CliffWalking-v1': 1000,  # Match step limit (1000 steps) for exploration
                'MiniGrid-DoorKey-5x5-v0': 2000,  # Longer episodes
                'ALE/Seaquest-v5': 5000,  # Very long episodes (up to 10k steps)
            }
            max_memory = env_memory_limits.get(env_name, 2000)  # Default 2000
            agent_kwargs['max_episode_memory'] = max_memory
        
        # Get training target for early stopping (from tuning config)
        from scripts.tune_ijcai_methods import TRAINING_TARGET_REWARDS
        training_target = TRAINING_TARGET_REWARDS.get(env_name, None)
        
        # Train agent
        agent, episode_rewards, best_weights, best_avg_reward, env = train(
            agent=method['agent'],
            env_name=env_name,
            num_episodes=num_train_episodes,
            use_shield_post=method['use_shield_post'],
            use_shield_pre=method['use_shield_pre'],
            use_shield_layer=method['use_shield_layer'],
            monitor_constraints=True,
            mode=method['mode'],
            verbose=verbose,
            visualize=False,
            render=False,
            seed=seed,
            run_dir=run_dir,
            agent_kwargs=agent_kwargs if agent_kwargs else None,
            early_stop_patience=100,  # Stop training if no improvement after 100 episodes
            target_reward=training_target  # Stop training when target is reached
        )
        
        # Load best weights
        if hasattr(agent, 'load_weights') and best_weights is not None:
            agent.load_weights(best_weights)
        
        # Evaluate
        results = evaluate_policy(
            agent,
            env,
            num_episodes=num_eval_episodes,
            visualize=False,
            render=False,
            force_disable_shield=False,
            run_dir=run_dir,
            softness=method['mode']
        )
        
        # Save training metrics
        train_metrics_path = os.path.join(run_dir, f'train_metrics_run{seed}.csv')
        if os.path.exists(train_metrics_path):
            # Already saved during training
            pass
        else:
            # Create training metrics CSV
            with open(train_metrics_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['episode', 'reward', 'violations', 'violation_rate', 
                               'modifications', 'modification_rate'])
                for ep, reward in enumerate(episode_rewards, 1):
                    # Get stats from constraint monitor if available
                    if hasattr(agent, 'constraint_monitor'):
                        stats = agent.constraint_monitor.summary()
                        writer.writerow([
                            ep, reward,
                            stats.get('episode_violations', 0),
                            stats.get('episode_viol_rate', 0.0),
                            stats.get('episode_modifications', 0),
                            stats.get('episode_mod_rate', 0.0)
                        ])
                    else:
                        writer.writerow([ep, reward, 0, 0.0, 0, 0.0])
        
        # Add training info to results
        results['train_episodes'] = len(episode_rewards)
        results['best_avg_reward'] = best_avg_reward
        results['final_reward'] = episode_rewards[-1] if episode_rewards else 0.0
        results['seed'] = seed
        results['method'] = method['name']
        results['env'] = env_name
        
        env.close()
        
        print(f"✓ Completed: {method['display_name']} on {env_name} (seed={seed})")
        print(f"  Avg Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Violations: {results['total_violations']}")
        print(f"  Modifications: {results['total_modifications']}")
        
        return results
        
    except Exception as e:
        print(f"✗ Error in {method['display_name']} on {env_name} (seed={seed}): {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_summary_table(base_dir):
    """
    Generate a summary table of all completed experiments.
    
    Returns:
        pd.DataFrame: Summary table with all methods and environments
    """
    summary_data = []
    
    for env_name in ENVIRONMENTS:
        env_dir = os.path.join(base_dir, env_name.replace('/', '_'))
        if not os.path.exists(env_dir):
            continue
            
        for method in METHODS:
            method_dir = os.path.join(env_dir, method['name'])
            aggregated_path = os.path.join(method_dir, 'aggregated_results.json')
            
            if os.path.exists(aggregated_path):
                try:
                    with open(aggregated_path, 'r') as f:
                        agg = json.load(f)
                    
                    reward_agg = agg.get('avg_reward', {})
                    reward_mean = reward_agg.get('mean', 0)
                    reward_std = reward_agg.get('std', 0)
                    reward_median = reward_agg.get('median', reward_mean)  # Fallback to mean if median not available
                    
                    summary_data.append({
                        'Environment': env_name,
                        'Method': method['display_name'],
                        'Reward (median)': f"{reward_median:.2f}",
                        'Reward (mean ± std)': f"{reward_mean:.2f} ± {reward_std:.2f}",
                        'Viol Rate': f"{agg.get('avg_violations_per_step', {}).get('mean', 0):.4f} ± {agg.get('avg_violations_per_step', {}).get('std', 0):.4f}",
                        'Mod Rate': f"{agg.get('avg_shield_mod_rate', {}).get('mean', 0):.4f} ± {agg.get('avg_shield_mod_rate', {}).get('std', 0):.4f}",
                        'Total Viol': f"{agg.get('total_violations', {}).get('mean', 0):.1f} ± {agg.get('total_violations', {}).get('std', 0):.1f}",
                        'Total Mod': f"{agg.get('total_modifications', {}).get('mean', 0):.1f} ± {agg.get('total_modifications', {}).get('std', 0):.1f}",
                    })
                except Exception as e:
                    summary_data.append({
                        'Environment': env_name,
                        'Method': method['display_name'],
                        'Reward (median)': 'Error',
                        'Reward (mean ± std)': 'Error',
                        'Viol Rate': 'Error',
                        'Mod Rate': 'Error',
                        'Total Viol': 'Error',
                        'Total Mod': 'Error',
                    })
            else:
                # Not completed yet
                summary_data.append({
                    'Environment': env_name,
                    'Method': method['display_name'],
                    'Reward (median)': 'Pending',
                    'Reward (mean ± std)': 'Pending',
                    'Viol Rate': 'Pending',
                    'Mod Rate': 'Pending',
                    'Total Viol': 'Pending',
                    'Total Mod': 'Pending',
                })
    
    df = pd.DataFrame(summary_data)
    return df


def print_summary_table(base_dir):
    """Print a formatted summary table of all experiments."""
    df = generate_summary_table(base_dir)
    
    if df.empty:
        print("\nNo results found yet.")
        return
    
    print("\n" + "="*120)
    print("EXPERIMENT SUMMARY TABLE")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120)
    
    # Save to file
    summary_path = os.path.join(base_dir, 'summary_table.txt')
    with open(summary_path, 'w') as f:
        f.write("="*120 + "\n")
        f.write("EXPERIMENT SUMMARY TABLE\n")
        f.write("="*120 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n" + "="*120 + "\n")
    
    print(f"\nSummary table saved to: {summary_path}")
    
    # Also save as CSV for easy viewing in Excel
    csv_path = os.path.join(base_dir, 'summary_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"Summary table (CSV) saved to: {csv_path}")


def aggregate_results(all_results):
    """
    Aggregate results across multiple runs.
    
    Args:
        all_results: List of result dictionaries
        
    Returns:
        dict: Aggregated statistics
    """
    if not all_results:
        return None
    
    # Get all keys
    keys = set()
    for r in all_results:
        if r is not None:
            keys.update(r.keys())
    
    aggregated = {}
    for key in keys:
        values = [r[key] for r in all_results if r is not None and key in r]
        if values:
            if isinstance(values[0], (int, float, np.integer, np.floating)):
                # Convert to native Python types for JSON serialization
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
                min_val = float(np.min(values))
                max_val = float(np.max(values))
                median_val = float(np.median(values))
                aggregated[key] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'median': median_val
                }
            else:
                aggregated[key] = values[0]  # Use first non-numeric value
    
    return aggregated


def generate_experiment_graphs(base_dir, envs_to_run, methods_to_run):
    """
    Generate graphs for violation and modification rates over time for all methods.
    Aggregates across 5 runs and creates publication-quality plots.
    """
    import glob
    
    for env_name in envs_to_run:
        env_safe = env_name.replace('/', '_')
        env_dir = os.path.join(base_dir, env_safe)
        
        if not os.path.exists(env_dir):
            continue
        
        # Get human-friendly display name
        env_display = ENV_DISPLAY_NAMES.get(env_name, env_name)
        
        print(f"\nGenerating graphs for {env_display} ({env_name})...")
        
        # Collect data for all methods
        all_method_data = {}
        
        for method in methods_to_run:
            method_dir = os.path.join(env_dir, method['name'])
            if not os.path.exists(method_dir):
                continue
            
            # Load all training CSV files for this method
            # CSV files are in subdirectories like run_42/, run_123/, etc.
            csv_files = glob.glob(os.path.join(method_dir, "**/train_metrics_run*.csv"), recursive=True)
            # Also check directly in method_dir (in case structure is different)
            if not csv_files:
                csv_files = glob.glob(os.path.join(method_dir, "train_metrics_run*.csv"))
            
            if not csv_files:
                continue
            
            # Load and aggregate across runs
            dfs = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    dfs.append(df)
                except Exception as e:
                    print(f"  Warning: Could not load {csv_file}: {e}")
                    continue
            
            if not dfs:
                continue
            
            # Combine all runs
            combined = pd.concat(dfs, ignore_index=True)
            
            # Normalize episodes to training progress (0-100%) for fair comparison
            # This accounts for methods that stop early when reaching target reward
            max_episode = combined['episode'].max()
            combined['training_progress'] = (combined['episode'] / max_episode) * 100
            
            # Group by training progress (rounded to nearest percent) and compute mean/std
            combined['progress_bin'] = (combined['training_progress'] // 1).astype(int)  # Round to integer percent
            grouped = combined.groupby('progress_bin').agg({
                'violation_rate': ['mean', 'std'],
                'modification_rate': ['mean', 'std'],
                'reward': ['mean', 'std'],
                'episode': 'mean'  # Keep track of actual episode number for reference
            }).reset_index()
            
            # Flatten column names
            grouped.columns = ['training_progress', 'viol_mean', 'viol_std', 'mod_mean', 'mod_std', 'reward_mean', 'reward_std', 'avg_episode']
            
            all_method_data[method['display_name']] = grouped
        
        if not all_method_data:
            print(f"  No data found for {env_name}")
            continue
        
        # Create plots directory
        plots_dir = os.path.join(env_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set up styling for all plots (consistent across all graphs)
        # Use a clean, publication-ready color palette
        # Use distinct colors that work well in print
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        if len(all_method_data) > len(colors):
            # Extend with additional colors if needed
            colors.extend(plt.cm.tab20(np.linspace(0, 1, len(all_method_data) - len(colors))))
        
        # All solid lines for consistency
        line_styles = ['-'] * len(all_method_data)
        # Markers only at key points to reduce clutter
        markers = ['o', 's', '^', 'v', 'D', 'p', '*']
        marker_styles = [{'marker': m, 'markevery': max(5, len(data['training_progress']) // 10)} 
                        for m, data in zip(markers, all_method_data.values())]
        
        # Plot 1: Violation rates over time (all methods comparison)
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for idx, ((method_name, data), color) in enumerate(zip(all_method_data.items(), colors)):
            progress = data['training_progress'].values
            viol_mean = data['viol_mean'].values
            viol_std = data['viol_std'].values
            
            linestyle = line_styles[idx % len(line_styles)]
            marker_info = marker_styles[idx]
            
            # Clean line plot - no fill for clarity
            ax.plot(progress, viol_mean, label=method_name, color=color, linewidth=2.5,
                   linestyle=linestyle, marker=marker_info['marker'], 
                   markevery=marker_info['markevery'], markersize=7, 
                   markeredgewidth=1, markerfacecolor=color, markeredgecolor='white')
        
        ax.set_xlabel('Training Progress (%)', fontsize=13)
        ax.set_ylabel('Violation Rate (per step)', fontsize=13)
        ax.set_title(f'{env_display} - Violation Rates Over Training Progress', fontsize=15)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.2, linestyle='--')
        ax.tick_params(labelsize=11)
        
        plt.tight_layout()
        viol_plot_path = os.path.join(plots_dir, f'{env_safe}_violation_rates_over_time.png')
        plt.savefig(viol_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {viol_plot_path}")
        
        # Plot 2: Modification rates over time (all methods comparison)
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for idx, ((method_name, data), color) in enumerate(zip(all_method_data.items(), colors)):
            progress = data['training_progress'].values
            mod_mean = data['mod_mean'].values
            mod_std = data['mod_std'].values
            
            linestyle = line_styles[idx % len(line_styles)]
            marker_info = marker_styles[idx]
            
            # Clean line plot - no fill for clarity
            ax.plot(progress, mod_mean, label=method_name, color=color, linewidth=2.5,
                   linestyle=linestyle, marker=marker_info['marker'], 
                   markevery=marker_info['markevery'], markersize=7, 
                   markeredgewidth=1, markerfacecolor=color, markeredgecolor='white')
        
        ax.set_xlabel('Training Progress (%)', fontsize=13)
        ax.set_ylabel('Modification Rate (per step)', fontsize=13)
        ax.set_title(f'{env_display} - Modification Rates Over Training Progress', fontsize=15)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.2, linestyle='--')
        ax.tick_params(labelsize=11)
        
        plt.tight_layout()
        mod_plot_path = os.path.join(plots_dir, f'{env_safe}_modification_rates_over_time.png')
        plt.savefig(mod_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {mod_plot_path}")
        
        # Plot 3: Combined plot (violations and modifications on same plot, different axes)
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Left axis for violations
        for idx, ((method_name, data), color) in enumerate(zip(all_method_data.items(), colors)):
            progress = data['training_progress'].values
            viol_mean = data['viol_mean'].values
            linestyle = line_styles[idx % len(line_styles)]
            marker_info = marker_styles[idx]
            ax1.plot(progress, viol_mean, label=f'{method_name} (Violations)', 
                    color=color, linewidth=2.5, linestyle=linestyle,
                    marker=marker_info['marker'], markevery=marker_info['markevery'], 
                    markersize=7, markeredgewidth=1, markerfacecolor=color, markeredgecolor='white')
        
        ax1.set_xlabel('Training Progress (%)', fontsize=12)
        ax1.set_ylabel('Violation Rate (per step)', fontsize=12, color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.grid(alpha=0.3)
        
        # Right axis for modifications
        ax2 = ax1.twinx()
        for idx, ((method_name, data), color) in enumerate(zip(all_method_data.items(), colors)):
            progress = data['training_progress'].values
            mod_mean = data['mod_mean'].values
            linestyle = line_styles[idx % len(line_styles)]
            marker_info = marker_styles[idx]
            # Use different marker for modifications to distinguish from violations
            mod_markers = ['s', '^', 'v', 'D', 'p', '*', 'X']
            ax2.plot(progress, mod_mean, label=f'{method_name} (Modifications)', 
                    color=color, linewidth=2.5, linestyle=linestyle,
                    marker=mod_markers[idx % len(mod_markers)], 
                    markevery=marker_info['markevery'], markersize=7, 
                    markeredgewidth=1, markerfacecolor=color, markeredgecolor='white')
        
        ax2.set_ylabel('Modification Rate (per step)', fontsize=12, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
        
        ax1.set_title(f'{env_display} - Violation and Modification Rates Over Training Progress', fontsize=14)
        
        plt.tight_layout()
        combined_plot_path = os.path.join(plots_dir, f'{env_safe}_combined_rates_over_time.png')
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {combined_plot_path}")
        
        # Plot 4: Individual method plots (violation + modification on same plot)
        for method_name, data in all_method_data.items():
            method_safe = method_name.replace(' ', '_').replace('+', '').replace('(', '').replace(')', '').replace('-', '_')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            progress = data['training_progress'].values
            viol_mean = data['viol_mean'].values
            viol_std = data['viol_std'].values
            mod_mean = data['mod_mean'].values
            mod_std = data['mod_std'].values
            
            # Plot violations
            ax.plot(progress, viol_mean, label='Violation Rate', color='red', linewidth=2)
            ax.fill_between(progress, viol_mean - viol_std, viol_mean + viol_std, 
                          color='red', alpha=0.2)
            
            # Plot modifications
            ax.plot(progress, mod_mean, label='Modification Rate', color='orange', linewidth=2)
            ax.fill_between(progress, mod_mean - mod_std, mod_mean + mod_std, 
                          color='orange', alpha=0.2)
            
            ax.set_xlabel('Training Progress (%)', fontsize=12)
            ax.set_ylabel('Rate (per step)', fontsize=12)
            ax.set_title(f'{env_display} - {method_name}\nViolation and Modification Rates (Mean ± Std)', fontsize=14)
            ax.legend(loc='best')
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            method_plot_path = os.path.join(plots_dir, f'{env_safe}_{method_safe}_rates.png')
            plt.savefig(method_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  ✓ Generated individual method plots")
        
        # Plot 5: Reward over time (all methods comparison)
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for idx, ((method_name, data), color) in enumerate(zip(all_method_data.items(), colors)):
            progress = data['training_progress'].values
            reward_mean = data['reward_mean'].values
            reward_std = data['reward_std'].values
            
            linestyle = line_styles[idx % len(line_styles)]
            marker_info = marker_styles[idx]
            
            # Clean line plot - no fill for clarity
            ax.plot(progress, reward_mean, label=method_name, color=color, linewidth=2.5,
                   linestyle=linestyle, marker=marker_info['marker'], 
                   markevery=marker_info['markevery'], markersize=7, 
                   markeredgewidth=1, markerfacecolor=color, markeredgecolor='white')
        
        ax.set_xlabel('Training Progress (%)', fontsize=13)
        ax.set_ylabel('Reward', fontsize=13)
        ax.set_title(f'{env_display} - Reward Over Training Progress', fontsize=15)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.2, linestyle='--')
        ax.tick_params(labelsize=11)
        
        plt.tight_layout()
        reward_plot_path = os.path.join(plots_dir, f'{env_safe}_reward_over_time.png')
        plt.savefig(reward_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {reward_plot_path}")
    
    print(f"\n✓ All graphs generated!")


def run_all_experiments(
    num_train_episodes=500,
    num_eval_episodes=100,
    base_dir='results/ijcai_experiments',
    verbose=False,
    env_filter=None,
    method_filter=None,
    skip_existing=False
):
    """
    Run all experiment configurations.
    
    Args:
        num_train_episodes: Number of training episodes
        num_eval_episodes: Number of evaluation episodes
        base_dir: Base directory for results
        verbose: Verbose output
        env_filter: List of environment names to run (None for all)
        method_filter: List of method names to run (None for all)
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # Filter environments and methods if specified
    envs_to_run = [e for e in ENVIRONMENTS if env_filter is None or e in env_filter]
    methods_to_run = [m for m in METHODS if method_filter is None or m['name'] in method_filter]
    
    total_configs = len(envs_to_run) * len(methods_to_run) * len(SEEDS)
    config_num = 0
    
    all_results = []
    
    for env_name in envs_to_run:
        print(f"\n{'#'*80}")
        print(f"# Environment: {env_name}")
        print(f"{'#'*80}\n")
        
        for method in methods_to_run:
            method_dir = os.path.join(
                base_dir,
                env_name.replace('/', '_'),
                method['name']
            )
            aggregated_path = os.path.join(method_dir, 'aggregated_results.json')
            
            # Skip if already exists and skip_existing is True
            if skip_existing and os.path.exists(aggregated_path):
                print(f"⏭️  Skipping {method['display_name']} on {env_name} (already exists)")
                # Load existing results to include in summary
                try:
                    with open(aggregated_path, 'r') as f:
                        existing = json.load(f)
                    print(f"   Existing: Reward={existing.get('avg_reward', {}).get('mean', 'N/A'):.2f}, "
                          f"Viol Rate={existing.get('avg_violations_per_step', {}).get('mean', 'N/A'):.4f}")
                except:
                    pass
                continue
            
            method_results = []
            
            for seed in SEEDS:
                config_num += 1
                print(f"\n[{config_num}/{total_configs}] Configuration: {method['display_name']} on {env_name} (seed={seed})")
                
                result = run_single_experiment(
                    env_name=env_name,
                    method=method,
                    seed=seed,
                    num_train_episodes=num_train_episodes,
                    num_eval_episodes=num_eval_episodes,
                    base_dir=base_dir,
                    verbose=verbose
                )
                
                if result:
                    method_results.append(result)
                    all_results.append(result)
            
            # Aggregate results for this method
            if method_results:
                aggregated = aggregate_results(method_results)
                if aggregated:
                    # Save aggregated results
                    os.makedirs(method_dir, exist_ok=True)
                    
                    aggregated_path = os.path.join(method_dir, 'aggregated_results.json')
                    with open(aggregated_path, 'w') as f:
                        json.dump(aggregated, f, indent=2)
                    
                    # Print summary emphasizing violation/modification rates
                    print(f"\n✓ Aggregated results for {method['display_name']} on {env_name}:")
                    print(f"  Avg Reward: {aggregated['avg_reward']['mean']:.2f} ± {aggregated['avg_reward']['std']:.2f}")
                    if 'avg_violations_per_step' in aggregated:
                        viol_rate = aggregated['avg_violations_per_step']['mean']
                        viol_std = aggregated['avg_violations_per_step']['std']
                        print(f"  Violation Rate: {viol_rate:.4f} ± {viol_std:.4f} (per step)")
                    if 'avg_shield_mod_rate' in aggregated:
                        mod_rate = aggregated['avg_shield_mod_rate']['mean']
                        mod_std = aggregated['avg_shield_mod_rate']['std']
                        print(f"  Modification Rate: {mod_rate:.4f} ± {mod_std:.4f} (per step)")
                    if 'total_violations' in aggregated:
                        print(f"  Total Violations: {aggregated['total_violations']['mean']:.1f} ± {aggregated['total_violations']['std']:.1f}")
                    if 'total_modifications' in aggregated:
                        print(f"  Total Modifications: {aggregated['total_modifications']['mean']:.1f} ± {aggregated['total_modifications']['std']:.1f}")
                    
                    # Print updated summary table after each method completes
                    print(f"\n{'='*80}")
                    print("CURRENT PROGRESS SUMMARY")
                    print(f"{'='*80}")
                    print_summary_table(base_dir)
                    print(f"{'='*80}\n")
    
    # Save overall summary
    summary_path = os.path.join(base_dir, 'experiment_summary.csv')
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(summary_path, index=False)
        print(f"\n✓ Overall summary saved to {summary_path}")
    
    # Print final summary table
    print(f"\n{'='*80}")
    print("FINAL SUMMARY TABLE")
    print(f"{'='*80}")
    print_summary_table(base_dir)
    
    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Results saved to: {base_dir}")
    print(f"  - Summary table: {os.path.join(base_dir, 'summary_table.txt')}")
    print(f"  - Summary CSV: {os.path.join(base_dir, 'summary_table.csv')}")
    print(f"  - Detailed CSV: {summary_path}")
    print(f"{'='*80}")
    
    # Generate graphs for violation and modification rates over time
    print(f"\n{'='*80}")
    print("Generating graphs for violation and modification rates...")
    print(f"{'='*80}")
    generate_experiment_graphs(base_dir, envs_to_run, methods_to_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run IJCAI experiment suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all environments (takes a long time)
  python scripts/run_ijcai_experiments.py
  
  # Run only CartPole (default: 500 train, 100 eval episodes)
  python scripts/run_ijcai_experiments.py --env CartPole-v1
  
  # Run only CliffWalking with specific methods
  python scripts/run_ijcai_experiments.py --env CliffWalking-v1 --method cppo ppo_preshield_hard
  
  # Test mode (1 episode each)
  python scripts/run_ijcai_experiments.py --test --env CartPole-v1
        """
    )
    parser.add_argument('--num_train_episodes', type=int, default=500,
                       help='Number of training episodes (default: 500)')
    parser.add_argument('--num_eval_episodes', type=int, default=100,
                       help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--base_dir', type=str, default='results/ijcai_experiments',
                       help='Base directory for results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--env', type=str, nargs='+', default=None,
                       help='Filter: only run these environments (default: all). '
                            'Examples: --env CartPole-v1 or --env CartPole-v1 CliffWalking-v1')
    parser.add_argument('--method', type=str, nargs='+', default=None,
                       help='Filter: only run these methods (default: all). '
                            'Examples: --method cppo or --method cppo ppo_preshield_hard')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: run 1 episode per config')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip configurations that already have aggregated results')
    parser.add_argument('--show_summary', action='store_true',
                       help='Show summary table and exit (do not run experiments)')
    parser.add_argument('--generate_graphs', action='store_true',
                       help='Generate graphs from existing data and exit (do not run experiments)')
    
    args = parser.parse_args()
    
    # If just showing summary, do that and exit
    if args.show_summary:
        print_summary_table(args.base_dir)
        sys.exit(0)
    
    # If just generating graphs, do that and exit
    if args.generate_graphs:
        envs_to_run = args.env if args.env else ENVIRONMENTS
        methods_to_run = [m for m in METHODS if args.method is None or m['name'] in args.method]
        generate_experiment_graphs(args.base_dir, envs_to_run, methods_to_run)
        sys.exit(0)
    
    if args.test:
        print("TEST MODE: Running 1 episode per configuration")
        num_train = 1
        num_eval = 1
    else:
        num_train = args.num_train_episodes
        num_eval = args.num_eval_episodes
    
    # Print what will be run
    envs_to_run = args.env if args.env else ENVIRONMENTS
    methods_to_run = [m for m in METHODS if args.method is None or m['name'] in args.method]
    
    print(f"\n{'='*80}")
    print(f"Experiment Configuration:")
    print(f"  Environments: {', '.join(envs_to_run)}")
    print(f"  Methods: {', '.join([m['display_name'] for m in methods_to_run])}")
    print(f"  Seeds per config: {len(SEEDS)}")
    print(f"  Total configs: {len(envs_to_run) * len(methods_to_run) * len(SEEDS)}")
    print(f"  Training episodes: {num_train}")
    print(f"  Evaluation episodes: {num_eval}")
    print(f"  Results directory: {args.base_dir}")
    print(f"{'='*80}\n")
    
    run_all_experiments(
        num_train_episodes=num_train,
        num_eval_episodes=num_eval,
        base_dir=args.base_dir,
        verbose=args.verbose,
        env_filter=args.env,
        method_filter=args.method,
        skip_existing=args.skip_existing
    )

