#!/usr/bin/env python3
"""
Generate training violation curves for all three environments.
Shows how violation rates evolve during training for each method.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path("/Users/jackbrand/Desktop/coding/imperial/ccn-rl/results/ijcai_experiments")
OUTPUT_DIR = RESULTS_DIR / "training_violation_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Environment configurations
ENVIRONMENTS = {
    'CartPole-v1': {
        'name': 'CartPole-v1',
        'methods': {
            'PPO (Unshielded)': 'ppo_unshielded',
            'PPO + Reward Shaping': 'ppo_reward_shaping',
            'PPO + Semantic Loss': 'ppo_semantic_loss',
            'PPO + Pre-emptive (Soft)': 'ppo_preshield_soft',
            'PPO + Pre-emptive (Hard)': 'ppo_preshield_hard',
            'PPO + Layer (Soft)': 'ppo_layer_soft',
            'PPO + Layer (Hard)': 'ppo_layer_hard',
            'CMDP': 'cppo',
        }
    },
    'CliffWalking-v1': {
        'name': 'CliffWalking-v1',
        'methods': {
            'PPO (Unshielded)': 'ppo_unshielded',
            'PPO + Reward Shaping': 'ppo_reward_shaping',
            'PPO + Semantic Loss': 'ppo_semantic_loss',
            'PPO + Pre-emptive (Soft)': 'ppo_preshield_soft',
            'PPO + Pre-emptive (Hard)': 'ppo_preshield_hard',
            'PPO + Layer (Soft)': 'ppo_layer_soft',
            'PPO + Layer (Hard)': 'ppo_layer_hard',
            'CMDP': 'cppo',
        }
    },
    'ALE_Seaquest-v5': {
        'name': 'ALE/Seaquest-v5',
        'methods': {
            'PPO (Unshielded)': 'ppo_unshielded',
            'PPO + Reward Shaping': 'ppo_reward_shaping',
            'PPO + Semantic Loss': 'ppo_semantic_loss',
            'PPO + Pre-emptive (Soft)': 'ppo_preshield_soft',
            'PPO + Pre-emptive (Hard)': 'ppo_preshield_hard',
            'PPO + Layer (Soft)': 'ppo_layer_soft',
            'PPO + Layer (Hard)': 'ppo_layer_hard',
            'CMDP': 'cppo',
        }
    }
}

# Color scheme for methods
METHOD_COLORS = {
    'PPO (Unshielded)': '#1f77b4',
    'PPO + Reward Shaping': '#ff7f0e',
    'PPO + Semantic Loss': '#2ca02c',
    'PPO + Pre-emptive (Soft)': '#d62728',
    'PPO + Pre-emptive (Hard)': '#9467bd',
    'PPO + Layer (Soft)': '#8c564b',
    'PPO + Layer (Hard)': '#e377c2',
    'CMDP': '#7f7f7f',
}

# Line styles for differentiation
METHOD_LINESTYLES = {
    'PPO (Unshielded)': '-',
    'PPO + Reward Shaping': '-',
    'PPO + Semantic Loss': '-',
    'PPO + Pre-emptive (Soft)': '--',
    'PPO + Pre-emptive (Hard)': '-.',
    'PPO + Layer (Soft)': '--',
    'PPO + Layer (Hard)': '-.',
    'CMDP': '-',
}

def load_training_data(env_dir, method_dir, seeds=[42, 123, 456, 789, 1011]):
    """Load training data for a specific method and environment."""
    method_path = env_dir / method_dir
    
    if not method_path.exists():
        print(f"Warning: Method directory not found: {method_path}")
        return None
    
    all_runs = []
    
    for seed in seeds:
        run_dir = method_path / f"run_{seed}"
        csv_file = run_dir / f"train_metrics_run{seed}.csv"
        
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            df['seed'] = seed
            all_runs.append(df)
        else:
            print(f"Warning: CSV not found: {csv_file}")
    
    if not all_runs:
        return None
    
    return pd.concat(all_runs, ignore_index=True)

def smooth_curve(data, window=10):
    """Apply moving average smoothing with edge handling."""
    if len(data) < window:
        return data
    # Use 'same' mode to preserve length, then handle edges
    smoothed = np.convolve(data, np.ones(window)/window, mode='same')
    # Fix edge effects by using smaller windows at the edges
    for i in range(len(smoothed)):
        if i < window // 2:
            # At the start, use available data
            smoothed[i] = np.mean(data[:i + window // 2 + 1])
        elif i >= len(data) - window // 2:
            # At the end, use available data
            smoothed[i] = np.mean(data[i - window // 2:])
    return smoothed

def plot_training_violations_for_environment(env_key, env_config):
    """Generate training violation curve for a single environment."""
    env_dir = RESULTS_DIR / env_key
    env_name = env_config['name']
    
    print(f"\n{'='*60}")
    print(f"Generating training violation curves for {env_name}")
    print(f"{'='*60}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot each method
    for method_name, method_dir in env_config['methods'].items():
        print(f"Processing {method_name}...")
        
        # Load data
        data = load_training_data(env_dir, method_dir)
        
        if data is None:
            print(f"  Skipping {method_name} (no data)")
            continue
        
        # For each seed, normalize episodes to percentage of training
        normalized_data = []
        for seed in data['seed'].unique():
            seed_data = data[data['seed'] == seed].copy()
            max_episode = seed_data['episode'].max()
            seed_data['training_progress'] = (seed_data['episode'] / max_episode) * 100
            normalized_data.append(seed_data)
        
        normalized_df = pd.concat(normalized_data, ignore_index=True)
        
        # Create bins for training progress (0-100%)
        bins = np.linspace(0, 100, 101)  # 101 points for 0-100%
        normalized_df['progress_bin'] = pd.cut(normalized_df['training_progress'], 
                                               bins=bins, 
                                               labels=bins[:-1])
        
        # Group by progress bin and compute statistics across seeds
        grouped = normalized_df.groupby('progress_bin')['violation_rate'].agg(['mean', 'std', 'count'])
        
        # Remove bins with no data
        grouped = grouped.dropna()
        
        progress = grouped.index.astype(float).values
        mean_viol = grouped['mean'].values
        std_viol = grouped['std'].values
        
        # Compute standard error
        se_viol = std_viol / np.sqrt(grouped['count'].values)
        
        # Apply smoothing
        smooth_window = min(10, len(progress) // 10)
        if smooth_window > 1:
            mean_smooth = smooth_curve(mean_viol, smooth_window)
            se_smooth = smooth_curve(se_viol, smooth_window)
        else:
            mean_smooth = mean_viol
            se_smooth = se_viol
        
        # Progress array stays the same length now
        progress_smooth = progress
        
        # Plot
        color = METHOD_COLORS.get(method_name, None)
        linestyle = METHOD_LINESTYLES.get(method_name, '-')
        
        ax.plot(progress_smooth, mean_smooth, 
                label=method_name, 
                color=color, 
                linestyle=linestyle,
                linewidth=2, 
                alpha=0.9)
        
        print(f"  Final violation rate: {mean_viol[-1]:.4f} ± {se_viol[-1]:.4f}")
    
    # Formatting
    ax.set_xlabel('Training Progress (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Violation Rate', fontsize=14, fontweight='bold')
    ax.set_title(f'Training Violation Rates - {env_name}', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Set axes limits
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save figure
    output_file = OUTPUT_DIR / f"{env_key}_training_violations.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {output_file}")
    
    plt.close()

def main():
    """Generate training violation curves for all environments."""
    print("\n" + "="*60)
    print("TRAINING VIOLATION CURVE GENERATOR")
    print("="*60)
    
    for env_key, env_config in ENVIRONMENTS.items():
        try:
            plot_training_violations_for_environment(env_key, env_config)
        except Exception as e:
            print(f"Error processing {env_key}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()

