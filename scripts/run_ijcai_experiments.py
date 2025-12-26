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

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train, evaluate_policy, make_run_dir

# Fixed seeds for reproducibility (5 runs per configuration)
SEEDS = [42, 123, 456, 789, 1011]

# Environments to test
ENVIRONMENTS = [
    'CliffWalking-v1',
    'CartPole-v1',
    'MiniGrid-DoorKey-5x5-v0',
    'ALE/Seaquest-v5',
]

# Methods to compare (8 total)
METHODS = [
    {
        'name': 'cpo',
        'agent': 'cpo',
        'use_shield_post': False,
        'use_shield_pre': False,
        'use_shield_layer': False,
        'mode': '',
        'lambda_sem': 0.0,
        'display_name': 'CPO'
    },
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
    {
        'name': 'ppo_postshield_hard',
        'agent': 'ppo',
        'use_shield_post': True,
        'use_shield_pre': False,
        'use_shield_layer': False,
        'mode': 'hard',
        'lambda_sem': 0.0,
        'display_name': 'PPO + Post-hoc (Hard)'
    },
    {
        'name': 'ppo_postshield_soft',
        'agent': 'ppo',
        'use_shield_post': True,
        'use_shield_pre': False,
        'use_shield_layer': False,
        'mode': 'soft',
        'lambda_sem': 0.0,
        'display_name': 'PPO + Post-hoc (Soft)'
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
        'name': 'ppo_layer_hard',
        'agent': 'ppo',
        'use_shield_post': False,
        'use_shield_pre': False,
        'use_shield_layer': True,
        'mode': 'hard',
        'lambda_sem': 0.0,
        'display_name': 'PPO + Layer (Hard)'
    },
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
        'name': 'ppo_semantic_loss',
        'agent': 'ppo',
        'use_shield_post': False,
        'use_shield_pre': False,
        'use_shield_layer': False,
        'mode': '',
        'lambda_sem': 1.0,  # Semantic loss coefficient
        'display_name': 'PPO + Semantic Loss'
    },
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
        
        # Set memory limits based on environment (to prevent OOM on large environments)
        if 'max_episode_memory' not in agent_kwargs:
            env_memory_limits = {
                'CartPole-v1': 500,      # Short episodes, small memory needed
                'CliffWalking-v1': 1000, # Medium episodes
                'MiniGrid-DoorKey-5x5-v0': 2000,  # Longer episodes
                'ALE/Seaquest-v5': 5000,  # Very long episodes (up to 10k steps)
            }
            max_memory = env_memory_limits.get(env_name, 2000)  # Default 2000
            agent_kwargs['max_episode_memory'] = max_memory
        
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
            agent_kwargs=agent_kwargs if agent_kwargs else None
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
                    
                    summary_data.append({
                        'Environment': env_name,
                        'Method': method['display_name'],
                        'Reward (mean ± std)': f"{agg.get('avg_reward', {}).get('mean', 0):.2f} ± {agg.get('avg_reward', {}).get('std', 0):.2f}",
                        'Viol Rate': f"{agg.get('avg_violations_per_step', {}).get('mean', 0):.4f} ± {agg.get('avg_violations_per_step', {}).get('std', 0):.4f}",
                        'Mod Rate': f"{agg.get('avg_shield_mod_rate', {}).get('mean', 0):.4f} ± {agg.get('avg_shield_mod_rate', {}).get('std', 0):.4f}",
                        'Total Viol': f"{agg.get('total_violations', {}).get('mean', 0):.1f} ± {agg.get('total_violations', {}).get('std', 0):.1f}",
                        'Total Mod': f"{agg.get('total_modifications', {}).get('mean', 0):.1f} ± {agg.get('total_modifications', {}).get('std', 0):.1f}",
                    })
                except Exception as e:
                    summary_data.append({
                        'Environment': env_name,
                        'Method': method['display_name'],
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
                aggregated[key] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val
                }
            else:
                aggregated[key] = values[0]  # Use first non-numeric value
    
    return aggregated


def run_all_experiments(
    num_train_episodes=2000,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run IJCAI experiment suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all environments (takes a long time)
  python scripts/run_ijcai_experiments.py --num_train_episodes 2000
  
  # Run only CartPole
  python scripts/run_ijcai_experiments.py --env CartPole-v1 --num_train_episodes 2000
  
  # Run only CliffWalking with specific methods
  python scripts/run_ijcai_experiments.py --env CliffWalking-v1 --method cppo ppo_postshield_hard
  
  # Test mode (1 episode each)
  python scripts/run_ijcai_experiments.py --test --env CartPole-v1
        """
    )
    parser.add_argument('--num_train_episodes', type=int, default=2000,
                       help='Number of training episodes')
    parser.add_argument('--num_eval_episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--base_dir', type=str, default='results/ijcai_experiments',
                       help='Base directory for results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--env', type=str, nargs='+', default=None,
                       help='Filter: only run these environments (default: all). '
                            'Examples: --env CartPole-v1 or --env CartPole-v1 CliffWalking-v1')
    parser.add_argument('--method', type=str, nargs='+', default=None,
                       help='Filter: only run these methods (default: all). '
                            'Examples: --method cppo or --method cppo ppo_postshield_hard')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: run 1 episode per config')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip configurations that already have aggregated results')
    parser.add_argument('--show_summary', action='store_true',
                       help='Show summary table and exit (do not run experiments)')
    
    args = parser.parse_args()
    
    # If just showing summary, do that and exit
    if args.show_summary:
        print_summary_table(args.base_dir)
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

