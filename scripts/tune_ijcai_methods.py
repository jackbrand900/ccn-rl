#!/usr/bin/env python3
"""
Hyperparameter tuning script for IJCAI experiments.
Tunes each method to achieve similar target rewards for fair comparison.
"""

import optuna
import numpy as np
import yaml
import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import train, evaluate_policy

# Target rewards per environment (to be tuned to match)
# Since we're focusing on violation/modification rates, we'll use more realistic targets
# that allow fair comparison even if absolute rewards are lower
TARGET_REWARDS = {
    'CartPole-v1': 300.0,  # Realistic target for constrained methods
    'CliffWalking-v1': -15.0,  # Slightly worse than optimal (-13)
    'MiniGrid-DoorKey-5x5-v0': 0.7,  # Realistic target
    'ALE/Seaquest-v5': 800.0,  # Realistic target
}

# Methods to tune (matching run_ijcai_experiments.py)
METHODS = [
    # CPO skipped for now
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
    # Post-hoc methods skipped per user request
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
        'lambda_sem': 1.0,
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
        'lambda_penalty': 1.0,
        'display_name': 'PPO + Reward Shaping'
    },
]


def objective(trial, env_name, method, target_reward, num_train_episodes=500, num_eval_episodes=50):
    """
    Objective function: minimize absolute difference from target reward.
    """
    # === Shared hyperparameters ===
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    use_orthogonal_init = trial.suggest_categorical("use_orthogonal_init", [True, False])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    
    agent_kwargs = {
        "lr": lr,
        "gamma": gamma,
        "hidden_dim": hidden_dim,
        "use_orthogonal_init": use_orthogonal_init,
        "num_layers": num_layers
    }
    
    # === Agent-specific parameters ===
    if method['agent'] == 'ppo':
        clip_eps = trial.suggest_float("clip_eps", 0.1, 0.3)
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
        epochs = trial.suggest_int("epochs", 1, 10)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        agent_kwargs.update({
            "clip_eps": clip_eps,
            "ent_coef": ent_coef,
            "epochs": epochs,
            "batch_size": batch_size,
        })
        
        # Semantic loss coefficient (if applicable)
        if method.get('lambda_sem', 0.0) > 0:
            lambda_sem = trial.suggest_float("lambda_sem", 0.1, 10.0, log=True)
            agent_kwargs['lambda_sem'] = lambda_sem
        
        # Reward shaping penalty (if applicable)
        if method.get('lambda_penalty', 0.0) > 0:
            lambda_penalty = trial.suggest_float("lambda_penalty", 0.1, 10.0, log=True)
            agent_kwargs['lambda_penalty'] = lambda_penalty
    
    elif method['agent'] == 'cpo':
        cost_gamma = trial.suggest_float("cost_gamma", 0.90, 0.999)
        cost_lam = trial.suggest_float("cost_lam", 0.90, 0.999)
        clip_eps = trial.suggest_float("clip_eps", 0.1, 0.3)
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
        epochs = trial.suggest_int("epochs", 1, 10)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        budget = trial.suggest_float("budget", 0.05, 0.30)
        max_kl = trial.suggest_float("max_kl", 0.005, 0.02)  # Trust region size
        
        agent_kwargs.update({
            "cost_gamma": cost_gamma,
            "cost_lam": cost_lam,
            "clip_eps": clip_eps,
            "ent_coef": ent_coef,
            "epochs": epochs,
            "batch_size": batch_size,
            "budget": budget,
            "max_kl": max_kl,
        })
    elif method['agent'] == 'cppo':
        cost_gamma = trial.suggest_float("cost_gamma", 0.90, 0.999)
        cost_lam = trial.suggest_float("cost_lam", 0.90, 0.999)
        clip_eps = trial.suggest_float("clip_eps", 0.1, 0.3)
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
        epochs = trial.suggest_int("epochs", 1, 10)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        budget = trial.suggest_float("budget", 0.05, 0.30)  # Tune budget
        nu_lr = trial.suggest_float("nu_lr", 1e-4, 1e-2, log=True)  # Tune nu learning rate
        
        agent_kwargs.update({
            "cost_gamma": cost_gamma,
            "cost_lam": cost_lam,
            "clip_eps": clip_eps,
            "ent_coef": ent_coef,
            "epochs": epochs,
            "batch_size": batch_size,
            "budget": budget,
            "nu_lr": nu_lr,
        })
    
    # === Train ===
    try:
        agent, episode_rewards, best_weights, best_avg_reward, env = train(
            agent=method['agent'],
            env_name=env_name,
            num_episodes=num_train_episodes,
            use_shield_post=method['use_shield_post'],
            use_shield_pre=method['use_shield_pre'],
            use_shield_layer=method['use_shield_layer'],
            monitor_constraints=True,
            mode=method['mode'],
            verbose=False,
            visualize=False,
            render=False,
            seed=42,  # Fixed seed for tuning
            agent_kwargs=agent_kwargs,
            early_stop_patience=100  # Stop training if no improvement after 100 episodes
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
            softness=method['mode']
        )
        
        avg_reward = results['avg_reward']
        
        # Objective: minimize absolute difference from target
        # Normalize by target to make it scale-invariant
        reward_diff = abs(avg_reward - target_reward)
        normalized_diff = reward_diff / (abs(target_reward) + 1e-6)
        
        # Store actual reward in trial for progress logging
        trial.set_user_attr("actual_reward", avg_reward)
        trial.set_user_attr("target_reward", target_reward)
        trial.set_user_attr("reward_diff", reward_diff)
        
        # Return negative normalized diff (Optuna minimizes, we want to minimize diff)
        return -normalized_diff
        
    except Exception as e:
        print(f"Error in trial: {e}")
        return -1000.0  # Very bad score


def tune_method(env_name, method, target_reward, n_trials=30, num_train_episodes=500, num_eval_episodes=50):
    """
    Tune hyperparameters for a specific method on a specific environment.
    """
    method_name = method['name']
    env_safe = env_name.replace('/', '_')
    
    storage = f"sqlite:///optuna_ijcai_{method_name}_{env_safe}.db"
    study = optuna.create_study(
        direction="maximize",  # We maximize negative diff (minimize diff)
        study_name=f"ijcai_{method_name}_{env_safe}",
        storage=storage,
        load_if_exists=True
    )
    
    print(f"\n{'='*80}")
    print(f"Tuning: {method['display_name']} on {env_name}")
    print(f"Target reward: {target_reward}")
    print(f"Trials: {n_trials}")
    print(f"{'='*80}\n")
    
    # Track timing for progress estimates
    method_start_time = time.time()
    trial_times = []
    early_stopped = False
    
    # Early stopping threshold: stop if within 5% of target (normalized diff < 0.05)
    # Since we return -normalized_diff, we stop if best_value > -0.05
    EARLY_STOP_THRESHOLD = -0.05  # Corresponds to 5% difference from target
    
    # Check if we already have a good result from previous runs
    # IMPORTANT: Re-evaluate against the NEW target, not the old stored target
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) >= 1:
        best_trial = study.best_trial
        best_actual_reward = best_trial.user_attrs.get("actual_reward", None)
        
        # Re-calculate diff against the NEW target (not the old stored target)
        if best_actual_reward is not None:
            reward_diff = abs(best_actual_reward - target_reward)
            normalized_diff = reward_diff / (abs(target_reward) + 1e-6)
            # Check if within 5% of NEW target
            if normalized_diff < 0.05:
                best_target_old = best_trial.user_attrs.get("target_reward", target_reward)
                print(f"\n[✓] Found existing trial within 5% of NEW target ({target_reward:.1f})!")
                print(f"    Best reward: {best_actual_reward:.2f} (new target: {target_reward:.2f}, diff: {reward_diff:.2f})")
                if best_target_old != target_reward:
                    print(f"    Note: This trial was originally tuned for target {best_target_old:.2f}")
                print(f"    Skipping new trials. Use existing best parameters.\n")
                early_stopped = True
    
    def progress_callback(study, trial):
        """Callback to track progress and estimate remaining time"""
        nonlocal early_stopped
        
        # Only process completed trials
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
            
        trial_times.append(time.time())
        
        # Get actual reward from best trial
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials) < 1:
            return
            
        best_trial = study.best_trial
        best_actual_reward = best_trial.user_attrs.get("actual_reward", None)
        
        # Get current trial info
        current_reward = trial.user_attrs.get("actual_reward", None)
        current_diff = trial.user_attrs.get("reward_diff", None)
        
        # Check for early stopping (within 5% of NEW target)
        # Re-calculate normalized diff against the NEW target, not the old stored target
        if len(completed_trials) >= 2 and best_actual_reward is not None and not early_stopped:
            reward_diff = abs(best_actual_reward - target_reward)
            normalized_diff = reward_diff / (abs(target_reward) + 1e-6)
            # Check if within 5% of NEW target (normalized_diff < 0.05)
            # Since we return -normalized_diff, we check if -normalized_diff > -0.05
            if normalized_diff < 0.05:
                early_stopped = True
                print(f"\n[✓] Early stopping: Found configuration within 5% of target ({target_reward:.1f})!")
                print(f"    Best reward: {best_actual_reward:.2f} (target: {target_reward:.2f}, diff: {reward_diff:.2f})")
                try:
                    study.stop()
                except (AttributeError, RuntimeError):
                    # Older Optuna versions might not have stop() method
                    # Or study might already be stopped
                    pass
                return
        
        # For display, use stored values but show current target
        best_target_old = best_trial.user_attrs.get("target_reward", target_reward)
        best_diff_old = best_trial.user_attrs.get("reward_diff", None)
        
        # Only show progress for completed trials in this run
        if len(trial_times) > 1:
            avg_trial_time = (trial_times[-1] - trial_times[0]) / len(trial_times)
            remaining_trials = n_trials - len(trial_times)
            estimated_remaining = timedelta(seconds=int(avg_trial_time * remaining_trials))
            
            elapsed = time.time() - method_start_time
            elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
            
            # Format output - show current target, not old stored target
            if best_actual_reward is not None:
                # Calculate diff against current target
                current_diff_from_target = abs(best_actual_reward - target_reward)
                reward_info = f"Best reward: {best_actual_reward:.2f} (target: {target_reward:.2f}, diff: {current_diff_from_target:.2f})"
            else:
                reward_info = f"Best value: {study.best_value:.4f}"
            
            if current_reward is not None:
                reward_info += f" | Current: {current_reward:.2f}"
                if current_diff is not None:
                    reward_info += f" (diff: {current_diff:.2f})"
            
            print(f"\n[Trial {len(trial_times)}/{n_trials}] "
                  f"Elapsed: {elapsed_str} | "
                  f"Est. remaining: {str(estimated_remaining).split('.')[0]} | "
                  f"{reward_info}")
    
    study.optimize(
        lambda trial: objective(
            trial,
            env_name=env_name,
            method=method,
            target_reward=target_reward,
            num_train_episodes=num_train_episodes,
            num_eval_episodes=num_eval_episodes
        ),
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[progress_callback]
    )
    
    if early_stopped:
        print(f"\n[ℹ] Tuning stopped early after {len(trial_times)} trials (found good configuration)")
    
    print(f"\n{'='*80}")
    print(f"Best trial for {method['display_name']} on {env_name}:")
    trial = study.best_trial
    print(f"  Reward difference: {-trial.value:.4f} (normalized)")
    print(f"  Params: {trial.params}")
    
    # Test the best params to get actual reward
    # Use the same number of episodes as tuning for fair comparison
    print(f"\n  Testing best parameters (using {num_train_episodes} episodes, same as tuning)...")
    best_agent_kwargs = trial.params.copy()
    
    # Train with best params
    agent, episode_rewards, best_weights, best_avg_reward, env = train(
        agent=method['agent'],
        env_name=env_name,
        num_episodes=num_train_episodes,
        use_shield_post=method['use_shield_post'],
        use_shield_pre=method['use_shield_pre'],
        use_shield_layer=method['use_shield_layer'],
        monitor_constraints=True,
        mode=method['mode'],
        verbose=False,
        visualize=False,
        render=False,
        seed=42,
        agent_kwargs=best_agent_kwargs,
        early_stop_patience=100  # Stop training if no improvement after 100 episodes
    )
    
    if hasattr(agent, 'load_weights') and best_weights is not None:
        agent.load_weights(best_weights)
    
    # Evaluate - this is the true performance metric (deterministic, no exploration)
    results = evaluate_policy(
        agent,
        env,
        num_episodes=num_eval_episodes,
        visualize=False,
        render=False,
        force_disable_shield=False,
        softness=method['mode']
    )
    
    actual_reward = results['avg_reward']
    print(f"  Evaluation reward: {actual_reward:.2f} (target: {target_reward:.2f}, diff: {abs(actual_reward - target_reward):.2f})")
    print(f"  Note: Training reward ({best_avg_reward:.2f}) may be higher due to exploration during training.")
    print(f"{'='*80}\n")
    
    # Save to YAML
    config_dir = Path("config/ijcai_tuned")
    config_dir.mkdir(exist_ok=True)
    
    filename = config_dir / f"{method_name}_{env_safe}_params.yaml"
    with open(filename, "w") as f:
        yaml.dump(trial.params, f, default_flow_style=False)
    
    print(f"[✓] Best hyperparameters saved to {filename}\n")
    
    return trial.params, actual_reward


def main():
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for IJCAI experiments to match target rewards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tune all methods on CartPole
  python scripts/tune_ijcai_methods.py --env CartPole-v1 --trials 20
  
  # Tune specific method
  python scripts/tune_ijcai_methods.py --env CartPole-v1 --method cppo --trials 50
  
  # Quick test (fewer episodes, fewer trials)
  python scripts/tune_ijcai_methods.py --env CartPole-v1 --trials 10 --train_episodes 200 --eval_episodes 20
        """
    )
    parser.add_argument('--env', type=str, required=True,
                       choices=['CartPole-v1', 'CliffWalking-v1', 'MiniGrid-DoorKey-5x5-v0', 'ALE/Seaquest-v5'],
                       help='Environment to tune')
    parser.add_argument('--method', type=str, default=None,
                       help='Specific method to tune (default: all)')
    parser.add_argument('--trials', type=int, default=15,
                       help='Number of Optuna trials per method (default: 15)')
    parser.add_argument('--train_episodes', type=int, default=500,
                       help='Number of training episodes during tuning (default: 500)')
    parser.add_argument('--eval_episodes', type=int, default=50,
                       help='Number of evaluation episodes during tuning')
    parser.add_argument('--target_reward', type=float, default=None,
                       help='Target reward (default: use predefined)')
    
    args = parser.parse_args()
    
    target_reward = args.target_reward if args.target_reward is not None else TARGET_REWARDS.get(args.env)
    if target_reward is None:
        print(f"Warning: No target reward for {args.env}, using 100.0")
        target_reward = 100.0
    
    methods_to_tune = [m for m in METHODS if args.method is None or m['name'] == args.method]
    
    if not methods_to_tune:
        print(f"Error: Method '{args.method}' not found")
        return
    
    print(f"\n{'#'*80}")
    print(f"# Hyperparameter Tuning for IJCAI Experiments")
    print(f"# Environment: {args.env}")
    print(f"# Target Reward: {target_reward}")
    print(f"# Methods: {len(methods_to_tune)}")
    print(f"# Trials per method: {args.trials}")
    print(f"# Train episodes per trial: {args.train_episodes}")
    print(f"# Eval episodes per trial: {args.eval_episodes}")
    print(f"{'#'*80}")
    print(f"\n📋 Methods to tune:")
    for i, method in enumerate(methods_to_tune, 1):
        shield_info = []
        if method['use_shield_post']:
            shield_info.append(f"Post-hoc ({method['mode']})")
        if method['use_shield_pre']:
            shield_info.append(f"Pre-emptive ({method['mode']})")
        if method['use_shield_layer']:
            shield_info.append(f"Layer ({method['mode']})")
        if method.get('lambda_sem', 0) > 0:
            shield_info.append("Semantic Loss")
        if method.get('lambda_penalty', 0) > 0:
            shield_info.append("Reward Shaping")
        
        shield_str = " + ".join(shield_info) if shield_info else "Unshielded"
        print(f"  {i}. {method['display_name']} ({shield_str})")
    print(f"{'#'*80}\n")
    
    overall_start_time = time.time()
    results_summary = []
    
    for method_idx, method in enumerate(methods_to_tune, 1):
        method_start = time.time()
        print(f"\n{'='*80}")
        print(f"METHOD {method_idx}/{len(methods_to_tune)}: {method['display_name']}")
        print(f"{'='*80}")
        try:
            params, actual_reward = tune_method(
                env_name=args.env,
                method=method,
                target_reward=target_reward,
                n_trials=args.trials,
                num_train_episodes=args.train_episodes,
                num_eval_episodes=args.eval_episodes
            )
            method_time = time.time() - method_start
            method_time_str = str(timedelta(seconds=int(method_time))).split('.')[0]
            
            results_summary.append({
                'method': method['display_name'],
                'target_reward': target_reward,
                'actual_reward': actual_reward,
                'diff': abs(actual_reward - target_reward),
                'time': method_time_str
            })
            
            # Estimate remaining time
            elapsed_total = time.time() - overall_start_time
            avg_time_per_method = elapsed_total / method_idx
            remaining_methods = len(methods_to_tune) - method_idx
            estimated_remaining = timedelta(seconds=int(avg_time_per_method * remaining_methods))
            
            print(f"\n[✓] {method['display_name']} completed in {method_time_str}")
            if remaining_methods > 0:
                print(f"[⏱] Estimated time remaining: {str(estimated_remaining).split('.')[0]} "
                      f"({remaining_methods} methods left)")
        except Exception as e:
            print(f"Error tuning {method['display_name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    total_time = time.time() - overall_start_time
    total_time_str = str(timedelta(seconds=int(total_time))).split('.')[0]
    
    print(f"\n{'#'*80}")
    print("# Tuning Summary")
    print(f"{'#'*80}")
    print(f"{'Method':<30} {'Target':<12} {'Actual':<12} {'Diff':<12} {'Time':<10}")
    print("-" * 80)
    for r in results_summary:
        time_str = r.get('time', 'N/A')
        print(f"{r['method']:<30} {r['target_reward']:<12.2f} {r['actual_reward']:<12.2f} "
              f"{r['diff']:<12.2f} {time_str:<10}")
    print("-" * 80)
    print(f"{'TOTAL TIME':<30} {'':<12} {'':<12} {'':<12} {total_time_str:<10}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()

