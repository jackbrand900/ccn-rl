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
# Aligned with TRAINING_TARGET_REWARDS to ensure hyperparameters are optimized
# for the actual training regime (early stopping at this target)
TARGET_REWARDS = {
    'CartPole-v1': 200.0,  # Match training target for fair comparison
    'CliffWalking-v1': -15.0,  # Slightly worse than optimal (-13)
    'MiniGrid-DoorKey-5x5-v0': 0.7,  # Realistic target
    'ALE/Seaquest-v5': 800.0,  # Realistic target
}

# Target rewards for early stopping during training
# Methods will stop training once they reach this target (using rolling average)
# This ensures all methods are evaluated at similar performance levels
TRAINING_TARGET_REWARDS = {
    'CartPole-v1': 200.0,  # Stop training when rolling average reaches 200
    'CliffWalking-v1': -15.0,
    'MiniGrid-DoorKey-5x5-v0': 0.7,
    'ALE/Seaquest-v5': 800.0,
}

# Methods to tune (ordered from lightest to heaviest for memory/computational efficiency)
# Complexity: Vanilla < Reward Shaping < Semantic Loss < Pre-emptive < Layer < CMDP
METHODS = [
    # Lightest: Baseline (no modifications)
    {
        'name': 'ppo_unshielded',
        'agent': 'ppo',
        'use_shield_post': False,
        'use_shield_pre': False,
        'use_shield_layer': False,
        'mode': '',
        'lambda_sem': 0.0,
        'display_name': 'PPO (Unshielded)'
    },
    # Light: Simple reward/loss modifications (no shield integration)
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
    # Medium: Pre-emptive shield (action modification before execution)
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
    # Heavier: Shield layer (integrated into network)
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
    # Heaviest: Constrained optimization (CMDP with Lagrangian multipliers)
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
        # Increased budget range - allow more violations to achieve higher rewards
        budget = trial.suggest_float("budget", 0.10, 0.50)  # Increased from 0.05-0.30 to 0.10-0.50
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
    agent = None
    env = None
    try:
        # Get training target reward for early stopping
        training_target = TRAINING_TARGET_REWARDS.get(env_name, target_reward)
        
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
            early_stop_patience=100,  # Stop training if no improvement after 100 episodes
            target_reward=training_target  # Stop training when target reward is reached
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
        
        result = -normalized_diff
        
    except Exception as e:
        print(f"Error in trial: {e}")
        result = -1000.0  # Very bad score
    
    finally:
        # Explicit cleanup to prevent memory accumulation
        if agent is not None:
            # Clear agent memory buffers
            if hasattr(agent, 'memory'):
                agent.memory.clear()
            if hasattr(agent, 'constraint_monitor'):
                agent.constraint_monitor.reset_all()
            # Delete agent
            del agent
        
        if env is not None:
            env.close()
            del env
        
        # Clear episode_rewards and best_weights if they exist
        if 'episode_rewards' in locals():
            del episode_rewards
        if 'best_weights' in locals():
            del best_weights
        
        # Force garbage collection and clear CUDA cache
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return result


def tune_method(env_name, method, target_reward, n_trials=30, num_train_episodes=500, num_eval_episodes=50):
    """
    Tune hyperparameters for a specific method on a specific environment.
    """
    method_name = method['name']
    env_safe = env_name.replace('/', '_')
    
    storage = f"sqlite:///optuna_ijcai_{method_name}_{env_safe}.db"
    
    # Check if database file exists and is writable
    db_path = f"optuna_ijcai_{method_name}_{env_safe}.db"
    if os.path.exists(db_path):
        if not os.access(db_path, os.W_OK):
            print(f"Warning: Database file {db_path} is not writable. Attempting to fix permissions...")
            try:
                os.chmod(db_path, 0o644)
            except Exception as e:
                print(f"Could not fix permissions: {e}")
                print(f"Please manually fix permissions or delete {db_path} and restart.")
                raise
        
        # Test if we can actually write to the database
        try:
            import sqlite3
            test_conn = sqlite3.connect(db_path, timeout=1.0)
            test_conn.execute("PRAGMA quick_check;")
            test_conn.close()
        except sqlite3.OperationalError as e:
            if "readonly" in str(e).lower() or "locked" in str(e).lower():
                print(f"Warning: Database {db_path} appears to be locked or readonly.")
                print(f"Backing up and recreating database...")
                backup_path = f"{db_path}.backup_{int(time.time())}"
                try:
                    import shutil
                    shutil.move(db_path, backup_path)
                    print(f"Backed up to {backup_path}")
                except Exception as backup_e:
                    print(f"Could not backup database: {backup_e}")
                    print(f"Please manually delete {db_path} and restart.")
                    raise
            else:
                print(f"Database check failed: {e}")
    
    try:
        study = optuna.create_study(
            direction="maximize",  # We maximize negative diff (minimize diff)
            study_name=f"ijcai_{method_name}_{env_safe}",
            storage=storage,
            load_if_exists=True
        )
    except Exception as e:
        print(f"Error creating/loading study: {e}")
        print("Attempting to create new study (will overwrite existing)...")
        # Try creating without load_if_exists
        study = optuna.create_study(
            direction="maximize",
            study_name=f"ijcai_{method_name}_{env_safe}",
            storage=storage,
            load_if_exists=False
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
    
    # Add callback to clean up memory after each trial
    def cleanup_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE or trial.state == optuna.trial.TrialState.FAIL:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    try:
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
            callbacks=[progress_callback, cleanup_callback]
        )
    except optuna.exceptions.StorageInternalError as e:
        print(f"\n[ERROR] Database storage error: {e}")
        print("This usually happens when the database file is locked or corrupted.")
        print(f"Try deleting the database file: {db_path}")
        print("Or check if another process is using it.")
        raise
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during optimization: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    if early_stopped:
        print(f"\n[ℹ] Tuning stopped early after {len(trial_times)} trials (found good configuration)")
    
    print(f"\n{'='*80}")
    print(f"Best trial for {method['display_name']} on {env_name}:")
    trial = study.best_trial
    print(f"  Reward difference: {-trial.value:.4f} (normalized)")
    print(f"  Params: {trial.params}")
    
    # Test the best params to get actual reward
    # Use same number of episodes as tuning for consistency with data generation
    test_episodes = num_train_episodes
    print(f"\n  Testing best parameters (using {test_episodes} episodes)...")
    best_agent_kwargs = trial.params.copy()
    
    # Train with best params
    test_agent = None
    test_env = None
    try:
        # Get training target for final test
        training_target = TRAINING_TARGET_REWARDS.get(env_name, target_reward)
        
        test_agent, episode_rewards, best_weights, best_avg_reward, test_env = train(
            agent=method['agent'],
            env_name=env_name,
            num_episodes=test_episodes,
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
            early_stop_patience=100,  # Stop training if no improvement after 100 episodes
            target_reward=training_target  # Stop training when target reward is reached
        )
        
        if hasattr(test_agent, 'load_weights') and best_weights is not None:
            test_agent.load_weights(best_weights)
        
        # Evaluate - this is the true performance metric (deterministic, no exploration)
        results = evaluate_policy(
            test_agent,
            test_env,
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
    finally:
        # Cleanup after test run
        if test_agent is not None:
            if hasattr(test_agent, 'memory'):
                test_agent.memory.clear()
            if hasattr(test_agent, 'constraint_monitor'):
                test_agent.constraint_monitor.reset_all()
            del test_agent
        if test_env is not None:
            test_env.close()
            del test_env
        if 'episode_rewards' in locals():
            del episode_rewards
        if 'best_weights' in locals():
            del best_weights
        
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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
    
    # Get base target reward for environment
    base_target_reward = args.target_reward if args.target_reward is not None else TARGET_REWARDS.get(args.env)
    if base_target_reward is None:
        print(f"Warning: No target reward for {args.env}, using 100.0")
        base_target_reward = 100.0
    
    methods_to_tune = [m for m in METHODS if args.method is None or m['name'] == args.method]
    
    if not methods_to_tune:
        print(f"Error: Method '{args.method}' not found")
        return
    
    training_target = TRAINING_TARGET_REWARDS.get(args.env, base_target_reward)
    
    print(f"\n{'#'*80}")
    print(f"# Hyperparameter Tuning for IJCAI Experiments")
    print(f"# Environment: {args.env}")
    print(f"# Training Target Reward: {training_target} (early stopping when reached)")
    print(f"# Tuning Objective Target: {base_target_reward} (minimize difference)")
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
                target_reward=base_target_reward,  # Used for tuning objective (minimize difference)
                n_trials=args.trials,
                num_train_episodes=args.train_episodes,
                num_eval_episodes=args.eval_episodes
            )
            method_time = time.time() - method_start
            method_time_str = str(timedelta(seconds=int(method_time))).split('.')[0]
            
            # Get training target for display
            training_target = TRAINING_TARGET_REWARDS.get(args.env, base_target_reward)
            
            results_summary.append({
                'method': method['display_name'],
                'training_target': training_target,  # Target used for early stopping
                'actual_reward': actual_reward,
                'diff': abs(actual_reward - training_target),
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
    print(f"{'Method':<30} {'Train Target':<12} {'Actual':<12} {'Diff':<12} {'Time':<10}")
    print("-" * 80)
    for r in results_summary:
        time_str = r.get('time', 'N/A')
        print(f"{r['method']:<30} {r['training_target']:<12.2f} {r['actual_reward']:<12.2f} "
              f"{r['diff']:<12.2f} {time_str:<10}")
    print("-" * 80)
    print(f"{'TOTAL TIME':<30} {'':<12} {'':<12} {'':<12} {total_time_str:<10}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()

