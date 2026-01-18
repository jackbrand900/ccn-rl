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
import subprocess
import json
import gc
import tempfile
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
    'CliffWalking-v1': -20.0,  # Target reward for CliffWalking
    'MiniGrid-DoorKey-5x5-v0': 0.7,  # Realistic target
    'ALE/Seaquest-v5': 250.0,  # More realistic target (best CMDP achieved ~180, PPO baseline ~200-300)
}

# Target rewards for early stopping during training
# Methods will stop training once they reach this target (using rolling average)
# This ensures all methods are evaluated at similar performance levels
TRAINING_TARGET_REWARDS = {
    'CartPole-v1': 200.0,  # Stop training when rolling average reaches 200
    'CliffWalking-v1': -20.0,  # Stop training when rolling average reaches -20
    'MiniGrid-DoorKey-5x5-v0': 0.7,
    'ALE/Seaquest-v5': 250.0,
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


def objective(trial, env_name, method, target_reward, num_train_episodes=500, num_eval_episodes=50, max_episode_steps=None, use_ram_obs=False, use_subprocess=False):
    """
    Objective function: minimize absolute difference from target reward.
    """
    # === Shared hyperparameters ===
    # Tuned ranges for CliffWalking PPO agents (capturing paper parameters)
    if env_name == 'CliffWalking-v1' and method['agent'] == 'ppo':
        # Expanded ranges that capture paper values: hidden_dim=128, gamma=0.99, clip_eps=0.2, 
        # ent_coef=0.015, epochs=7, batch_size=512, lr=3e-4 or 5e-4
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])  # Captures 128, wider range
        use_orthogonal_init = trial.suggest_categorical("use_orthogonal_init", [False, True])  # Captures False
        gamma = trial.suggest_float("gamma", 0.90, 0.999)  # Captures 0.99, wider range
        num_layers = trial.suggest_int("num_layers", 2, 4)
        
        # Learning rate ranges: 3e-4 for reward shaping/semantic loss, 5e-4 for preemptive/layer
        if method['name'] in ['ppo_reward_shaping', 'ppo_semantic_loss']:
            lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)  # Wider range around 3e-4
        elif method['name'] in ['ppo_preshield_soft', 'ppo_preshield_hard', 
                                'ppo_layer_soft', 'ppo_layer_hard']:
            lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)  # Wider range around 5e-4
        else:
            lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)  # Default wider range
        
        clip_eps = trial.suggest_float("clip_eps", 0.1, 0.3)  # Wider range around 0.2
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)  # Wider range around 0.015
        epochs = trial.suggest_int("epochs", 1, 10)  # Full range, captures 7
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])  # Captures 512, wider range
        
        agent_kwargs = {
            "lr": lr,
            "gamma": gamma,
            "hidden_dim": hidden_dim,
            "use_orthogonal_init": use_orthogonal_init,
            "num_layers": num_layers,
            "clip_eps": clip_eps,
            "ent_coef": ent_coef,
            "epochs": epochs,
            "batch_size": batch_size,
        }
        
        # Semantic loss coefficient (if applicable) - still tune this
        if method.get('lambda_sem', 0.0) > 0:
            lambda_sem = trial.suggest_float("lambda_sem", 0.1, 10.0, log=True)
            agent_kwargs['lambda_sem'] = lambda_sem
        
        # Reward shaping penalty (if applicable) - still tune this
        if method.get('lambda_penalty', 0.0) > 0:
            lambda_penalty = trial.suggest_float("lambda_penalty", 0.1, 10.0, log=True)
            agent_kwargs['lambda_penalty'] = lambda_penalty
    
    # CMDP-specific handling for CliffWalking
    elif env_name == 'CliffWalking-v1' and method['agent'] == 'cppo':
        lr = trial.suggest_float("lr", 0.012, 0.022, log=True)  # Tight range around trial 18 (0.0172)
        gamma = trial.suggest_float("gamma", 0.93, 0.97)  # Around 0.951
        hidden_dim = trial.suggest_categorical("hidden_dim", [256])  # Fix to 256
        use_orthogonal_init = trial.suggest_categorical("use_orthogonal_init", [False])  # Fix to False
        num_layers = trial.suggest_int("num_layers", 3, 3)  # Fix to 3
        
        agent_kwargs = {
            "lr": lr,
            "gamma": gamma,
            "hidden_dim": hidden_dim,
            "use_orthogonal_init": use_orthogonal_init,
            "num_layers": num_layers
        }
    
    # Seaquest-specific tuning (Atari environment with sparse rewards)
    elif env_name == 'ALE/Seaquest-v5':
        # Learning rate: 1e-4 to 3e-4 (optimized range for Atari)
        lr = trial.suggest_float("lr", 1e-4, 3e-4, log=True)
        # Gamma: 0.95-0.999, focusing around 0.99
        gamma = trial.suggest_float("gamma", 0.95, 0.999)
        # Hidden dimensions: reasonable range for deep networks
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
        
        # PPO-specific parameters for Seaquest
        if method['agent'] == 'ppo':
            # Clip range: 0.1 to 0.2 (tighter range for stability)
            clip_eps = trial.suggest_float("clip_eps", 0.1, 0.2)
            # Entropy coefficient: 0.005 to 0.1 (crucial for exploration in sparse rewards)
            ent_coef = trial.suggest_float("ent_coef", 0.005, 0.1, log=True)
            # Epochs: 4-10 (multiple epochs per batch for stability)
            epochs = trial.suggest_int("epochs", 4, 10)
            # Batch size: 128, 256 (larger batches for stability)
            batch_size = trial.suggest_categorical("batch_size", [128, 256])
            
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
        
        # CPO/CPPO-specific parameters for Seaquest
        elif method['agent'] in ['cpo', 'cppo']:
            cost_gamma = trial.suggest_float("cost_gamma", 0.95, 0.999)  # Similar to gamma
            cost_lam = trial.suggest_float("cost_lam", 0.90, 0.99)  # GAE lambda: 0.95-0.99
            clip_eps = trial.suggest_float("clip_eps", 0.1, 0.2)
            # Much higher entropy for better exploration and to prevent collapse (0.05-0.25)
            ent_coef = trial.suggest_float("ent_coef", 0.05, 0.25, log=True)
            epochs = trial.suggest_int("epochs", 4, 10)
            batch_size = trial.suggest_categorical("batch_size", [128, 256])
            # Much higher budget to allow violations during learning (0.25-0.60)
            # Violations are ~0.10-0.12, so budget should be comfortably above that
            budget = trial.suggest_float("budget", 0.25, 0.60)
            # Much lower nu_lr for very stable Lagrangian updates (1e-5 to 1e-3)
            # Prevents nu from growing too fast and causing collapse
            nu_lr = trial.suggest_float("nu_lr", 1e-5, 1e-3, log=True)
            
            agent_kwargs.update({
                "cost_gamma": cost_gamma,
                "cost_lam": cost_lam,
                "clip_eps": clip_eps,
                "ent_coef": ent_coef,
                "epochs": epochs,
                "batch_size": batch_size,
                "budget": budget,
            })
            if method['agent'] == 'cppo':
                agent_kwargs['nu_lr'] = nu_lr
    
    # Default tuning for other environments/methods
    else:
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
    
    # === Agent-specific parameters (for non-CliffWalking, non-Seaquest PPO) ===
    if method['agent'] == 'ppo' and not (env_name == 'CliffWalking-v1') and not (env_name == 'ALE/Seaquest-v5'):
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
        # Environment-specific adjustments for CliffWalking
        if env_name == 'CliffWalking-v1':
            # Tight ranges around trial 18's best params
            lr = trial.suggest_float("lr", 0.012, 0.022, log=True)  # Around 0.0172
            cost_gamma = trial.suggest_float("cost_gamma", 0.90, 0.93)  # Around 0.917
            cost_lam = trial.suggest_float("cost_lam", 0.89, 0.92)  # Around 0.909
            clip_eps = trial.suggest_float("clip_eps", 0.28, 0.37)  # Around 0.326
            ent_coef = trial.suggest_float("ent_coef", 0.09, 0.15)  # Around 0.119
            epochs = trial.suggest_int("epochs", 10, 12)  # Around 11
            # Keep same categorical choices to avoid Optuna TPE indexing errors
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            budget = trial.suggest_float("budget", 0.25, 0.35)  # Around 0.298
            nu_lr = trial.suggest_float("nu_lr", 5e-4, 1e-3, log=True)  # Around 0.000752
        elif env_name == 'ALE/Seaquest-v5':
            # Seaquest CMDP parameters are already set in the Seaquest-specific section above
            # Don't override them here to avoid log configuration conflicts
            pass
        else:
            # Default ranges for other environments
            cost_gamma = trial.suggest_float("cost_gamma", 0.90, 0.999)
            cost_lam = trial.suggest_float("cost_lam", 0.90, 0.999)
            clip_eps = trial.suggest_float("clip_eps", 0.1, 0.3)
            ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
            epochs = trial.suggest_int("epochs", 1, 10)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            budget = trial.suggest_float("budget", 0.10, 0.50)
            nu_lr = trial.suggest_float("nu_lr", 1e-4, 1e-2, log=True)
        
        # Only update agent_kwargs if we're not in Seaquest (Seaquest params already set)
        if env_name != 'ALE/Seaquest-v5':
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
    # If use_subprocess, spawn subprocess for training/evaluation to free memory
    if use_subprocess:
        # Serialize parameters for subprocess
        trial_data = {
            'agent_kwargs': agent_kwargs,
            'env_name': env_name,
            'method': method,
            'target_reward': target_reward,
            'num_train_episodes': num_train_episodes,
            'num_eval_episodes': num_eval_episodes,
            'max_episode_steps': max_episode_steps,
            'use_ram_obs': use_ram_obs
        }
        
        # Write to temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(trial_data, temp_file)
        temp_file.close()
        
        # Spawn subprocess
        script_path = os.path.abspath(__file__)
        cmd = [
            sys.executable, script_path,
            '--subprocess-trial',
            '--trial-data', temp_file.name
        ]
        
        # Write result to a file so we can read it after subprocess completes
        result_file = temp_file.name.replace('.json', '_result.txt')
        
        try:
            # Run subprocess - output will print naturally to terminal
            # The subprocess writes the result to a file which we read after
            print(f"\n[Subprocess] Starting trial evaluation (output will appear below)...")
            result_code = subprocess.run(cmd, check=False)
            print(f"[Subprocess] Trial evaluation completed (exit code: {result_code.returncode})\n")
            
            if result_code.returncode == 0:
                # Read result from file
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)
                        result = result_data['result']
                        avg_reward = result_data.get('actual_reward', None)
                        if avg_reward is not None:
                            trial.set_user_attr("actual_reward", avg_reward)
                            trial.set_user_attr("target_reward", target_reward)
                            reward_diff = abs(avg_reward - target_reward)
                            trial.set_user_attr("reward_diff", reward_diff)
                            print(f"[Subprocess] Trial result: reward={avg_reward:.2f}, normalized_diff={-result:.4f}")
                        else:
                            print(f"[Subprocess] Warning: No actual_reward in result file")
                else:
                    # Fallback: try to read from stdout if file doesn't exist
                    print(f"[Subprocess] Warning: Result file not found, subprocess may have failed")
                    result = -1000.0
            else:
                print(f"[Subprocess] Failed with exit code {result_code.returncode}")
                result = -1000.0
        finally:
            # Clean up temp files
            try:
                os.unlink(temp_file.name)
            except:
                pass
            try:
                if os.path.exists(result_file):
                    os.unlink(result_file)
            except:
                pass
        
        # Force garbage collection
        gc.collect()
        return result
    
    # Normal execution (no subprocess)
    agent = None
    env = None
    try:
        # Get training target reward for early stopping
        training_target = TRAINING_TARGET_REWARDS.get(env_name, target_reward)
        # Environment-specific early stopping patience (longer for Seaquest)
        early_stop_patience = 250 if env_name == 'ALE/Seaquest-v5' else 100
        
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
            early_stop_patience=early_stop_patience,  # Environment-specific early stopping
            target_reward=training_target,  # Stop training when target reward is reached
            max_episode_steps=max_episode_steps,
            use_ram_obs=use_ram_obs
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
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return result


def tune_method(env_name, method, target_reward, n_trials=30, num_train_episodes=500, num_eval_episodes=50, max_episode_steps=None, use_ram_obs=False, use_subprocess=False):
    """
    Tune hyperparameters for a specific method on a specific environment.
    """
    method_name = method['name']
    env_safe = env_name.replace('/', '_')
    
    # Add version suffix to study name to avoid conflicts with old trials
    # Change this version number when you modify hyperparameter ranges
    study_version = "v11"  # Increment this when changing hyperparameter ranges (v11 = Seaquest CMDP anti-collapse: much higher ent_coef/budget, much lower nu_lr)
    study_name = f"ijcai_{method_name}_{env_safe}_{study_version}"
    
    storage = f"sqlite:///optuna_ijcai_{method_name}_{env_safe}_{study_version}.db"
    
    # Check if database file exists and is writable
    db_path = f"optuna_ijcai_{method_name}_{env_safe}_{study_version}.db"
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
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
    except Exception as e:
        print(f"Error creating/loading study: {e}")
        print("Attempting to create new study (will overwrite existing)...")
        # Try creating without load_if_exists
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
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
                num_eval_episodes=num_eval_episodes,
                max_episode_steps=max_episode_steps,
                use_ram_obs=use_ram_obs,
                use_subprocess=use_subprocess
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
        # Environment-specific early stopping patience (longer for Seaquest)
        early_stop_patience = 250 if env_name == 'ALE/Seaquest-v5' else 100
        
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
            early_stop_patience=early_stop_patience,  # Environment-specific early stopping
            target_reward=training_target,  # Stop training when target reward is reached
            max_episode_steps=max_episode_steps,
            use_ram_obs=use_ram_obs
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
  
  # Tune Seaquest with step cap for faster tuning (default: 1000 steps)
  python scripts/tune_ijcai_methods.py --env ALE/Seaquest-v5 --trials 20 --max_episode_steps 1000
  
  # Tune Seaquest with RAM observations
  python scripts/tune_ijcai_methods.py --env ALE/Seaquest-v5 --trials 20 --max_episode_steps 2000 --use_ram_obs
        """
    )
    parser.add_argument('--env', type=str, required=False,  # Not required when --subprocess-trial is used
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
    parser.add_argument('--max_episode_steps', type=int, default=None,
                       help='Maximum steps per episode (default: env-specific, 1000 for Seaquest)')
    parser.add_argument('--use_ram_obs', action='store_true',
                       help='Use RAM observations instead of image observations (for Atari games)')
    parser.add_argument('--use_subprocess', action='store_true',
                       help='Run each trial in a separate subprocess to free memory between runs')
    # Internal flag for subprocess mode (not shown in help)
    parser.add_argument('--subprocess-trial', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--trial-data', type=str, default=None, help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # Handle subprocess trial mode (when called with --subprocess-trial)
    if args.subprocess_trial:
        if args.trial_data is None:
            print("Error: --subprocess-trial requires --trial-data")
            sys.exit(1)
        
        # Load trial data
        with open(args.trial_data, 'r') as f:
            trial_data = json.load(f)
        
        # Run training and evaluation
        try:
            training_target = TRAINING_TARGET_REWARDS.get(trial_data['env_name'], trial_data['target_reward'])
            # Environment-specific early stopping patience (longer for Seaquest)
            early_stop_patience = 250 if trial_data['env_name'] == 'ALE/Seaquest-v5' else 100
            
            agent, episode_rewards, best_weights, best_avg_reward, env = train(
                agent=trial_data['method']['agent'],
                env_name=trial_data['env_name'],
                num_episodes=trial_data['num_train_episodes'],
                use_shield_post=trial_data['method']['use_shield_post'],
                use_shield_pre=trial_data['method']['use_shield_pre'],
                use_shield_layer=trial_data['method']['use_shield_layer'],
                monitor_constraints=True,
                mode=trial_data['method']['mode'],
                verbose=False,
                visualize=False,
                render=False,
                seed=42,
                agent_kwargs=trial_data['agent_kwargs'],
                early_stop_patience=early_stop_patience,  # Environment-specific early stopping
                target_reward=training_target,
                max_episode_steps=trial_data['max_episode_steps'],
                use_ram_obs=trial_data['use_ram_obs']
            )
            
            # Load best weights
            if hasattr(agent, 'load_weights') and best_weights is not None:
                agent.load_weights(best_weights)
            
            # Evaluate
            results = evaluate_policy(
                agent,
                env,
                num_episodes=trial_data['num_eval_episodes'],
                visualize=False,
                render=False,
                force_disable_shield=False,
                softness=trial_data['method']['mode']
            )
            
            avg_reward = results['avg_reward']
            reward_diff = abs(avg_reward - trial_data['target_reward'])
            normalized_diff = reward_diff / (abs(trial_data['target_reward']) + 1e-6)
            result = -normalized_diff
            
            # Write result to file for parent process to read
            result_file = args.trial_data.replace('.json', '_result.txt')
            with open(result_file, 'w') as f:
                json.dump({
                    'result': result,
                    'actual_reward': avg_reward,
                    'target_reward': trial_data['target_reward'],
                    'reward_diff': reward_diff
                }, f)
            
            # Cleanup
            if agent is not None:
                if hasattr(agent, 'memory'):
                    agent.memory.clear()
                if hasattr(agent, 'constraint_monitor'):
                    agent.constraint_monitor.reset_all()
                del agent
            if env is not None:
                env.close()
                del env
            
            gc.collect()
            sys.exit(0)
        except Exception as e:
            print(f"Error in subprocess trial: {e}")
            import traceback
            traceback.print_exc()
            # Write error result to file
            result_file = args.trial_data.replace('.json', '_result.txt')
            try:
                with open(result_file, 'w') as f:
                    json.dump({
                        'result': -1000.0,
                        'actual_reward': None,
                        'target_reward': trial_data.get('target_reward', 0),
                        'reward_diff': None,
                        'error': str(e)
                    }, f)
            except:
                pass
            sys.exit(1)
    
    # Validate that --env is provided (unless in subprocess mode)
    if args.env is None:
        parser.error("--env is required (unless using --subprocess-trial)")
    
    # Set default max_episode_steps for Seaquest if not specified
    if args.max_episode_steps is None and args.env == 'ALE/Seaquest-v5':
        args.max_episode_steps = 1000  # Reduced from 10000 for faster tuning
    
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
    if args.max_episode_steps is not None:
        print(f"# Max episode steps: {args.max_episode_steps}")
    if args.use_ram_obs:
        print(f"# Using RAM observations")
    if args.use_subprocess:
        print(f"# Using subprocess mode (memory isolation between trials)")
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
                num_eval_episodes=args.eval_episodes,
                max_episode_steps=args.max_episode_steps,
                use_ram_obs=args.use_ram_obs,
                use_subprocess=args.use_subprocess
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

