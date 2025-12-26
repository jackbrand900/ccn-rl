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
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import train, evaluate_policy

# Target rewards per environment (to be tuned to match)
# Since we're focusing on violation/modification rates, we'll use more realistic targets
# that allow fair comparison even if absolute rewards are lower
TARGET_REWARDS = {
    'CartPole-v1': 150.0,  # Realistic target for constrained methods
    'CliffWalking-v1': -15.0,  # Slightly worse than optimal (-13)
    'MiniGrid-DoorKey-5x5-v0': 0.7,  # Realistic target
    'ALE/Seaquest-v5': 800.0,  # Realistic target
}

# Methods to tune (matching run_ijcai_experiments.py)
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
            agent_kwargs=agent_kwargs
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
        show_progress_bar=True
    )
    
    print(f"\n{'='*80}")
    print(f"Best trial for {method['display_name']} on {env_name}:")
    trial = study.best_trial
    print(f"  Reward difference: {-trial.value:.4f} (normalized)")
    print(f"  Params: {trial.params}")
    
    # Test the best params to get actual reward
    print(f"\n  Testing best parameters...")
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
        agent_kwargs=best_agent_kwargs
    )
    
    if hasattr(agent, 'load_weights') and best_weights is not None:
        agent.load_weights(best_weights)
    
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
    print(f"  Actual reward: {actual_reward:.2f} (target: {target_reward:.2f}, diff: {abs(actual_reward - target_reward):.2f})")
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
    parser.add_argument('--trials', type=int, default=20,
                       help='Number of Optuna trials per method')
    parser.add_argument('--train_episodes', type=int, default=500,
                       help='Number of training episodes during tuning')
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
    print(f"{'#'*80}\n")
    
    results_summary = []
    
    for method in methods_to_tune:
        try:
            params, actual_reward = tune_method(
                env_name=args.env,
                method=method,
                target_reward=target_reward,
                n_trials=args.trials,
                num_train_episodes=args.train_episodes,
                num_eval_episodes=args.eval_episodes
            )
            results_summary.append({
                'method': method['display_name'],
                'target_reward': target_reward,
                'actual_reward': actual_reward,
                'diff': abs(actual_reward - target_reward)
            })
        except Exception as e:
            print(f"Error tuning {method['display_name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'#'*80}")
    print("# Tuning Summary")
    print(f"{'#'*80}")
    print(f"{'Method':<30} {'Target':<12} {'Actual':<12} {'Diff':<12}")
    print("-" * 80)
    for r in results_summary:
        print(f"{r['method']:<30} {r['target_reward']:<12.2f} {r['actual_reward']:<12.2f} {r['diff']:<12.2f}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()

