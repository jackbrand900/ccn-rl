#!/usr/bin/env python3
"""Tune CCN+ (PiShield shield-layer) PPO on the wrapper-free Acrobot-v1.

Search is centred on the known-good unshielded config (optuna trial 4 of
ppo_unshielded_Acrobot-v1_v18: lr 1.33e-4, gamma 0.956, hidden 256,
clip 0.108, ent 0.044, epochs 7, batch 64 -> reward -108.6) but kept wide.
Hard shielding removes the over-spin torque that energy-pumping swing-up
relies on, so the policy needs MORE exploration than the unshielded baseline
to find a constrained swing-up -- hence the widened entropy / lr / gamma
ranges rather than a tight local search.

The known-good config is enqueued as the first trial so the study can never
end up worse than the trial-4 neighbourhood.

Objective (matches scripts/tune_methods.py): maximise
    -|avg_reward - target| / |target|
i.e. drive eval reward AS CLOSE TO `target` AS POSSIBLE. NOTE this rewards
*matching* the target, not exceeding it -- see --target help below.

Usage:
    python -m scripts.tune_acrobot_ccn --method ppo_layer_hard --trials 40
    python -m scripts.tune_acrobot_ccn --method ppo_layer_soft --trials 40 --target -100
"""
import argparse
import gc
import os
import sys
from pathlib import Path

import optuna
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.train import train, evaluate_policy  # noqa: E402

ENV_NAME = "Acrobot-v1"

# CCN+ shield-layer methods this script knows how to tune.
METHODS = {
    "ppo_layer_hard": {"use_shield_layer": True, "mode": "hard"},
    "ppo_layer_soft": {"use_shield_layer": True, "mode": "soft"},
}

# Known-good unshielded config (optuna trial 4) -- enqueued as trial 0 and used
# as the centre of the search ranges below.
TRIAL4 = {
    "lr": 0.0001331707625849592,
    "gamma": 0.9555122888613474,
    "hidden_dim": 256,
    "use_orthogonal_init": True,
    "num_layers": 2,
    "clip_eps": 0.10786489226725048,
    "ent_coef": 0.04431278592736043,
    "epochs": 7,
    "batch_size": 64,
}


def suggest_params(trial):
    """Wide ranges centred on trial 4. Entropy/lr/gamma are deliberately broad
    because the hard shield needs extra exploration to recover swing-up."""
    return {
        # spans trial4's 1.33e-4 with ~7x headroom either side
        "lr": trial.suggest_float("lr", 5e-5, 1e-3, log=True),
        # high gamma matters for sparse swing-up reward; trial4 was 0.956
        "gamma": trial.suggest_float("gamma", 0.93, 0.999),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        "use_orthogonal_init": trial.suggest_categorical("use_orthogonal_init", [True, False]),
        "num_layers": trial.suggest_int("num_layers", 2, 3),
        "clip_eps": trial.suggest_float("clip_eps", 0.1, 0.3),
        # KEY exploration knob: much wider than the unshielded default (0-0.05).
        # log scale spans trial4's 0.044 up to 0.3 for shield-constrained search.
        "ent_coef": trial.suggest_float("ent_coef", 0.01, 0.3, log=True),
        "epochs": trial.suggest_int("epochs", 3, 10),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }


def make_objective(method, target, train_episodes, eval_episodes, seed):
    cfg = METHODS[method]

    def objective(trial):
        agent = env = None
        try:
            agent_kwargs = suggest_params(trial)
            agent, _, best_weights, _, env = train(
                agent="ppo",
                env_name=ENV_NAME,
                num_episodes=train_episodes,
                use_shield_layer=cfg["use_shield_layer"],
                mode=cfg["mode"],
                monitor_constraints=True,
                verbose=False,
                seed=seed,
                agent_kwargs=agent_kwargs,
                early_stop_patience=100,
                target_reward=target,
            )
            if hasattr(agent, "load_weights") and best_weights is not None:
                agent.load_weights(best_weights)

            results = evaluate_policy(
                agent, env, num_episodes=eval_episodes,
                force_disable_shield=False, softness=cfg["mode"],
            )
            avg_reward = results["avg_reward"]
            viol_rate = results.get("avg_violations_per_step", float("nan"))

            trial.set_user_attr("actual_reward", float(avg_reward))
            trial.set_user_attr("viol_rate", float(viol_rate))
            trial.set_user_attr("target_reward", target)

            # maximise negative normalised distance from target
            score = -abs(avg_reward - target) / (abs(target) + 1e-6)
            print(f"[trial {trial.number}] reward={avg_reward:.1f} "
                  f"viol_rate={viol_rate:.4f} score={score:.4f}")
            return score
        except Exception as e:  # noqa: BLE001
            print(f"[trial {trial.number}] ERROR: {e}")
            return -1000.0
        finally:
            if agent is not None:
                if hasattr(agent, "memory"):
                    agent.memory.clear()
                if hasattr(agent, "constraint_monitor"):
                    agent.constraint_monitor.reset_all()
                del agent
            if env is not None:
                env.close()
                del env
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective


def main():
    p = argparse.ArgumentParser(description="Tune CCN+ PPO on wrapper-free Acrobot.")
    p.add_argument("--method", choices=list(METHODS), default="ppo_layer_hard")
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--train-episodes", type=int, default=500)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--version", default="v1", help="study version suffix")
    p.add_argument(
        "--target", type=float, default=-100.0,
        help="Reward to MATCH (not exceed). The objective minimises |reward-target|, "
             "so a config that reaches a HIGHER reward than target scores WORSE than "
             "one that lands exactly on it. Keep this at or slightly above the best "
             "reward the shielded method can actually achieve (Acrobot solves at -100; "
             "unshielded reaches -108). Setting it well below the achievable frontier "
             "(e.g. -200) selects for a deliberately worse policy -- see module docstring.",
    )
    args = p.parse_args()

    study_name = f"acrobot_ccn_{args.method}_{args.version}"
    storage = f"sqlite:///optuna_{study_name}.db"
    study = optuna.create_study(
        direction="maximize", study_name=study_name,
        storage=storage, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    # Seed the known-good config as the first trial (only if not already run).
    if len(study.trials) == 0:
        study.enqueue_trial(TRIAL4)
        print(f"Enqueued trial-4 config as the first trial.")

    print(f"Tuning {args.method} on {ENV_NAME} | target={args.target} | "
          f"trials={args.trials} | train_eps={args.train_episodes}")
    study.optimize(
        make_objective(args.method, args.target, args.train_episodes,
                       args.eval_episodes, args.seed),
        n_trials=args.trials,
    )

    best = study.best_trial
    print("\n" + "=" * 70)
    print(f"BEST: reward={best.user_attrs.get('actual_reward')} "
          f"viol_rate={best.user_attrs.get('viol_rate')} score={best.value:.4f}")
    print("params:", best.params)

    config_dir = Path("config/tuned")
    config_dir.mkdir(parents=True, exist_ok=True)
    out = config_dir / f"{args.method}_{ENV_NAME}_params.yaml"
    with open(out, "w") as f:
        yaml.dump(best.params, f, default_flow_style=False)
    print(f"Saved best params -> {out}")


if __name__ == "__main__":
    main()
