#!/usr/bin/env python3
"""
Sanity benchmark: our PPOAgent(use_action_mask=True) vs. sb3-contrib's MaskablePPO.

Both methods:
  - run on the same env (default: CartPole-v1)
  - source their action mask from the same ShieldController.compute_action_mask
    (so the constraint set is identical; only the PPO machinery differs)
  - train for the same total_timesteps budget
  - are evaluated on the same number of episodes
  - report reward + (would-have-violated) violation rate + modification rate

Disagreement => one of the two impls has a bug.
Agreement => our action_mask number is real, not an implementation artifact.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import deque

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.agents.ppo_agent import PPOAgent
from src.train import create_environment, preprocess_state, set_seed
from src.utils.constraint_monitor import ConstraintMonitor
from src.utils.shield_controller import ShieldController
import src.utils.context_provider as context_provider


# Map env -> (CNF, action_dim, num_flags) so the bench is env-agnostic.
ENV_CONFIG = {
    "CartPole-v1": {
        "requirements": "src/requirements/emergency_cartpole.cnf",
        "action_dim": 2,
    },
    "CliffWalking-v1": {
        "requirements": "src/requirements/cliff_safe.cnf",
        "action_dim": 4,
    },
}


# ----------------------------------------------------------------------
# Shared context builder (mask source). Wraps context_provider so the
# sb3 wrapper has access without an "agent" object.
# ----------------------------------------------------------------------
class _LastObsHolder:
    """Minimal stand-in for an agent: context_provider only reads .last_obs."""
    def __init__(self):
        self.last_obs = None
        self.learn_step_counter = 0


def build_context(env, last_obs):
    holder = _LastObsHolder()
    holder.last_obs = last_obs
    return context_provider.build_context(env, holder)


# ----------------------------------------------------------------------
# sb3-contrib MaskablePPO branch
# ----------------------------------------------------------------------
class MaskedEnv(gym.Wrapper):
    """gym wrapper exposing action_masks() via our ShieldController."""

    def __init__(self, env, shield_controller):
        super().__init__(env)
        self.shield_controller = shield_controller
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def action_masks(self):
        ctx = build_context(self.env, self._last_obs)
        return self.shield_controller.compute_action_mask(ctx).cpu().numpy()


def train_sb3_maskable(env_name, seed, total_timesteps, hp):
    base = create_environment(env_name, seed=seed)
    cfg = ENV_CONFIG[env_name]
    sc = ShieldController(cfg["requirements"], cfg["action_dim"], mode="hard",
                          is_shield_active=True)
    env = MaskedEnv(base, sc)

    # Independent monitor so we measure "would-have-violated" rate fairly.
    monitor = ConstraintMonitor(verbose=False)
    sc.constraint_monitor = monitor

    model = MaskablePPO(
        "MlpPolicy", env, seed=seed, verbose=0,
        learning_rate=hp["lr"], gamma=hp["gamma"],
        n_steps=hp["n_steps"], batch_size=hp["batch_size"],
        n_epochs=hp["n_epochs"], clip_range=hp["clip_range"],
        ent_coef=hp["ent_coef"],
    )

    # During learn(), wrap predict+step ourselves to track viol/mod rates.
    # MaskablePPO's collect_rollouts is internal, so we just measure
    # post-training behavior via our own eval loop instead.
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    return _eval_policy_sb3(model, env, sc, monitor, n_episodes=100)


def _eval_policy_sb3(model, env, sc, monitor, n_episodes):
    monitor.reset_all()
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        monitor.reset()
        done = False
        ep_r = 0.0
        while not done:
            mask = env.action_masks()
            # Sample raw + masked from the policy distribution (so we can
            # log "would-have-violated" via the unshielded action).
            action_masked, _ = model.predict(obs, action_masks=mask, deterministic=False)
            # For an unshielded sample, query the model with an all-true mask.
            action_unmasked, _ = model.predict(
                obs, action_masks=np.ones_like(mask, dtype=bool), deterministic=False
            )
            ctx = build_context(env, obs)
            # Manual viol/mod logging (we can't access internal probs here,
            # so feed dummy probs; the monitor only uses actions for is_shield_active).
            num_a = sc.num_actions
            dummy = torch.full((num_a,), 1.0 / num_a)
            monitor.log_step_from_probs_and_actions(
                raw_probs=dummy, corrected_probs=dummy,
                a_unshielded=int(action_unmasked), a_shielded=int(action_masked),
                context=ctx, shield_controller=sc,
            )
            obs, reward, terminated, truncated, _ = env.step(int(action_masked))
            done = terminated or truncated
            ep_r += reward
        rewards.append(ep_r)
    stats = monitor.summary()
    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "viol_rate": stats["total_viol_rate"],
        "mod_rate": stats["total_mod_rate"],
    }


# ----------------------------------------------------------------------
# Our PPOAgent branch
# ----------------------------------------------------------------------
def train_ours(env_name, seed, total_timesteps, hp):
    env = create_environment(env_name, seed=seed)
    set_seed(seed)

    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Discrete):
        input_shape = (obs_space.n,)
        action_dim = env.action_space.n
    else:
        input_shape = obs_space.shape
        action_dim = env.action_space.n

    cfg = ENV_CONFIG[env_name]
    agent = PPOAgent(
        input_shape=input_shape,
        action_dim=action_dim,
        use_action_mask=True,
        use_orthogonal_init=True,
        agent_kwargs={
            "lr": hp["lr"], "gamma": hp["gamma"],
            "clip_eps": hp["clip_range"], "ent_coef": hp["ent_coef"],
            "epochs": hp["n_epochs"], "batch_size": hp["batch_size"],
            "hidden_dim": 64, "num_layers": 2,
        },
        monitor_constraints=True,
        mode="hard",
        verbose=False,
        requirements_path=cfg["requirements"],
        env=env,
        use_cnn=False,
    )

    steps = 0
    while steps < total_timesteps:
        state, _ = env.reset()
        state = preprocess_state(state)
        done = False
        while not done and steps < total_timesteps:
            sel, a_un, a_sh, ctx = agent.select_action(state)
            next_state, reward, term, trunc, _ = env.step(sel)
            done = term or trunc
            next_state = preprocess_state(next_state)
            agent.store_transition(state, sel, reward, next_state, ctx, done)
            agent.update()
            state = next_state
            steps += 1

    # Evaluation loop with the existing constraint monitor.
    return _eval_policy_ours(agent, env, n_episodes=100)


def _eval_policy_ours(agent, env, n_episodes):
    agent.constraint_monitor.reset_all()
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        agent.constraint_monitor.reset()
        done = False
        ep_r = 0.0
        while not done:
            sel, _, _, _ = agent.select_action(state)
            state, reward, term, trunc, _ = env.step(sel)
            state = preprocess_state(state)
            done = term or trunc
            ep_r += reward
        rewards.append(ep_r)
    stats = agent.constraint_monitor.summary()
    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "viol_rate": stats["total_viol_rate"],
        "mod_rate": stats["total_mod_rate"],
    }


# ----------------------------------------------------------------------
# Vanilla PPO (no masking) — diagnostic to check if any reward gap is
# general PPO machinery or masking-specific.
# ----------------------------------------------------------------------
def train_sb3_vanilla(env_name, seed, total_timesteps, hp):
    env = create_environment(env_name, seed=seed)
    model = SB3PPO(
        "MlpPolicy", env, seed=seed, verbose=0,
        learning_rate=hp["lr"], gamma=hp["gamma"],
        n_steps=hp["n_steps"], batch_size=hp["batch_size"],
        n_epochs=hp["n_epochs"], clip_range=hp["clip_range"],
        ent_coef=hp["ent_coef"],
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    rewards = []
    for _ in range(100):
        obs, _ = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, term, trunc, _ = env.step(int(action))
            done = term or trunc
            ep_r += reward
        rewards.append(ep_r)
    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "viol_rate": 0.0,
        "mod_rate": 0.0,
    }


def train_ours_vanilla(env_name, seed, total_timesteps, hp):
    env = create_environment(env_name, seed=seed)
    set_seed(seed)

    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Discrete):
        input_shape = (obs_space.n,)
        action_dim = env.action_space.n
    else:
        input_shape = obs_space.shape
        action_dim = env.action_space.n

    cfg = ENV_CONFIG[env_name]
    agent = PPOAgent(
        input_shape=input_shape,
        action_dim=action_dim,
        use_shield_post=False,
        use_shield_pre=False,
        use_shield_layer=False,
        use_action_mask=False,
        use_orthogonal_init=True,
        agent_kwargs={
            "lr": hp["lr"], "gamma": hp["gamma"],
            "clip_eps": hp["clip_range"], "ent_coef": hp["ent_coef"],
            "epochs": hp["n_epochs"], "batch_size": hp["batch_size"],
            "hidden_dim": 64, "num_layers": 2,
        },
        monitor_constraints=False,
        mode="hard",
        verbose=False,
        # ShieldController still needs a CNF file even when unused.
        requirements_path=cfg["requirements"],
        env=env,
        use_cnn=False,
    )

    steps = 0
    while steps < total_timesteps:
        state, _ = env.reset()
        state = preprocess_state(state)
        done = False
        while not done and steps < total_timesteps:
            sel, _, _, ctx = agent.select_action(state)
            next_state, reward, term, trunc, _ = env.step(sel)
            done = term or trunc
            next_state = preprocess_state(next_state)
            agent.store_transition(state, sel, reward, next_state, ctx, done)
            agent.update()
            state = next_state
            steps += 1

    rewards = []
    for _ in range(100):
        state, _ = env.reset()
        state = preprocess_state(state)
        done = False
        ep_r = 0.0
        while not done:
            sel, _, _, _ = agent.select_action(state)
            state, reward, term, trunc, _ = env.step(sel)
            state = preprocess_state(state)
            done = term or trunc
            ep_r += reward
        rewards.append(ep_r)
    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "viol_rate": 0.0,
        "mod_rate": 0.0,
    }


# ----------------------------------------------------------------------
# Matched-reward (early-stopping) variants. Train both methods until a
# rolling-window mean reaches the target, then evaluate. This mirrors the
# methodology: methods are compared at *matched performance*, not
# at saturation, so the safety metrics are read at the same reward level.
# ----------------------------------------------------------------------
class _SB3TargetCallback(BaseCallback):
    def __init__(self, target_reward, window):
        super().__init__(verbose=0)
        self.target_reward = target_reward
        self.window = window
        self.reached = False

    def _on_step(self):
        buf = self.model.ep_info_buffer
        if buf is not None and len(buf) >= self.window:
            recent = list(buf)[-self.window:]
            mean_r = float(np.mean([e["r"] for e in recent]))
            if mean_r >= self.target_reward:
                self.reached = True
                return False
        return True


def train_sb3_maskable_target(env_name, seed, target_reward, max_timesteps, hp, window=20):
    base = create_environment(env_name, seed=seed)
    cfg = ENV_CONFIG[env_name]
    sc = ShieldController(cfg["requirements"], cfg["action_dim"], mode="hard",
                          is_shield_active=True)
    env = MaskedEnv(base, sc)
    monitor = ConstraintMonitor(verbose=False)
    sc.constraint_monitor = monitor

    model = MaskablePPO(
        "MlpPolicy", env, seed=seed, verbose=0,
        learning_rate=hp["lr"], gamma=hp["gamma"],
        n_steps=hp["n_steps"], batch_size=hp["batch_size"],
        n_epochs=hp["n_epochs"], clip_range=hp["clip_range"],
        ent_coef=hp["ent_coef"],
    )
    cb = _SB3TargetCallback(target_reward, window)
    model.learn(total_timesteps=max_timesteps, callback=cb, progress_bar=False)
    train_steps = int(model.num_timesteps)

    eval_results = _eval_policy_sb3(model, env, sc, monitor, n_episodes=100)
    eval_results["train_steps"] = train_steps
    eval_results["reached_target"] = cb.reached
    return eval_results


def train_ours_target(env_name, seed, target_reward, max_timesteps, hp, window=20):
    env = create_environment(env_name, seed=seed)
    set_seed(seed)
    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Discrete):
        input_shape = (obs_space.n,)
    else:
        input_shape = obs_space.shape
    action_dim = env.action_space.n

    cfg = ENV_CONFIG[env_name]
    agent = PPOAgent(
        input_shape=input_shape, action_dim=action_dim,
        use_action_mask=True, use_orthogonal_init=True,
        agent_kwargs={
            "lr": hp["lr"], "gamma": hp["gamma"],
            "clip_eps": hp["clip_range"], "ent_coef": hp["ent_coef"],
            "epochs": hp["n_epochs"], "batch_size": hp["batch_size"],
            "hidden_dim": 64, "num_layers": 2,
        },
        monitor_constraints=True, mode="hard", verbose=False,
        requirements_path=cfg["requirements"], env=env, use_cnn=False,
    )

    recent = deque(maxlen=window)
    steps = 0
    reached = False
    while steps < max_timesteps:
        state, _ = env.reset()
        state = preprocess_state(state)
        done = False
        ep_r = 0.0
        while not done and steps < max_timesteps:
            sel, _, _, ctx = agent.select_action(state)
            next_state, reward, term, trunc, _ = env.step(sel)
            done = term or trunc
            next_state = preprocess_state(next_state)
            agent.store_transition(state, sel, reward, next_state, ctx, done)
            agent.update()
            state = next_state
            ep_r += reward
            steps += 1
        recent.append(ep_r)
        if len(recent) == window and float(np.mean(recent)) >= target_reward:
            reached = True
            break

    eval_results = _eval_policy_ours(agent, env, n_episodes=100)
    eval_results["train_steps"] = steps
    eval_results["reached_target"] = reached
    return eval_results


# ----------------------------------------------------------------------
# Bench driver
# ----------------------------------------------------------------------
def run_bench(env_name, seeds, total_timesteps, mode="masked", target_reward=None):
    # Hyperparameters held identical between both methods (sb3 defaults).
    hp = {
        "lr": 3e-4, "gamma": 0.99,
        "n_steps": 2048, "batch_size": 64,
        "n_epochs": 10, "clip_range": 0.2,
        "ent_coef": 0.0,
    }

    print(f"\n{'='*72}")
    print(f"Benchmark [{mode}]: {env_name} | budget={total_timesteps} | seeds={seeds}")
    if mode == "matched":
        print(f"Early stopping at rolling-mean reward >= {target_reward}")
    print(f"Hyperparameters: {hp}")
    print(f"{'='*72}\n")

    if mode == "masked":
        methods = [
            ("ours (PPOAgent + use_action_mask)", train_ours),
            ("sb3-contrib MaskablePPO", train_sb3_maskable),
        ]
        runner = lambda fn, seed: fn(env_name, seed, total_timesteps, hp)
    elif mode == "vanilla":
        methods = [
            ("ours (PPOAgent vanilla)", train_ours_vanilla),
            ("sb3 PPO vanilla", train_sb3_vanilla),
        ]
        runner = lambda fn, seed: fn(env_name, seed, total_timesteps, hp)
    elif mode == "matched":
        methods = [
            ("ours (PPOAgent + use_action_mask, early-stop)", train_ours_target),
            ("sb3-contrib MaskablePPO (early-stop)", train_sb3_maskable_target),
        ]
        runner = lambda fn, seed: fn(env_name, seed, target_reward, total_timesteps, hp)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    table = {name: [] for name, _ in methods}

    for seed in seeds:
        for name, fn in methods:
            print(f"  [seed={seed}] {name} ...")
            t0 = time.time()
            res = runner(fn, seed)
            res["wall_s"] = time.time() - t0
            table[name].append(res)
            extras = ""
            if "train_steps" in res:
                extras = f" steps={res['train_steps']:>6} reached={res.get('reached_target', '?')}"
            print(f"    reward={res['reward_mean']:.2f}±{res['reward_std']:.2f} "
                  f"viol_rate={res['viol_rate']:.4f} mod_rate={res['mod_rate']:.4f}"
                  f"{extras} ({res['wall_s']:.1f}s)")

    print(f"\n{'='*72}")
    print(f"SUMMARY ({env_name}, {len(seeds)} seeds, mode={mode})")
    print(f"{'='*72}")
    if mode == "matched":
        print(f"{'method':<48s} {'reward':>16s} {'viol_rate':>12s} {'mod_rate':>12s} {'avg_steps':>12s}")
    else:
        print(f"{'method':<48s} {'reward':>16s} {'viol_rate':>12s} {'mod_rate':>12s}")
    print("-" * 110)
    for name, results in table.items():
        rewards = [r["reward_mean"] for r in results]
        viols = [r["viol_rate"] for r in results]
        mods = [r["mod_rate"] for r in results]
        line = (f"{name:<48s} {np.mean(rewards):>8.2f} ± {np.std(rewards):>5.2f} "
                f"{np.mean(viols):>12.4f} {np.mean(mods):>12.4f}")
        if mode == "matched":
            steps = [r["train_steps"] for r in results]
            line += f" {int(np.mean(steps)):>12d}"
        print(line)
    print(f"{'='*72}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", choices=list(ENV_CONFIG.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="max timesteps per run (also used as the budget for masked/vanilla modes)")
    parser.add_argument("--mode", choices=["masked", "vanilla", "matched"], default="masked",
                        help="masked: action_mask vs MaskablePPO at full budget; "
                             "vanilla: plain PPO comparison; "
                             "matched: early-stop at target reward, compare safety at matched perf")
    parser.add_argument("--target-reward", type=float, default=200.0,
                        help="target reward for matched mode (default: 200, mirrors CartPole target)")
    args = parser.parse_args()
    run_bench(args.env, args.seeds, args.timesteps, mode=args.mode,
              target_reward=args.target_reward)


if __name__ == "__main__":
    main()
