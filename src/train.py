import json
from datetime import datetime
from random import random

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit, AtariPreprocessing
import torch
import src.utils.graphing as graphing
import argparse
from gymnasium.envs.registration import register
from minigrid.wrappers import FlatObsWrapper, FullyObsWrapper, RGBImgObsWrapper

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.a2c_agent import A2CAgent
from src.utils.config import config_by_env
from src.utils.env_helpers import find_key
import sys
import os
import ale_py

from src.utils.wrappers import RAMObservationWrapper, SeaquestRAMWrapper, DemonAttackRAMWrapper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
# import cv2

def set_seed(seed: int):
    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU + CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: ensure hash-based ops are deterministic
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # for CUDA >=10.2 deterministic matmul

    print(f"[Seed set to {seed}]")

register(
    id="CarRacingWithTrafficLights-v0",
    entry_point="src.envs.car_racing_with_lights:CarRacingWithTrafficLights",
)

def register_env_if_needed(env_id, entry_point, kwargs=None):
    if env_id not in gym.envs.registry.keys():
        register(
            id=env_id,
            entry_point=entry_point,
            kwargs=kwargs or {}
        )


custom_envs = {
    "MiniGrid-Empty-5x5-v0": ("minigrid.envs:EmptyEnv", {'size': 5}),
    "MiniGrid-DoorKey-5x5-v0": ("minigrid.envs:DoorKeyEnv", {'size': 5}),
    "MiniGrid-Empty-6x6-v0": ("minigrid.envs:EmptyEnv", {'size': 6}),
    "MiniGrid-DoorKey-6x6-v0": ("minigrid.envs:DoorKeyEnv", {'size': 6}),
    "MiniGrid-MultiRoom-N2-S4-v0": ("minigrid.envs:MultiRoomEnv", {'num_rooms': 2, 'max_room_size': 4}),
    "CartPole-v1": (None, None),
    "CarRacing-v3": (None, None),
    "CarRacingWithTrafficLights-v0": (None, None),
    "ALE/Freeway-v5": (None, None),
    "FreewayNoFrameskip-v4": (None, None),
    "ALE/Seaquest-v5": (None, None),
    "ALE/DemonAttack-v5": (None, None)
}

def create_environment(env_name, render=False, use_ram_obs=False, seed=42):
    if env_name in custom_envs:
        entry_point, kwargs = custom_envs[env_name]
        if entry_point:
            register_env_if_needed(env_name, entry_point, kwargs)

        # Handle CartPole rendering differently
        if env_name == "CartPole-v1":
            env = gym.make(env_name, render_mode="human" if render else None)
            env.env_name = env_name
            env.use_ram = False
            return env

        if env_name == "CarRacing-v3":
            env = gym.make(env_name, render_mode="human" if render else None, continuous=False)
            env = TimeLimit(env, max_episode_steps=200)
            env.env_name = env_name
            env.use_ram = False
            return env

        if env_name == "CarRacingWithTrafficLights-v0":
            env = gym.make(env_name, render_mode="human" if render else None, continuous=False)
            env = TimeLimit(env, max_episode_steps=300)
            env.env_name = env_name
            env.use_ram = False
            return env

        if env_name == "ALE/Freeway-v5":
            env = gym.make(env_name, render_mode="human" if render else None, frameskip=1, repeat_action_probability=0.0)
            env = AtariPreprocessing(env, frame_skip=8, scale_obs=True, terminal_on_life_loss=True)
            if use_ram_obs:
                env = RAMObservationWrapper(env)
            env = TimeLimit(env, max_episode_steps=3000)
            env.env_name = env_name
            env.use_ram = use_ram_obs
            return env

        if env_name == "ALE/Seaquest-v5":
            env = gym.make(env_name, render_mode="human" if render else None, frameskip=1)
            env = AtariPreprocessing(env, frame_skip=4, scale_obs=True)
            if use_ram_obs:
                env = RAMObservationWrapper(env)
            env = TimeLimit(env, max_episode_steps=10000)
            env.env_name = env_name
            env.use_ram = use_ram_obs
            return env

        if env_name == "ALE/DemonAttack-v5":
            env = gym.make(env_name, render_mode="human" if render else None, frameskip=1)
            env = AtariPreprocessing(env, frame_skip=4, scale_obs=True)
            if use_ram_obs:
                env = RAMObservationWrapper(env)
            # env = TimeLimit(env, max_episode_steps=1000)
            env.env_name = env_name
            env.use_ram = use_ram_obs
            return env

        # Handle MiniGrid environments
        env = gym.make(env_name, render_mode="human" if render else None)
        if "MiniGrid" in env_name:
            if render:
                env = RGBImgObsWrapper(env)
                env = FullyObsWrapper(env)
            else:
                env = FlatObsWrapper(env)
            env.env_name = env_name
            env.use_ram = False
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        return env

def log_ram(obs, prev_obs, step):
    print(f"\n[RAM] Step {step}: {obs.astype(int)}")
    if prev_obs is not None:
        delta = obs.astype(int) - prev_obs.astype(int)
        changed = np.nonzero(delta)[0]
        print(f"[RAM] Changed indices: {changed}, deltas: {delta[changed]}")


def preprocess_state(state, use_cnn=False):
    if isinstance(state, dict):
        state = state['image']
    if isinstance(state, np.ndarray):
        if use_cnn:
            if state.ndim == 2:
                # Grayscale image (H, W) → (1, H, W)
                state = np.expand_dims(state.astype(np.float32), axis=0)
            elif state.ndim == 3:
                if state.shape[-1] == 3:
                    # RGB image (H, W, C) → (C, H, W)
                    state = state.astype(np.float32).transpose(2, 0, 1)
                else:
                    # Already (C, H, W)
                    state = state.astype(np.float32)
            elif state.ndim == 4:
                # (stack, H, W, C) → (stack * C, H, W)
                stack, h, w, c = state.shape
                state = state.astype(np.float32).transpose(0, 3, 1, 2).reshape(stack * c, h, w)
            else:
                raise ValueError(f"Unexpected CNN state shape: {state.shape}")
        else:
            state = state.flatten().astype(np.float32)
    return state


def reset_env(env):
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        next_state, reward, done, info = result
    return next_state, reward, done, info


def run_training(agent, env, num_episodes=100, print_interval=10, monitor_constraints=True, visualize=False,
                 softness="", verbose=False, log_rewards=False, use_cnn=False, render=False):
    episode_rewards = []
    mod_rate_per_episode = []
    viol_rate_per_episode = []
    best_avg_reward = float('-inf')
    best_weights = None
    no_improve_counter = 0  # Early stopping counter
    early_stop_patience = 2000 # Stop if no improvement after 500 episodes

    if not softness:
        softness = ""
    else:
        softness = softness.capitalize()

    env_name = getattr(env, 'spec', None).id if hasattr(env, 'spec') else str(env)
    if getattr(agent, 'use_shield_layer', False):
        shield_mode = "Layered Shield"
    elif getattr(agent, 'use_shield_post', False):
        shield_mode = "Post-hoc Shield"
    elif getattr(agent, 'use_shield_pre', False):
        shield_mode = "Pre-emptive Shield"
    elif agent.lambda_sem > 0:
        shield_mode = "Semantic Loss"
    else:
        shield_mode = "Unshielded"

    for episode in range(1, num_episodes + 1):
        use_cnn = getattr(agent, "use_cnn", False)
        if hasattr(agent, "start_new_episode"):
            agent.start_new_episode()
        state, _ = env.reset()
        state = preprocess_state(state, use_cnn=use_cnn)
        try:
            key_pos = find_key(env)
            env.key_pos = key_pos
        except AttributeError:
            key_pos = None

        if agent.shield_controller.mode == "progressive":
            agent.shield_controller.set_episode(episode)

        done = False
        total_reward = 0
        step_count = 0
        while not done:
            result = agent.select_action(state)
            if isinstance(result, tuple) and len(result) == 4:
                selected_action, a_unshielded, a_shielded, context = result
                if agent.use_shield_pre or agent.use_shield_layer:
                    action_for_training = a_shielded
                else:
                    action_for_training = a_unshielded
            else:
                raise ValueError("Expected select_action to return 4-tuple (selected_action, a_unshielded, "
                                 "a_shielded, context)")

            step_count += 1
            # if isinstance(result, tuple) and len(result) == 3:
            #     action, log_prob, value = result
            #     context = {}
            # else:
            #     action, context = result
            #     log_prob = None
            #     value = None

            pos = context.get("position", None)
            x, y = pos if pos is not None else (None, None)
            # actions_taken.append((x, y, selected_action))
            next_state, reward, terminated, truncated, info = env.step(selected_action)

            if render:
                env.render()

            next_state = preprocess_state(next_state, use_cnn=use_cnn)

            done = terminated or truncated

            agent.store_transition(state, action_for_training, reward, next_state, context, done)
            agent.update()
            state = next_state
            total_reward += reward

        constraint_monitor = agent.constraint_monitor
        stats = constraint_monitor.summary()
        episode_mod_rate = stats['episode_mod_rate']
        episode_viol_rate = stats['episode_viol_rate']
        episode_rewards.append(total_reward)
        mod_rate_per_episode.append(episode_mod_rate)
        viol_rate_per_episode.append(episode_viol_rate)

        if print_interval and episode % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            use_ram = 'RAM' if env.use_ram else 'OBS'
            log_msg = (
                f"[{env.env_name}] "
                f"[{str(type(agent)).split('.')[-1][:-2]}] "
                f"[{shield_mode.capitalize()}] "
                f"[{softness.capitalize()}] "
                f"[{use_ram}], "
                f"Episode {episode}, "
                f"Avg Reward ({print_interval}): {avg_reward:.2f}, "
            )
            if monitor_constraints:
                constraint_monitor = agent.constraint_monitor
                stats = constraint_monitor.summary()
                total_mods = stats['total_modifications']
                episode_mods = stats['episode_modifications']
                total_violations = stats['total_violations']
                episode_viols = stats['episode_violations']
                mod_rate = stats['total_mod_rate']
                viol_rate = stats['total_viol_rate']
                monitor_logs = (f"Total Mods: {total_mods}, Mod Rate: {mod_rate:.3f}, "
                                f"Total Violations: {total_violations}, Viol Rate: {viol_rate: .3f}")
                log_msg += monitor_logs


            if hasattr(agent, "epsilon"):
                log_msg += f", Epsilon: {agent.epsilon:.3f}"
            print(log_msg)

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_weights = copy.deepcopy(agent.get_weights())
                no_improve_counter = 0
                print(f"[Checkpoint] New best avg reward: {best_avg_reward:.2f} at episode {episode}")
            else:
                no_improve_counter += print_interval
                if no_improve_counter >= early_stop_patience:
                    print(f"[Early Stopping] No improvement for {early_stop_patience} episodes. Stopping training.")
                    break


    if visualize:
        agent_name = agent.__class__.__name__

        title_prefix = f"{agent_name} on {env_name}, {shield_mode} {softness}"

        graphing.plot_rewards(
            rewards=episode_rewards,
            title=f"{title_prefix} – Training Reward Curve",
            xlabel="Episode",
            ylabel="Reward",
            rolling_window=20,
            show=True,
            run_dir=run_dir
        )

        graphing.plot_summary_metrics(
            rewards=episode_rewards,
            mod_rate=mod_rate_per_episode,
            viol_rate=viol_rate_per_episode,
            title_prefix=title_prefix,
            run_dir=run_dir
        )
    env.close()
    return agent, episode_rewards, best_weights, best_avg_reward


def train(agent='ppo',
          use_shield_post=False,
          use_shield_pre=False,
          use_shield_layer=False,
          mode='hard',
          monitor_constraints=True,
          num_episodes=100,
          verbose=False,
          visualize=False,
          use_ram_obs=False,
          agent_kwargs=None,
          env_name='MiniGrid-Empty-5x5-v0',
          render=False,
          seed=42):
    # rand_seed = np.random.randint(0, 2**32 - 1)
    # print(f'Rand_seed: {rand_seed}')
    env = create_environment(env_name, render=render, use_ram_obs=use_ram_obs, seed=seed)
    set_seed(seed)
    print("Observation space:", env.observation_space)
    obs_space = env.observation_space
    if agent_kwargs is None:
        agent_kwargs = {}

    env_config = config_by_env(env_name, use_ram_obs)
    if env_config['use_cnn'] and len(obs_space.shape) == 2:
        # Expand grayscale (H, W) → (C, H, W)
        input_shape = (1, *obs_space.shape)
    else:
        input_shape = obs_space.shape
    use_cnn = env_config['use_cnn']
    print(f"[DEBUG] use_cnn: {use_cnn}")
    print(f"[DEBUG] Gym observation shape: {obs_space.shape}")
    print(f"[DEBUG] Input shape passed to ModularNetwork: {input_shape}")

    torch.backends.cudnn.benchmark = True
    print(f"[DEBUG] Using benchmark")
    if isinstance(obs_space, gym.spaces.Box):
        state_dim = int(np.prod(obs_space.shape))
    elif isinstance(obs_space, gym.spaces.Dict) and 'image' in obs_space.spaces:
        state_dim = int(np.prod(obs_space.spaces['image'].shape))
    else:
        raise ValueError(f"Unsupported observation space type: {obs_space}")

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    else:
        action_dim = env.action_space.shape[0]

    requirements_path = 'src/requirements/red_light_stop.cnf'

    if agent == 'dqn':
        agent = DQNAgent(input_shape=input_shape,
                         action_dim=action_dim,
                         use_shield_post=use_shield_post,
                         use_shield_pre=use_shield_pre,
                         use_shield_layer=use_shield_layer,
                         agent_kwargs=agent_kwargs,
                         monitor_constraints=monitor_constraints,
                         mode=mode,
                         verbose=verbose,
                         requirements_path=requirements_path,
                         env=env,
                         use_cnn=use_cnn)
    elif agent == 'ppo':
        agent = PPOAgent(input_shape=input_shape,
                         action_dim=action_dim,
                         use_shield_post=use_shield_post,
                         use_shield_pre=use_shield_pre,
                         use_shield_layer=use_shield_layer,
                         use_orthogonal_init=True,
                         agent_kwargs=agent_kwargs,
                         monitor_constraints=monitor_constraints,
                         mode=mode,
                         verbose=verbose,
                         requirements_path=requirements_path,
                         env=env,
                         use_cnn=use_cnn)
    elif agent == 'a2c':
        agent = A2CAgent(input_shape=input_shape,
                         action_dim=action_dim,
                         use_shield_post=use_shield_post,
                         use_shield_pre=use_shield_pre,
                         use_shield_layer=use_shield_layer,
                         monitor_constraints=monitor_constraints,
                         mode=mode,
                         verbose=verbose,
                         requirements_path=requirements_path,
                         env=env,
                         use_cnn=use_cnn)
    else:
        raise ValueError("Unsupported agent type.")

    print(f"Training {agent.__class__.__name__} agent on {env_name} with shield post: {use_shield_post} "
          f"with shield layer: {use_shield_layer} with shield pre: {use_shield_pre}")

    agent_trained, episode_rewards, best_weights, best_avg_reward = run_training(
        agent, env,
        verbose=verbose,
        num_episodes=num_episodes,
        visualize=visualize,
        render=render,
        softness=mode,
        use_cnn=use_cnn
    )

    if hasattr(agent_trained, 'load_weights') and best_weights is not None:
        print(f"\n[Post-Training] Loading best weights with avg reward: {best_avg_reward:.2f}")
        agent_trained.load_weights(best_weights)

    return agent_trained, episode_rewards, best_weights, best_avg_reward, env


def evaluate_policy(agent, env, num_episodes=100, visualize=False, render=False, force_disable_shield=False, run_dir=None, softness=""):
    use_layer = getattr(agent, 'use_shield_layer', False)
    use_post = getattr(agent, 'use_shield_post', False)
    use_pre = getattr(agent, 'use_shield_pre', False)

    if not softness:
        softness = ""
    else:
        softness = softness.capitalize()

    # Optionally disable shielding entirely
    if force_disable_shield:
        print("[Evaluation] Forcing shield OFF for evaluation")
        use_layer = False
        use_post = False

    print(f"\n[Evaluation] Evaluating {agent.__class__.__name__} for {num_episodes} episodes "
          f"(Shield Layer: {use_layer}, Shield Post: {use_post})")

    # Set policy to eval mode
    if hasattr(agent, 'q_network') and hasattr(agent.q_network, 'eval'):
        agent.q_network.eval()
    elif hasattr(agent, 'policy') and hasattr(agent.policy, 'eval'):
        agent.policy.eval()
    elif hasattr(agent, 'model') and hasattr(agent.model, 'eval'):
        agent.model.eval()

    # Disable exploration
    original_epsilon = getattr(agent, 'epsilon', None)
    if original_epsilon is not None:
        agent.epsilon = 0.0

    total_rewards = []
    total_steps = 0
    violations_per_episode = []

    # Reset global counters
    if hasattr(agent, 'constraint_monitor') and agent.constraint_monitor:
        agent.constraint_monitor.reset_all()

    for episode in range(num_episodes):
        state = reset_env(env)
        done = False
        episode_reward = 0

        if hasattr(agent, 'constraint_monitor') and agent.constraint_monitor:
            agent.constraint_monitor.reset()

        while not done:
            try:
                selected_action, a_unshielded, a_shielded, context = agent.select_action(
                    state, env, do_apply_shield=not force_disable_shield
                )
            except TypeError:
                selected_action, a_unshielded, a_shielded, context = agent.select_action(state, env, do_apply_shield=not force_disable_shield)

            state, reward, done, _ = step_env(env, selected_action)
            episode_reward += reward
            total_steps += 1

            if render:
                env.render()

        total_rewards.append(episode_reward)

        # === Episode stats ===
        if hasattr(agent, 'constraint_monitor'):
            stats = agent.constraint_monitor.summary()
            episode_modifications = stats['episode_modifications']
            modification_rate = stats['total_mod_rate']
            episode_violations = stats['episode_violations']
            total_modifications = stats['total_modifications']
            total_violations = stats['total_violations']
            violation_rate = stats['total_viol_rate']
            violations_per_episode.append(episode_violations)

            # print(
            #     f"Episode {episode + 1}: "
            #     f"Reward = {episode_reward:.2f}, "
            #     f"Episode Modifications = {episode_modifications}, "
            #     f"Episode Violations = {episode_violations}, "
            #     f"Total Modifications = {total_modifications}, "
            #     f"Total Violations = {total_violations}, "
            #     f"Mod Rate = {modification_rate:.4f}, "
            #     f"Viol Rate = {violation_rate:.4f}"
            # )

    if original_epsilon is not None:
        agent.epsilon = original_epsilon

    # === Final Summary ===
    if hasattr(agent, 'constraint_monitor'):
        stats = agent.constraint_monitor.summary()
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        avg_mod_rate = stats['total_modifications'] / max(stats['total_steps'], 1)
        avg_viol_rate = stats['total_violations'] / max(stats['total_steps'], 1)

        print("\nEvaluation Summary:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Standard Deviation: {std_reward:.2f}")
        print(f"Avg Shield Modifications per Step: {avg_mod_rate:.4f}")
        print(f"Total Constraint Violations: {stats['total_violations']}")
        print(f"Avg Constraint Violations per Step: {avg_viol_rate:.4f}")
    else:
        avg_reward = np.mean(total_rewards)
        print("\nEvaluation Summary (No ConstraintMonitor):")
        print(f"Average Reward: {avg_reward:.2f}")

    if visualize:
        agent_name = agent.__class__.__name__
        env_name = getattr(env, 'spec', None).id if hasattr(env, 'spec') else str(env)
        if force_disable_shield:
            shield_mode = "Unshielded (forced)"
        elif use_layer:
            shield_mode = "Shield Layer"
        elif use_post:
            shield_mode = "Post Shield"
        elif use_pre:
            shield_mode = "Pre-emptive Shield"
        elif agent.lambda_sem > 0:
            shield_mode = "Semantic Loss"
        else:
            shield_mode = "Unshielded"

        title_prefix = f"{agent_name} on {env_name} {shield_mode} {softness}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        graphing.plot_rewards(
            rewards=total_rewards,
            title=f"{title_prefix} – Reward Curve",
            xlabel="Episode",
            ylabel="Reward",
            rolling_window=5,
            run_dir=run_dir,
            show=False,
        )

        # graphing.plot_violations(
        #     violations=violations_per_episode,
        #     total_steps=total_steps,
        #     title=f"{title_prefix} – Violations per Episode",
        #     save_path=f"plots/{agent_name}_{env_name}_{shield_mode}_{timestamp}_violations.png",
        #     show=True
        # )

    total_violations = stats["total_violations"]
    total_modifications = stats["total_modifications"]
    total_steps_eval = stats["total_steps"]
    env.close()

    return {
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "max_reward": np.max(total_rewards),
        "min_reward": np.min(total_rewards),
        "avg_shield_mod_rate": total_modifications / max(total_steps_eval, 1),
        "avg_violations_per_step": total_violations / max(total_steps_eval, 1),
        "avg_violations_per_episode": total_violations / max(num_episodes, 1),
        "avg_modifications_per_episode": total_modifications / max(num_episodes, 1),
        "total_violations": total_violations,
        "total_modifications": total_modifications,
        "total_steps": total_steps_eval,
    }


def run_multiple_evaluations(
        agent_name='dqn',
        env_name='MiniGrid-Empty-5x5-v0',
        use_ram_obs=False,
        num_runs=3,
        num_train_episodes=2000,
        num_eval_episodes=15,
        use_shield_post=False,
        use_shield_pre=False,
        use_shield_layer=False,
        monitor_constraints=True,
        mode='hard',
        verbose=False,
        visualize=False,
        render=False,
        force_disable_shield=False,
        run_dir=None
):
    all_results = []

    for run in range(num_runs):
        print(f"\n=== Run {run + 1}/{num_runs} ===")

        agent, episode_rewards, best_weights, best_avg_reward, env = train(
            agent=agent_name,
            env_name=env_name,
            use_ram_obs=use_ram_obs,
            num_episodes=num_train_episodes,
            use_shield_post=use_shield_post,
            use_shield_pre=use_shield_pre,
            use_shield_layer=use_shield_layer,
            monitor_constraints=monitor_constraints,
            mode=mode,
            verbose=verbose,
            visualize=visualize,
            render=render,
            seed=run+1 # seed is run number (1-indexed)
        )

        if hasattr(agent, 'load_weights') and best_weights is not None:
            agent.load_weights(best_weights)

        results = evaluate_policy(
            agent,
            env,
            num_episodes=num_eval_episodes,
            visualize=visualize,
            render=render,
            softness=mode,
            force_disable_shield=force_disable_shield,
            run_dir=run_dir
        )
        env.close()

        # === CSV-Formatted Output for Excel Logging ===
        if force_disable_shield:
            shield_mode = "Unshielded (forced)"
        elif use_shield_layer:
            shield_mode = "ShieldLayer"
        elif use_shield_post:
            shield_mode = "PostShield"
        elif use_shield_pre:
            shield_mode = "PreemptiveShield"
        elif hasattr(agent, 'lambda_sem') and agent.lambda_sem > 0:
            shield_mode = "SemanticLoss"
        else:
            shield_mode = "Unshielded"

        csv_line = "\t".join([
            env_name,
            agent_name.upper(),
            "red_light_stop.cnf", # TODO: make this not hardcoded
            shield_mode,
            f"{run + 1}",
            f"{results['avg_reward']:.2f}",
            f"{results['std_reward']:.2f}",
            f"{results['total_modifications']}",
            f"{results['avg_shield_mod_rate']:.4f}",
            f"{results['total_violations']}",
            f"{results['avg_violations_per_step']:.4f}"
        ])
        csv_path = os.path.join(run_dir, "evaluation_summary.csv")
        with open(csv_path, "a") as f:
            f.write(csv_line + "\n")

        all_results.append(results)

    # Aggregate final statistics
    def aggregate_results(all_results):
        keys = all_results[0].keys()
        aggregated = {}
        for key in keys:
            values = [r[key] for r in all_results]
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        return aggregated

    final_summary = aggregate_results(all_results)

    print("\n=== Final Aggregated Evaluation over", num_runs, "runs ===")
    for key, val in final_summary.items():
        print(f"{key}: {val['mean']:.4f} ± {val['std']:.4f}")

    return final_summary

def make_run_dir(base_dir="results", env_name=None, agent_name=None, softness="",
                 use_shield_post=False, use_shield_pre=False, use_shield_layer=False,
                 run_index=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shield_mode = (
        "ShieldLayer" if use_shield_layer else
        "PostShield" if use_shield_post else
        "PreemptiveShield" if use_shield_pre else
        "Unshielded"
    )
    parts = [env_name, agent_name, shield_mode, softness, timestamp]
    if run_index is not None:
        parts.append(f"run{run_index}")
    dir_name = "_".join(filter(None, parts))
    run_dir = os.path.join(base_dir, dir_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent (DQN, A2C, PPO) with optional shield and environment.")
    parser.add_argument('--agent', choices=['dqn', 'ppo', 'a2c'], default='ppo',
                        help='Which agent to use: dqn, ppo, a2c')
    parser.add_argument('--use_shield_post', action='store_true', help='Enable PiShield constraints during training')
    parser.add_argument('--use_shield_pre', action='store_true', help='Enable preemptive constraints during training')
    parser.add_argument('--use_shield_layer', action='store_true', help='Enable shield layer')
    parser.add_argument('--mode', choices=['soft', 'hard', 'progressive', ''], default='', help='Constraint mode')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--visualize', action='store_true', help='Visualize training plots')
    parser.add_argument('--render', action='store_true', help='Render environment (RGB image)')
    parser.add_argument('--env', type=str, default='MiniGrid-Empty-5x5-v0', help='Gym environment to train on')
    parser.add_argument('--force_disable_shield', action='store_true', help='Force shield off during evaluation')
    parser.add_argument('--num_episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--monitor_constraints', action='store_true', help='Enable constraint monitor')
    parser.add_argument('--no_eval', action='store_true', help='Do not run eval')
    parser.add_argument('--use_ram_obs', action='store_true', help='Use RAM for observation space')
    args = parser.parse_args()
    print(f'parser use shield post: {args.use_shield_post}')

    if not args.no_eval:
        if not args.mode:
            softness = ""

        run_dir = make_run_dir(
            env_name=args.env,
            agent_name=args.agent,
            use_shield_post=args.use_shield_post,
            use_shield_pre=args.use_shield_pre,
            use_shield_layer=args.use_shield_layer,
            softness=args.mode,
            run_index=1  # Optional if running multiple experiments
        )

        run_multiple_evaluations(
            agent_name=args.agent,
            env_name=args.env,
            use_ram_obs=args.use_ram_obs,
            num_runs=3,
            num_train_episodes=args.num_episodes,
            num_eval_episodes=100,
            use_shield_post=args.use_shield_post,
            use_shield_pre=args.use_shield_pre,
            use_shield_layer=args.use_shield_layer,
            monitor_constraints=args.monitor_constraints,
            mode=args.mode,
            verbose=args.verbose,
            visualize=args.visualize,
            render=args.render,
            force_disable_shield=args.force_disable_shield,
            run_dir=run_dir
        )