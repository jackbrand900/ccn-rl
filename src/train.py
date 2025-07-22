import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit, TransformObservation, FrameStack, AtariPreprocessing
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
from src.utils.shield_controller import ShieldController
from PIL import Image
from gymnasium.spaces import Box
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
# import cv2

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
    "ALE/Seaquest-v5": (None, None)
}

# def to_grayscale(obs):
#     obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # shape (96, 96)
#     obs = cv2.resize(obs, (48, 48))
#     obs = obs.astype(np.float32) / 255.0  # Normalize to [0, 1]
#     return obs  # shape (48, 48)

# def resize_rgb(obs):
#     obs = cv2.resize(obs, (96, 96))  # or 64×64 if performance needed
#     obs = obs.astype(np.float32) / 255.0
#     return obs  # shape: (96, 96, 3)

def create_environment(env_name, render=False):
    if env_name in custom_envs:
        entry_point, kwargs = custom_envs[env_name]
        if entry_point:
            register_env_if_needed(env_name, entry_point, kwargs)

        # Handle CartPole rendering differently
        if env_name == "CartPole-v1":
            env = gym.make(env_name, render_mode="human" if render else None)
            return env

        if env_name == "CarRacing-v3":
            env = gym.make(env_name, render_mode="human" if render else None, continuous=False)
            env = TimeLimit(env, max_episode_steps=200)
            return env

        if env_name == "CarRacingWithTrafficLights-v0":
            env = gym.make(env_name, render_mode="human" if render else None, continuous=False)

            # Define the transformed space: grayscale (96, 96), uint8
            # transformed_obs_space = Box(low=0.0, high=1.0, shape=(96, 96, 3), dtype=np.float32)

            # env = TransformObservation(env, resize_rgb, observation_space=transformed_obs_space)
            env = FrameStack(env, 4)
            env = TimeLimit(env, max_episode_steps=100)  # ✅ Set timestep limit
            return env

        if env_name == "ALE/Freeway-v5":
            env = gym.make(env_name, render_mode="human" if render else None)

            # Optional preprocessing (resize, normalize, etc.)
            # def resize_obs(obs):
            #     obs = cv2.resize(obs, (96, 96))  # optional
            #     return obs.astype(np.float32) / 255.0

            # env = TransformObservation(env, resize_obs, observation_space=Box(low=0.0, high=1.0, shape=(96, 96, 3), dtype=np.float32))
            env = gym.make(env_name, render_mode="rgb_array" if render else None, frameskip=1)
            env = AtariPreprocessing(env, frame_skip=4, scale_obs=True, terminal_on_life_loss=True)
            env = FrameStack(env, 4)
            env = TimeLimit(env, max_episode_steps=1000)  # ✅ Set timestep limit
            return env

        if env_name == "ALE/Seaquest-v5":
            env = gym.make(env_name, render_mode="human" if render else None)

            # Optional: resize or normalize observation if needed
            # def resize_obs(obs):
            #     obs = cv2.resize(obs, (96, 96))
            #     return obs.astype(np.float32) / 255.0
            # env = TransformObservation(env, resize_obs, observation_space=Box(low=0.0, high=1.0, shape=(96, 96, 3), dtype=np.float32))

            # env = FrameStack(env, 4)  # stack 4 frames
            env = TimeLimit(env, max_episode_steps=100)
            return env

        # Handle MiniGrid environments
        env = gym.make(env_name, render_mode="human" if render else None)
        if "MiniGrid" in env_name:
            if render:
                env = RGBImgObsWrapper(env)
                env = FullyObsWrapper(env)
            else:
                env = FlatObsWrapper(env)
        return env


def preprocess_state(state, use_cnn=False):
    if isinstance(state, dict):
        state = state['image']
    if isinstance(state, np.ndarray):
        if use_cnn:
            if state.ndim == 3:  # (H, W, C)
                state = state.astype(np.float32)
            elif state.ndim == 4:  # (stack, H, W, C)
                state = state.astype(np.float32)
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


def run_training(agent, env, num_episodes=1000, print_interval=10, log_rewards=False, use_cnn=False, visualize=False, verbose=False,
                 render=False):
    episode_rewards = []
    best_avg_reward = float('-inf')
    best_weights = None
    actions_taken = []
    modifications = 0
    for episode in range(1, num_episodes + 1):
        use_cnn = getattr(agent, "use_cnn", False)

        state, _ = env.reset()
        state = preprocess_state(state, use_cnn=use_cnn)
        try:  # TODO: extract this to the context provider
            key_pos = find_key(env)
            env.key_pos = key_pos
        except AttributeError:
            key_pos = None  # Not a MiniGrid environment

        done = False
        total_reward = 0
        batch_size = 128
        while not done:
            action, context, modified = agent.select_action(state, env)
            if modified:
                modifications += 1

            # TODO: make this environment agnostic
            pos = context.get("position", None)
            x, y = pos if pos is not None else (None, None)
            # print(f"Action taken: {action}")
            actions_taken.append((x, y, action))
            next_state, reward, terminated, truncated, info = env.step(action)
            # if context['at_red_light'] and verbose:
            #     print(f"[Episode {episode}] 🚦 At red light!")
            if render:
                env.render()

            next_state = preprocess_state(next_state, use_cnn=use_cnn)

            # next_state = next_state.flatten()
            done = terminated or truncated

            # if verbose:
            #     print(f"Episode {episode}, State: ({x}, {y}), Action: {action}, Reward: {reward}, Done: {done}")

            # print(f"Episode {episode}, State: ({x}, {y}), Action: {action}, Reward: {reward}, Done: {done}")
            agent.store_transition(state, action, reward, next_state, context, done)
            agent.update(batch_size=batch_size)
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        if print_interval and episode % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            log_msg = f"Episode {episode}, Avg Reward (last {print_interval}): {avg_reward:.2f}, Num Modifications: {modifications}"
            if hasattr(agent, "epsilon"):
                log_msg += f", Epsilon: {agent.epsilon:.3f}"
            print(log_msg)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_weights = copy.deepcopy(agent.get_weights())
                print(f"[Checkpoint] New best avg reward: {best_avg_reward:.2f} at episode {episode}")

    env.close()
    if visualize:
        graphing.plot_losses(agent.training_logs)
        graphing.plot_prob_shift(agent.training_logs)
        # action_counts = graphing.get_action_counts_per_state(actions_taken)
        # graphing.plot_action_histograms(action_counts)
        graphing.plot_rewards(
            rewards=episode_rewards,
            title="Training Performance",
            rolling_window=20,
            save_path="plots/training_rewards.png",
            show=True
        )

    return agent, episode_rewards, best_weights, best_avg_reward


def train(agent='dqn',
          use_shield=False,
          mode='hard',
          num_episodes=500,
          verbose=False,
          visualize=False,
          env_name='MiniGrid-Empty-5x5-v0',
          render=False):
    env = create_environment(env_name, render=render)
    print("Observation space:", env.observation_space)
    obs_space = env.observation_space

    env_config = config_by_env(env_name)
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

    requirements_path = 'src/requirements/wheel_on_grass.cnf'

    if agent == 'dqn':
        agent = DQNAgent(input_shape=input_shape,
                         action_dim=action_dim,
                         use_shield=use_shield,
                         mode=mode,
                         verbose=verbose,
                         requirements_path=requirements_path,
                         env=env,
                         use_cnn=use_cnn)
    elif agent == 'ppo':
        agent = PPOAgent(input_shape=input_shape,
                         action_dim=action_dim,
                         use_shield=use_shield,
                         mode=mode,
                         verbose=verbose,
                         requirements_path=requirements_path,
                         env=env,
                         use_cnn=use_cnn)
    elif agent == 'a2c':
        agent = A2CAgent(input_shape=input_shape,
                         action_dim=action_dim,
                         use_shield=use_shield,
                         mode=mode,
                         verbose=verbose,
                         requirements_path=requirements_path,
                         env=env,
                         use_cnn=use_cnn)
    else:
        raise ValueError("Unsupported agent type.")

    print(f"Training {agent.__class__.__name__} agent on {env_name} with shield: {use_shield}, render: {render}")
    agent_trained, _, best_weights, best_avg_reward = run_training(
        agent, env,
        verbose=verbose,
        num_episodes=num_episodes,
        visualize=visualize,
        render=render,
        use_cnn=use_cnn
    )

    if hasattr(agent_trained, 'load_weights') and best_weights is not None:
        print(f"\n[Post-Training] Loading best weights with avg reward: {best_avg_reward:.2f}")
        agent_trained.load_weights(best_weights)

    agent.load_weights(best_weights)
    return agent, env


def evaluate_policy(agent, env, num_episodes=100, eval_with_shield=False, visualize=False, render=False):
    print(f"\n[Evaluation] Starting evaluation on agent: {agent.__class__.__name__} | "
          f"Episodes: {num_episodes} | Shield: {getattr(agent, 'use_shield', False)} | "
          f"Render: {render} | Visualize: {visualize}")

    # Switch agent model to evaluation mode if applicable
    if hasattr(agent, 'q_network') and hasattr(agent.q_network, 'eval'):
        agent.q_network.eval()
    elif hasattr(agent, 'policy') and hasattr(agent.policy, 'eval'):
        agent.policy.eval()
    elif hasattr(agent, 'model') and hasattr(agent.model, 'eval'):
        agent.model.eval()

    # Store original exploration parameter if it exists
    original_epsilon = getattr(agent, 'epsilon', None)
    if original_epsilon is not None:
        agent.epsilon = 0.0  # Force deterministic policy if epsilon-greedy

    violations_per_episode = []
    total_rewards = []
    total_shield_modifications = 0
    total_steps = 0
    total_violations = 0

    # # Dynamically create a shield controller if needed for violation checking
    # if not eval_with_shield and (not hasattr(agent, "shield_controller") or agent.shield_controller is None):
    #     agent.shield_controller = ShieldController(requirements_path, action_dim, mode)

    for episode in range(num_episodes):
        state = reset_env(env)
        done = False
        episode_reward = 0
        episode_modifications = 0
        episode_violations = 0

        while not done:
            # Attempt to get (action, context, was_modified), fall back if needed
            try:
                result = agent.select_action(state, env, do_apply_shield=eval_with_shield)
                if isinstance(result, tuple) and len(result) == 3:
                    action, context, was_modified = result
                else:
                    action = result
                    was_modified = False

                if not eval_with_shield and agent.shield_controller:
                    violation = agent.shield_controller.would_violate(action, context)
                    total_violations += violation
                    episode_violations += violation

            except TypeError:
                action = agent.select_action(state, env, do_apply_shield=eval_with_shield)
                was_modified = False

            next_state, reward, done, _ = step_env(env, action)

            episode_reward += reward
            total_steps += 1
            state = next_state

            if was_modified:
                episode_modifications += 1

            if render:
                env.render()

        total_rewards.append(episode_reward)
        total_shield_modifications += episode_modifications
        violations_per_episode.append(episode_violations)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Shield Activations = {episode_modifications}")

    # Restore original exploration setting
    if original_epsilon is not None:
        agent.epsilon = original_epsilon

    avg_reward = np.mean(total_rewards)
    avg_shield_rate = total_shield_modifications / total_steps if total_steps > 0 else 0
    avg_violation_rate = total_violations / total_steps if total_steps > 0 else 0

    print(f"\nEvaluation Summary:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Avg Shield Modifications per Step: {avg_shield_rate:.4f}")
    print(f"Total Constraint Violations: {total_violations}")
    print(f"Avg Constraint Violations per Step: {avg_violation_rate:.4f}")
    graphing.plot_rewards(
        rewards=total_rewards,
        title=f"Evaluation Rewards Using Shield: {eval_with_shield}",
        xlabel="Episode",
        ylabel="Reward",
        rolling_window=5,
        save_path="plots/evaluation_rewards.png",
        show=True
    )
    graphing.plot_violations(
        violations=violations_per_episode,
        total_steps=total_steps,
        title=f"Constraint Violations per Episode",
        save_path=f"plots/evaluation_violations_{'with' if eval_with_shield else 'without'}_shield.png",
        show=True
    )

    return {
        "avg_reward": avg_reward,
        "avg_shield_mod_rate": avg_shield_rate,
        "rewards": total_rewards,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent (DQN, A2C, PPO) with optional shield and environment.")
    parser.add_argument('--agent', choices=['dqn', 'ppo', 'a2c'], default='dqn',
                        help='Which agent to use: dqn, ppo, or a2c')
    parser.add_argument('--use_shield', action='store_true', help='Enable PiShield constraints during training')
    parser.add_argument('--mode', choices=['soft', 'hard'], default='hard', help='Constraint mode')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--visualize', action='store_true', help='Visualize training plots')
    parser.add_argument('--render', action='store_true', help='Render environment (RGB image)')
    parser.add_argument('--env', type=str, default='MiniGrid-Empty-5x5-v0', help='Gym environment to train on')
    parser.add_argument('--eval_with_shield', action='store_true', help='Enable shield during evaluation')
    parser.add_argument('--num_episodes', type=int, default=500, help='Number of training episodes')
    args = parser.parse_args()

    trained_agent, env = train(agent=args.agent,
                               use_shield=args.use_shield,
                               mode=args.mode,
                               num_episodes=args.num_episodes,
                               verbose=args.verbose,
                               visualize=args.visualize,
                               env_name=args.env,
                               render=args.render)

    # results = evaluate_policy(trained_agent, env, eval_with_shield=args.eval_with_shield, num_episodes=20,
    # visualize=args.visualize, render=args.render)
    results1 = evaluate_policy(trained_agent, env, eval_with_shield=False, num_episodes=100, visualize=args.visualize)
    results2 = evaluate_policy(trained_agent, env, eval_with_shield=True, num_episodes=100, visualize=args.visualize)
