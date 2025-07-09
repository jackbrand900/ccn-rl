import gymnasium as gym
import numpy as np
import src.utils.graphing as graphing
import argparse
from gymnasium.envs.registration import register
from minigrid.wrappers import FlatObsWrapper, FullyObsWrapper, RGBImgObsWrapper
from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.a2c_agent import A2CAgent
from src.utils import env_helpers
from src.utils.env_helpers import find_key


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
}


def create_environment(env_name, render=False):
    if env_name in custom_envs:
        entry_point, kwargs = custom_envs[env_name]
        if entry_point:
            register_env_if_needed(env_name, entry_point, kwargs)

        # Handle CartPole rendering differently
        if env_name == "CartPole-v1":
            env = gym.make(env_name, render_mode="human" if render else None)
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


def run_training(agent, env, num_episodes=500, print_interval=10, log_rewards=False, visualize=False, verbose=False,
                 render=False):
    episode_rewards = []
    actions_taken = []
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        try:  # TODO: extract this to the context provider
            key_pos = find_key(env)
            env.key_pos = key_pos
        except AttributeError:
            key_pos = None  # Not a MiniGrid environment

        if isinstance(state, dict):
            state = state['image'].flatten()
        else:
            state = state.flatten()
        done = False
        total_reward = 0

        while not done:
            action, context, modified = agent.select_action(state, env)
            # TODO: make this environment agnostic
            pos = context.get("position", None)
            x, y = pos if pos is not None else (None, None)
            actions_taken.append((x, y, action))
            next_state, reward, terminated, truncated, _ = env.step(action)
            if render:
                env.render()

            if isinstance(next_state, dict):
                next_state = next_state['image'].flatten()
            else:
                next_state = next_state.flatten()

            next_state = next_state.flatten()
            done = terminated or truncated

            if verbose:
                print(f"Episode {episode}, State: ({x}, {y}), Action: {action}, Reward: {reward}, Done: {done}")

            agent.store_transition(state, action, reward, next_state, context, done)
            agent.update()
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        if print_interval and episode % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            log_msg = f"Episode {episode}, Avg Reward (last {print_interval}): {avg_reward:.2f}"
            if hasattr(agent, "epsilon"):
                log_msg += f", Epsilon: {agent.epsilon:.3f}"
            print(log_msg)

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

    return episode_rewards if log_rewards else None


def train(agent='dqn', use_shield=False, verbose=False, visualize=False, env_name='MiniGrid-Empty-5x5-v0',
          render=False):
    env = create_environment(env_name, render=render)
    print("Observation space:", env.observation_space)
    obs_space = env.observation_space

    if isinstance(obs_space, gym.spaces.Box):
        state_dim = int(np.prod(obs_space.shape))
    elif isinstance(obs_space, gym.spaces.Dict) and 'image' in obs_space.spaces:
        state_dim = int(np.prod(obs_space.spaces['image'].shape))
    else:
        raise ValueError(f"Unsupported observation space type: {obs_space}")

    action_dim = env.action_space.n
    requirements_path = 'src/requirements/pickup_on_key.cnf'

    if agent == 'dqn':
        agent = DQNAgent(state_dim, action_dim,
                         use_shield=use_shield,
                         verbose=verbose,
                         requirements_path=requirements_path,
                         env=env)
    elif agent == 'ppo':
        agent = PPOAgent(state_dim, action_dim,
                         use_shield=use_shield,
                         verbose=verbose,
                         requirements_path=requirements_path,
                         env=env)
    elif agent == 'a2c':
        agent = A2CAgent(state_dim, action_dim,
                         use_shield=use_shield,
                         verbose=verbose,
                         requirements_path=requirements_path,
                         env=env)
    else:
        raise ValueError("Unsupported agent type.")

    print(f"Training {agent.__class__.__name__} agent on {env_name} with shield: {use_shield}, render: {render}")
    run_training(agent, env, verbose=verbose, visualize=visualize, render=render)
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

    total_rewards = []
    total_shield_modifications = 0
    total_steps = 0

    for episode in range(num_episodes):
        state = reset_env(env)
        done = False
        episode_reward = 0
        episode_modifications = 0

        while not done:
            # Attempt to get (action, context, was_modified), fall back if needed
            try:
                result = agent.select_action(state, env, do_apply_shield=eval_with_shield)
                if isinstance(result, tuple) and len(result) == 3:
                    action, context, was_modified = result
                else:
                    action = result
                    was_modified = False
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
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Shield Activations = {episode_modifications}")

    # Restore original exploration setting
    if original_epsilon is not None:
        agent.epsilon = original_epsilon

    avg_reward = np.mean(total_rewards)
    avg_shield_rate = total_shield_modifications / total_steps if total_steps > 0 else 0

    print(f"\nEvaluation Summary:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Avg Shield Modifications per Step: {avg_shield_rate:.4f}")

    graphing.plot_rewards(
        rewards=total_rewards,
        title=f"Evaluation Rewards Using Shield: {eval_with_shield}",
        xlabel="Episode",
        ylabel="Reward",
        rolling_window=5,
        save_path="plots/evaluation_rewards.png",
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
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--visualize', action='store_true', help='Visualize training plots')
    parser.add_argument('--render', action='store_true', help='Render environment (RGB image)')
    parser.add_argument('--env', type=str, default='MiniGrid-Empty-5x5-v0', help='Gym environment to train on')
    parser.add_argument('--eval_with_shield', action='store_true', help='Enable shield during evaluation')
    args = parser.parse_args()

    trained_agent, env = train(agent=args.agent,
                               use_shield=args.use_shield,
                               verbose=args.verbose,
                               visualize=args.visualize,
                               env_name=args.env,
                               render=args.render)

    # results = evaluate_policy(trained_agent, env, eval_with_shield=args.eval_with_shield, num_episodes=20,
    # visualize=args.visualize, render=args.render)
    results1 = evaluate_policy(trained_agent, env, eval_with_shield=False, num_episodes=100, visualize=args.visualize)
    results2 = evaluate_policy(trained_agent, env, eval_with_shield=True, num_episodes=100, visualize=args.visualize)
