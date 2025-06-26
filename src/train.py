import gymnasium as gym
import numpy as np
import src.utils.graphing as graphing
import argparse
import torch
from gymnasium.envs.registration import register
from src.agents.dqn_agent import DQNAgent
from minigrid.wrappers import FullyObsWrapper, FlatObsWrapper

# Register the environment with Gymnasium
if 'MiniGrid-Empty-5x5-v0' not in gym.envs.registry.keys():
    register(
        id='MiniGrid-Empty-5x5-v0',
        entry_point='minigrid.envs:EmptyEnv',
        kwargs={'size': 5},
    )

def create_environment():
    env = gym.make("MiniGrid-Empty-5x5-v0")
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

def run_training(agent, env, num_episodes=100, print_interval=10, log_rewards=False, visualize=False, verbose=False):
    episode_rewards = []
    actions_taken = []
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, context, modified = agent.select_action(state, env)
            x, y = context.get("position", None)
            actions_taken.append((x, y, action))
            next_state, reward, terminated, truncated, _ = env.step(action)
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
            print(f"Episode {episode}, Avg Reward (last {print_interval}): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    # evaluation results
    print("\nBeginning evaluation...")
    results = evaluate_policy(agent, env, num_episodes=20)

    env.close()
    if visualize:
        graphing.plot_losses(agent.training_logs)
        graphing.plot_prob_shift(agent.training_logs)
        action_counts = graphing.get_action_counts_per_state(actions_taken)
        graphing.plot_action_histograms(action_counts)
        graphing.plot_rewards(
            rewards=episode_rewards,
            title="Training Performance",
            rolling_window=20,
            save_path="plots/training_rewards.png",
            show=True
        )
        # actions = [action for _, _, action in actions_taken]
        # graphing.plot_action_frequencies(actions,
        #                         action_labels=['Left', 'Right', 'Forward', 'Pickup', 'Drop', 'Toggle', 'Done'])
    if log_rewards:
        return episode_rewards
    else:
        return None

def evaluate_policy(agent, env, num_episodes=10, render=False):
    agent.q_network.eval()
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # deterministic actions

    total_rewards = []
    total_shield_modifications = 0
    total_steps = 0

    for episode in range(num_episodes):
        state = reset_env(env)
        done = False
        episode_reward = 0
        episode_modifications = 0

        while not done:
            action, context, was_modified = agent.select_action(state, env)
            if was_modified:
                episode_modifications += 1
            next_state, reward, done, _ = step_env(env, action)

            episode_reward += reward
            total_steps += 1
            state = next_state

            if render:
                env.render()

        total_rewards.append(episode_reward)
        total_shield_modifications += episode_modifications
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Shield Activations = {episode_modifications}")

    agent.epsilon = original_epsilon  # restore exploration setting

    avg_reward = np.mean(total_rewards)
    avg_shield_rate = total_shield_modifications / total_steps if total_steps > 0 else 0

    print(f"\nEvaluation Summary:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Avg Shield Modifications per Step: {avg_shield_rate:.4f}")

    return {
        "avg_reward": avg_reward,
        "avg_shield_mod_rate": avg_shield_rate,
        "rewards": total_rewards,
    }


def train(use_shield=False, verbose=False, visualize=False):
    env = create_environment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim,
                     action_dim,
                     use_shield=use_shield,
                     verbose=verbose,
                     requirements_path = 'src/requirements/forward_on_flag.cnf',
                     env=env)
    run_training(agent, env, verbose=verbose, visualize=visualize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent with optional shield and verbose.")
    parser.add_argument('--use_shield', action='store_true', help='Enable PiShield constraints during training')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--visualize', action='store_true', help='Visualize action frequencies and rewards during training')
    args = parser.parse_args()

    train(use_shield=args.use_shield, verbose=args.verbose, visualize=args.visualize)
