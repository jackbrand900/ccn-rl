import gymnasium as gym
import numpy as np
import argparse
from gymnasium.envs.registration import register
from src.agents.dqn_agent import DQNAgent
from minigrid.wrappers import FullyObsWrapper, FlatObsWrapper
from src.utils.graphing import plot_rewards, plot_action_frequencies
from src.utils.visualize_env import visualize_env

# Register the environment with Gymnasium
if 'MiniGrid-Empty-5x5-v0' not in gym.envs.registry.keys():
    register(
        id='MiniGrid-Empty-5x5-v0',
        entry_point='minigrid.envs:EmptyEnv',
        kwargs={'size': 5},
    )

def create_environment():
    env = gym.make("MiniGrid-Empty-5x5-v0")
    env = FullyObsWrapper(env)
    env = FlatObsWrapper(env)
    return env

def run_training(agent, env, num_episodes=500, print_interval=10, log_rewards=False, visualize=False):
    episode_rewards = []
    actions_taken = []
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state, env)
            actions_taken.append(action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward

            if visualize:
                visualize_env(env)  # visualize current env state

        episode_rewards.append(total_reward)

        if print_interval and episode % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            print(f"Episode {episode}, Avg Reward (last {print_interval}): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()
    plot_rewards(
        rewards=episode_rewards,
        title="Training Performance",
        rolling_window=20,
        save_path="plots/training_rewards.png",
        show=True
    )
    plot_action_frequencies(actions_taken,
                            action_labels=['Left', 'Right', 'Forward', 'Pickup', 'Drop', 'Toggle', 'Done'])

    if log_rewards:
        return episode_rewards
    else:
        return None

def train(use_shield=False, verbose=False):
    env = create_environment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim,
                     action_dim,
                     use_shield=use_shield,
                     verbose=verbose,
                     requirements_path = 'src/requirements/left_only.linear',)
    run_training(agent, env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent with optional shield and verbose.")
    parser.add_argument('--use_shield', action='store_true', help='Enable PiShield constraints during training')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    train(use_shield=args.use_shield, verbose=args.verbose)
