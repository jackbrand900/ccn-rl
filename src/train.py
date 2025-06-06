import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from src.agents.dqn_agent import DQNAgent
from minigrid.envs import EmptyEnv
from minigrid.wrappers import FullyObsWrapper, FlatObsWrapper

# Register the environment with Gymnasium (skip if already registered)
if 'MiniGrid-Empty-5x5-v0' not in gym.envs.registry.keys():
    register(
        id='MiniGrid-Empty-5x5-v0',
        entry_point='minigrid.envs:EmptyEnv',
        kwargs={'size': 5},
    )

def train():
    # Create and wrap the environment
    env = gym.make("MiniGrid-Empty-5x5-v0")
    env = FullyObsWrapper(env)   # Provides full observability
    env = FlatObsWrapper(env)    # Flattens observation to 1D vector

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    num_episodes = 1000
    print_interval = 10  # Print average reward every 10 episodes
    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        if episode % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            print(f"Episode {episode}, Avg Reward (last {print_interval}): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()

if __name__ == "__main__":
    train()
