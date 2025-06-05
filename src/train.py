import gym
from src.agents.dqn_agent import DQNAgent

def train():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    num_episodes = 500
    for episode in range(num_episodes):
        state, _ = env.reset()  # Updated to unpack state and info
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Combine both for proper 'done' logic
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward

        print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    env.close()

if __name__ == "__main__":
    train()
