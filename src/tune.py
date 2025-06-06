import numpy as np
import optuna
from src.agents.dqn_agent import DQNAgent
from src.train import create_environment, run_training

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.90, 0.99)
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.95, 0.999)
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
    target_update_freq = trial.suggest_categorical('target_update_freq', [500, 1000, 2000])

    # Create environment
    env = create_environment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create agent
    agent = DQNAgent(
        state_dim,
        action_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq
    )

    # Train the agent
    rewards = run_training(agent, env, num_episodes=200, print_interval=None, log_rewards=True)
    avg_reward = np.mean(rewards[-20:])  # Last 20 episodes

    return avg_reward

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
