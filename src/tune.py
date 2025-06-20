import optuna
import numpy as np
import yaml
from src.train import create_environment, run_training
from src.agents.dqn_agent import DQNAgent

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.90, 0.99)
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.95, 0.999)
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
    target_update_freq = trial.suggest_categorical('target_update_freq', [500, 1000, 2000])

    env = create_environment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim,
        action_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq,
        use_shield=False,
        verbose=False,
        requirements_path = 'src/requirements/forward_on_flag.cnf',
    )

    rewards = run_training(agent, env, num_episodes=200, print_interval=None, log_rewards=True)
    avg_reward = np.mean(rewards[-20:])

    return avg_reward

if __name__ == "__main__":
    agent_type = "dqn" # TODO: make this configurable
    storage = "sqlite:///optuna_study.db"
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{agent_type}_tuning",
        storage=storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")

    # Save the best hyperparameters to a YAML file for later use or analysis
    best_params = study.best_trial.params
    # with open(f"../config/{agent_type}_hyperparameters.yaml", "w") as file:
    #     yaml.dump(best_params, file)

    print(f"Hyperparameters saved to config/{agent_type}_hyperparameters.yaml")
