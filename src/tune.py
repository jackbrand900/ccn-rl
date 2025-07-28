import optuna
import numpy as np
import yaml
import argparse

from src.train import train

def objective(trial, agent_type="ppo", env_name="ALE/Freeway-v5", use_shield_post=False, use_shield_layer=False):
    # Shared hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])

    # Agent-specific
    agent_kwargs = {
        "lr": lr,
        "gamma": gamma,
        "hidden_dim": hidden_dim,
    }

    if agent_type == "ppo":
        clip_eps = trial.suggest_float("clip_eps", 0.1, 0.3)
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
        epochs = trial.suggest_int("epochs", 3, 10)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        agent_kwargs.update({
            "clip_eps": clip_eps,
            "ent_coef": ent_coef,
            "epochs": epochs,
            "batch_size": batch_size,
        })

    elif agent_type == "dqn":
        epsilon_decay = trial.suggest_float("epsilon_decay", 0.95, 0.999)
        target_update_freq = trial.suggest_categorical("target_update_freq", [250, 500, 1000])
        agent_kwargs.update({
            "epsilon_decay": epsilon_decay,
            "target_update_freq": target_update_freq,
        })

    elif agent_type == "a2c":
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
        agent_kwargs["ent_coef"] = ent_coef

    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    # === Train ===
    agent, rewards, _, best_avg_reward = train(
        agent=agent_type,
        env_name=env_name,
        use_shield_post=use_shield_post,
        use_shield_layer=use_shield_layer,
        monitor_constraints=False,
        num_episodes=500,
        verbose=False,
        visualize=False,
    )

    return best_avg_reward

def run_optuna(agent_type, env_name, n_trials=30, use_shield=False):
    storage = f"sqlite:///optuna_{agent_type}_{env_name.replace('/', '_')}.db"
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{agent_type}_{env_name}_tuning",
        storage=storage,
        load_if_exists=True
    )

    study.optimize(lambda trial: objective(trial, agent_type=agent_type, env_name=env_name),
                   n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")

    # Save to YAML
    filename = f"config/{agent_type}_{env_name.replace('/', '_')}_params.yaml"
    with open(filename, "w") as f:
        yaml.dump(trial.params, f)
    print(f"[✓] Best hyperparameters saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["ppo", "dqn", "a2c", "sac"], default="ppo")
    parser.add_argument("--env", type=str, default="ALE/Freeway-v5")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    run_optuna(agent_type=args.agent, env_name=args.env, n_trials=args.trials)
