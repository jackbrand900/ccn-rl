import optuna
import numpy as np
import yaml
import argparse

from src.train import train

def objective(trial, agent_type="ppo", env_name="ALE/Freeway-v5", use_shield_post=False, use_shield_layer=False, use_ram_obs=False):
    # === Shared hyperparameters ===
    lr = trial.suggest_float("lr", 1e-4, 4e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    use_orthogonal_init = trial.suggest_categorical("use_orthogonal_init", [True, False])  # <--- new param
    num_layers = trial.suggest_int("num_layers", 2, 4)

    agent_kwargs = {
        "lr": lr,
        "gamma": gamma,
        "hidden_dim": hidden_dim,
        "use_orthogonal_init": use_orthogonal_init,
        "num_layers": num_layers
    }

    # === Agent-specific parameters ===
    if agent_type == "ppo":
        clip_eps = trial.suggest_float("clip_eps", 0.1, 0.3)
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
        epochs = trial.suggest_int("epochs", 1, 10)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        agent_kwargs.update({
            "clip_eps": clip_eps,
            "ent_coef": ent_coef,
            "epochs": epochs,
            "batch_size": batch_size,
        })

    elif agent_type == "cppo":
        cost_gamma = trial.suggest_float("cost_gamma", 0.90, 0.999)
        cost_lam = trial.suggest_float("cost_lam", 0.90, 0.999)
        clip_eps = trial.suggest_float("clip_eps", 0.1, 0.3)
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
        epochs = trial.suggest_int("epochs", 1, 10)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        agent_kwargs.update({
            "cost_gamma": cost_gamma,
            "cost_lam": cost_lam,
            "clip_eps": clip_eps,
            "ent_coef": ent_coef,
            "epochs": epochs,
            "batch_size": batch_size,
        })

    elif agent_type == "dqn":
        epsilon_decay = trial.suggest_float("epsilon_decay", 50_000, 500_000)
        epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.1)
        target_update_freq = trial.suggest_categorical("target_update_freq", [250, 500, 1000])
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        buffer_size = trial.suggest_int("buffer_size", 100_000, 300_000)
        agent_kwargs.update({
            "epsilon_decay": epsilon_decay,
            "epsilon_end": epsilon_end,
            "target_update_freq": target_update_freq,
            "buffer_size": buffer_size,
            "batch_size": batch_size
        })

    elif agent_type == "a2c":
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
        agent_kwargs["ent_coef"] = ent_coef

    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    # === Train ===
    agent, rewards, _, best_avg_reward, env = train(
        agent=agent_type,
        env_name=env_name,
        use_shield_post=use_shield_post,
        use_shield_layer=use_shield_layer,
        monitor_constraints=False,
        num_episodes=1000,
        verbose=False,
        visualize=False,
        use_ram_obs=use_ram_obs,
        agent_kwargs=agent_kwargs,
    )

    return best_avg_reward


def run_optuna(agent_type, env_name, n_trials=50, use_shield_post=False, use_shield_layer=False, use_ram_obs=False):
    storage = f"sqlite:///optuna_{agent_type}_{env_name.replace('/', '_')}_1.db"
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{agent_type}_{env_name}_tuning",
        storage=storage,
        load_if_exists=True
    )

    study.optimize(lambda trial: objective(
        trial,
        agent_type=agent_type,
        env_name=env_name,
        use_shield_post=use_shield_post,
        use_shield_layer=use_shield_layer,
        use_ram_obs=use_ram_obs
    ), n_trials=n_trials)

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
    parser.add_argument("--agent", choices=["ppo", "dqn", "a2c", "cppo"], default="ppo")
    parser.add_argument("--env", type=str, default="ALE/Freeway-v5")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--use_ram_obs", action="store_true", help="Use RAM observations instead of pixels")
    parser.add_argument("--use_shield_post", action="store_true", help="Enable post-hoc PiShield during training")
    parser.add_argument("--use_shield_layer", action="store_true", help="Enable ShieldLayer training integration")
    args = parser.parse_args()

    run_optuna(
        agent_type=args.agent,
        env_name=args.env,
        n_trials=args.trials,
        use_shield_post=args.use_shield_post,
        use_shield_layer=args.use_shield_layer,
        use_ram_obs=args.use_ram_obs,
    )
