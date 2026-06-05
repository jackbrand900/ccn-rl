"""Reproduce optuna trial 4 (unshielded PPO, reached -108.6) on the wrapper-free
Acrobot env to confirm it still learns to ~-100. Same regime as tuning:
500 train episodes, seed 42, target -100 early-stop, 50 eval episodes."""
import warnings; warnings.filterwarnings("ignore")
from src.train import train, evaluate_policy

# trial 4 params (optuna_ijcai_ppo_unshielded_Acrobot-v1_v18, value -0.086, reward -108.6)
agent_kwargs = {
    "lr": 0.0001331707625849592,
    "gamma": 0.9555122888613474,
    "hidden_dim": 256,
    "use_orthogonal_init": True,
    "num_layers": 2,
    "clip_eps": 0.10786489226725048,
    "ent_coef": 0.04431278592736043,
    "epochs": 7,
    "batch_size": 64,
}

agent, episode_rewards, best_weights, best_avg_reward, env = train(
    agent="ppo",
    env_name="Acrobot-v1",
    num_episodes=500,
    use_shield_layer=False,
    mode="",
    monitor_constraints=True,
    seed=42,
    agent_kwargs=agent_kwargs,
    early_stop_patience=100,
    target_reward=-100.0,
)
if hasattr(agent, "load_weights") and best_weights is not None:
    agent.load_weights(best_weights)

results = evaluate_policy(agent, env, num_episodes=50, force_disable_shield=False, softness="")
print("\n===== ACROBOT TRIAL-4 REPRO (no wrapper) =====")
print(f"episodes trained        : {len(episode_rewards)}")
print(f"best_avg_reward (train) : {best_avg_reward:.1f}")
print(f"eval avg_reward (50 ep) : {results['avg_reward']:.1f}   (trial-4 was -108.6)")
