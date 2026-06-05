"""Real Acrobot CCN+ hard-mode training to see whether the dead-row guard
actually fires under normal PPO dynamics (vs. only at synthetic extreme logits)."""
import warnings
warnings.filterwarnings("ignore")
from src.train import train

agent, rewards, _, best, env = train(
    agent="ppo",
    env_name="Acrobot-v1",
    use_shield_layer=True,
    mode="hard",
    monitor_constraints=True,
    num_episodes=300,
    verbose=False,
)

sc = agent.shield_controller
mon = getattr(agent, "constraint_monitor", None)
print("\n===== ACROBOT CCN+ HARD RESULT =====")
print(f"episodes trained      : {len(rewards)}")
print(f"final mean reward (last 50): {sum(rewards[-50:]) / max(len(rewards[-50:]),1):.1f}")
print(f"shield_activations    : {sc.shield_activations}")
print(f"dead_row_activations  : {sc.dead_row_activations}")
if mon is not None:
    print(f"monitor total_steps   : {mon.total_steps}")
    print(f"monitor flagged_steps : {mon.total_flagged_steps}")
    print(f"monitor violations    : {mon.total_violations}")
