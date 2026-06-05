"""Controlled test: run layer_hard's WINNING config in both hard and soft mode,
identical everything else. Isolates 'soft mechanism is worse' from 'soft search
got unlucky'. If soft floors/avoids (viol_rate->0, low reward) with these good
hyperparameters while hard reaches ~-111, the gap is mechanism, not search luck.
"""
import warnings; warnings.filterwarnings("ignore")
import yaml
from src.train import train, evaluate_policy

with open("config/tuned/ppo_layer_hard_Acrobot-v1_params.yaml") as f:
    cfg = yaml.safe_load(f)
print("Using layer_hard winning config:", cfg)

results = {}
for mode in ["hard", "soft"]:
    print(f"\n{'='*60}\n  RUN mode={mode}\n{'='*60}")
    agent, rewards, best_w, best_avg, env = train(
        agent="ppo", env_name="Acrobot-v1",
        use_shield_layer=True, mode=mode,
        monitor_constraints=True, verbose=False,
        seed=42, agent_kwargs=dict(cfg),
        num_episodes=500, early_stop_patience=100, target_reward=-100.0,
    )
    if hasattr(agent, "load_weights") and best_w is not None:
        agent.load_weights(best_w)
    ev = evaluate_policy(agent, env, num_episodes=50, force_disable_shield=False, softness=mode)
    results[mode] = (best_avg, ev["avg_reward"], ev.get("avg_violations_per_step", float("nan")),
                     agent.shield_controller.dead_row_activations)
    env.close()

print("\n" + "=" * 60)
print("CONTROLLED RESULT (same winning config, both modes):")
for mode, (ba, er, vr, dr) in results.items():
    print(f"  {mode:5s}  train_best={ba:7.1f}  eval_reward={er:7.1f}  eval_viol_rate={vr:.4f}  dead_rows={dr}")
print("Interpretation: soft flooring/avoiding here => mechanism, not search luck.")
