"""Multi-seed validation of the tuned Acrobot configs for the core methods.
Answers: do the single-seed tuning rewards (vanilla -108.6, layer_hard -111.2,
layer_soft ~-220) hold up, or are they fragile single-seed escapes?

Reports mean +/- std of eval reward AND violation rate across seeds per method.
"""
import warnings; warnings.filterwarnings("ignore")
import argparse
import statistics as st
import yaml
from src.train import train, evaluate_policy

_p = argparse.ArgumentParser()
_p.add_argument("--episodes", type=int, default=1000)
_p.add_argument("--patience", type=int, default=200)
_p.add_argument("--seeds", type=int, nargs="+",
                default=[42, 123, 456, 789, 1011, 2024, 1337, 7, 314, 271])
_p.add_argument("--methods", type=str, nargs="+", default=None,
                help="subset of method names to run (default: all)")
_args = _p.parse_args()
SEEDS = _args.seeds

# method -> train() shield kwargs
METHODS = {
    "ppo_unshielded":  dict(use_shield_layer=False, use_action_mask=False, mode=""),
    "ppo_action_mask": dict(use_shield_layer=False, use_action_mask=True,  mode="hard"),
    "ppo_layer_hard":  dict(use_shield_layer=True,  use_action_mask=False, mode="hard"),
    "ppo_layer_soft":  dict(use_shield_layer=True,  use_action_mask=False, mode="soft"),
}

def load_cfg(method):
    with open(f"config/ijcai_tuned/{method}_Acrobot-v1_params.yaml") as f:
        return yaml.safe_load(f)

if _args.methods:
    METHODS = {k: v for k, v in METHODS.items() if k in _args.methods}

summary = {}
for method, shield in METHODS.items():
    cfg = load_cfg(method)
    rewards, viols = [], []
    for seed in SEEDS:
        agent, _, best_w, _, env = train(
            agent="ppo", env_name="Acrobot-v1",
            monitor_constraints=True, verbose=False,
            seed=seed, agent_kwargs=dict(cfg),
            num_episodes=_args.episodes, early_stop_patience=_args.patience,
            target_reward=-100.0,
            **shield,
        )
        if hasattr(agent, "load_weights") and best_w is not None:
            agent.load_weights(best_w)
        ev = evaluate_policy(agent, env, num_episodes=50, force_disable_shield=False,
                             softness=shield["mode"])
        rewards.append(ev["avg_reward"])
        viols.append(ev.get("avg_violations_per_step", float("nan")))
        env.close()
        print(f"[{method} seed={seed}] reward={ev['avg_reward']:.1f} viol={viols[-1]:.4f}")
    summary[method] = (rewards, viols)

print("\n" + "=" * 78)
print(f"MULTI-SEED ACROBOT ({len(SEEDS)} seeds, {_args.episodes} ep, patience {_args.patience})")
print(f"{'method':18} {'mean+/-std':>16} {'median':>8} {'solved>=-130':>12} {'viol_rate':>10}")
for method, (rs, vs) in summary.items():
    rm, rsd = st.mean(rs), (st.stdev(rs) if len(rs) > 1 else 0.0)
    med = st.median(rs)
    solved = sum(1 for x in rs if x >= -130)
    vm = st.mean(vs)
    print(f"{method:18} {rm:7.1f}+/-{rsd:5.1f} {med:8.1f} {solved:>8}/{len(rs):<3} {vm:>10.4f}")
    print(f"{'  rewards:':18} {sorted(round(x,1) for x in rs)}")
