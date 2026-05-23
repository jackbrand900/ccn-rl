# CartPole pipeline summary — NeSy submission

Self-contained briefing for a fresh Claude session. Repo is a safe-RL benchmark
comparing 9 methods on logical-constraint enforcement. This doc covers the
CartPole leg of the experiments (other envs: CliffWalking, Seaquest — separate work).

---

## Repo layout (relevant bits)

```
ccn-rl/
├── config/ijcai_tuned/           # Optuna-tuned hyperparameter YAMLs per (method, env)
├── results/nesy_experiments/     # Clean NeSy-submission results live here
│   └── CartPole-v1/
│       ├── <method>/run_<seed>/results.json        (per-seed eval metrics)
│       ├── <method>/run_<seed>/train_metrics_*.csv (per-episode training trace)
│       ├── <method>/aggregated_results.json        (10-seed stats)
│       ├── plots/                                  (PNG figures)
│       └── summary_table.{txt,csv}
│   ├── CartPole-v1_per_method_stats.csv            (mean/std/median/IQR/bootstrap CI)
│   ├── CartPole-v1_pairwise_reward.csv             (Welch's t + Mann-Whitney + FDR)
│   └── CartPole-v1_pairwise_viol_rate.csv
├── scripts/
│   ├── run_ijcai_experiments.py            (main experiment driver)
│   ├── tune_ijcai_methods.py               (Optuna tuning)
│   ├── analyze_nesy_results.py             (stat tests + bootstrap CI)
│   ├── plot_action_mask_trajectories.py    (per-seed trajectory figure)
│   └── bench_action_mask_vs_sb3.py         (sb3-contrib validation)
└── src/
    ├── agents/ppo_agent.py                 (custom PPO with shield/mask hooks)
    ├── utils/shield_controller.py          (CNF constraint enforcement)
    └── requirements/emergency_cartpole.cnf (the CartPole constraint)
```

---

## What's being compared

9 safe-RL methods, all on the **same custom PPO base** (controlled comparison):

1. `ppo_unshielded` — no constraint enforcement (baseline)
2. `ppo_reward_shaping` — penalty in reward for violations
3. `ppo_semantic_loss` — constraint-aware auxiliary loss term
4. `ppo_action_mask` — MaskablePPO-style hard mask on logits (NEW baseline)
5. `ppo_preshield_soft` — CCN+ differentiable shield, soft mode
6. `ppo_preshield_hard` — CCN+ differentiable shield, hard mode
7. `ppo_layer_soft` — CCN+ shield as differentiable network layer, soft
8. `ppo_layer_hard` — CCN+ shield as differentiable network layer, hard
9. `cppo` — CMDP (constrained MDP with Lagrangian dual)

## Constraint

`src/requirements/emergency_cartpole.cnf` (4 CNF clauses):
- `y_0`, `y_1` = actions (push left, push right)
- `y_2`, `y_3` = state flags (emergency: pole tilting left/right)
- When `y_2` active → must take `y_0`; when `y_3` active → must take `y_1`

In ~5% of CartPole states the constraint is active (forces one action). In the
other ~95% both actions are valid. This is the regime where the action-mask
gradient-signal weakness shows up most clearly.

---

## Protocol

- **Seeds (10):** `[42, 123, 456, 789, 1011, 2024, 1337, 7, 314, 271]`
- **Tuning:** 100-trial Optuna per (method, env), objective = `-|avg_reward - target|`.
  Target = 200 for CartPole. Calibrates all methods to comparable performance so
  safety metrics are the actual point of comparison.
- **Training:** max 1000 episodes, early-stop when rolling-25 reward ≥ 200,
  patience = 200 (longer than IJCAI's 500/100; helps sample-limited methods)
- **Eval:** best weights (snapshot of highest rolling-25 reward during training),
  100 evaluation episodes, sampled actions

## How it was run

```bash
conda activate ccn_rl
python scripts/run_ijcai_experiments.py --env CartPole-v1 \
    --base_dir results/nesy_experiments \
    --use_subprocess --num_train_episodes 1000
```
`--use_subprocess` spawns a fresh Python process per (method, seed) to reclaim
memory. 90 runs total (9 methods × 10 seeds), ~3-4 hr unattended.

After: ran statistical analysis and trajectory plot.

```bash
python scripts/analyze_nesy_results.py --env CartPole-v1
python scripts/plot_action_mask_trajectories.py --env CartPole-v1
```

---

## Results (10 seeds, fresh NeSy run)

| method | mean ± std (reward) | median | viol_rate | mod_rate |
|---|---|---|---|---|
| PPO (Unshielded) | 194 ± 27 | 186 | 0.018 | 0 |
| PPO + Reward Shaping | 219 ± 72 | 234 | 0.025 | 0 |
| PPO + Semantic Loss | 254 ± 94 | 256 | 0.018 | 0 |
| **PPO + Action Mask** | **172 ± 17** | **170** | **0.054** | 0.054 |
| PPO + Pre-emptive (Soft) | 231 ± 35 | 234 | **0.014** | 0.011 |
| PPO + Pre-emptive (Hard) | 270 ± 121 | 239 | 0.028 | 0.028 |
| PPO + Layer (Soft) | 264 ± 106 | 220 | **0.013** | 0.010 |
| PPO + Layer (Hard) | 181 ± 154 | 152 | 0.053 | 0.053 |
| CMDP | 162 ± 95 | 138 | 0.010 | 0 |

`viol_rate` = "would-have-violated by raw policy" for shielded methods (action_mask,
preshield_*, layer_*); "actual violations" for unshielded/reward_shaping/semantic_loss/cppo.
**This naming overlap is the most-likely reviewer flag and needs to be made
explicit in the paper.**

## Statistical findings (Welch's t + Mann-Whitney U, BH-FDR α=0.05)

**Viol_rate — robust significance:**
- Action_mask (0.054) is significantly higher than: unshielded (0.018), reward_shaping (0.025), semantic_loss (0.018), preshield_soft (0.014), layer_soft (0.013), CMDP (0.010) — all FDR p < 0.05
- Action_mask NOT significantly different from preshield_hard (0.028, p=0.10) or layer_hard (0.053, p=0.96)
- **Hard-enforcement methods (action_mask, layer_hard, preshield_hard) cluster together statistically on viol_rate**

**Reward — only one pairwise comparison survives FDR:**
- Action_mask (172) < Preshield_soft (231), Welch FDR p = 0.011 *
- All others not significant after correction. High-variance methods (preshield_hard
  ±121, layer_soft ±106, layer_hard ±154) bury other differences.

---

## The paper claim (mechanism)

**Action masking provides zero gradient signal on forced-action states.** When
the mask sets a logit to `-inf`, softmax → 0, gradient → 0. In states where only
one action is valid, the masked PPO update produces no learning signal: the
policy never internalizes the constraint, it just gets blocked at runtime.

**Soft shielding methods give the policy a teaching gradient.** The pishield
CCN+ projection is differentiable, so the policy gradient flows through the
shielded distribution. The policy learns to anticipate the shield's output,
which means in shielded states it's still being trained.

**Empirical signature:** action_mask shows peak-and-collapse training
trajectories — the policy can briefly luck into competent play in dual-valid
states but cannot consolidate it. Soft methods converge smoothly to higher
sustained performance. See:
`results/nesy_experiments/CartPole-v1/plots/CartPole-v1_action_mask_vs_preshield_soft_trajectories.png`

**Theory predicts the clustering observed empirically:**
- Hard methods (action_mask, layer_hard, preshield_hard) share the weak/no gradient teaching → high viol_rate cluster
- Soft methods (preshield_soft, layer_soft) have continuous gradient through shield → low viol_rate cluster

---

## sb3-contrib validation (appendix)

To rule out implementation bugs in our custom action_mask, ran matched-reward
comparison vs the reference `sb3-contrib.MaskablePPO`:

| | reward at target=200 | viol_rate |
|---|---|---|
| ours (PPOAgent + use_action_mask) | 195.6 ± 34.5 | 0.052 |
| sb3-contrib MaskablePPO | 227.8 ± 32.9 | 0.030 |

Both reach the target. sb3's viol_rate is lower than ours (sample efficiency
difference), but both are higher than soft-shielding methods (0.013-0.014).
**Relative ranking preserved across PPO bases.**

Script: `scripts/bench_action_mask_vs_sb3.py`. Currently 3 seeds CartPole only —
plan to bump to 5 seeds and add CliffWalking before submission.

---

## Caveats for paper-side framing

1. **Custom PPO has ~30× fewer gradient updates per env-step than sb3 defaults**
   (rollout 64 vs 2048, 1 vs 32 minibatches per rollout). Add a footnote
   explaining this; cite the matched-bench as evidence relative comparisons
   are preserved.
2. **Reward variance is high** (some methods >100 std at n=10). Most reward
   pairwise differences are not statistically significant. Phrase reward
   findings around clusters/trends, not absolute "X > Y" claims (except the
   one significant pair).
3. **"Violation rate" column ambiguity.** It mixes "would-have-violated by
   raw policy" (shielded methods) and "actual violations" (others). Either
   split into two columns or clearly label per-row.
4. **Report median + IQR alongside mean ± std.** Cleaner for the high-variance
   methods (layer_hard mean 181 vs median 152, cppo 162 vs 138, preshield_hard
   270 vs 239).

---

## Files produced for the paper

| file | use |
|---|---|
| `results/nesy_experiments/CartPole-v1_per_method_stats.csv` | per-method stats (mean/std/median/IQR/95% CI) |
| `results/nesy_experiments/CartPole-v1_pairwise_reward.csv` | 36 pairwise reward comparisons, FDR-corrected |
| `results/nesy_experiments/CartPole-v1_pairwise_viol_rate.csv` | 36 pairwise viol_rate comparisons, FDR-corrected |
| `results/nesy_experiments/CartPole-v1/plots/CartPole-v1_action_mask_vs_preshield_soft_trajectories.png` | headline figure for action_mask peak-and-collapse story |
| `results/nesy_experiments/CartPole-v1/plots/` (rest) | standard violation/modification/reward curves |
| `results/nesy_experiments/summary_table.txt` | human-readable summary |

---

## State of the wider submission (for context)

- CartPole: ✅ done (this doc)
- CliffWalking: in progress — action_mask tuning done, full 10-seed sweep next
- Seaquest: tuning source needs to be resolved before sweep
- Paper-side fixes pending: metric column ambiguity, PPO footnote, median reporting

Deadlines: NeSy abstract June 9 (~24 days), full submission June 16 (~31 days).
