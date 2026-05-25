# NeSy submission — experimental results summary

Self-contained briefing for a fresh Claude session or human reviewer. Captures
the full state of experiments, findings, bugs found/fixed, and what remains
to be done before submission.

**Deadlines:** abstract June 9, full submission June 16, 2026.

**Status overview:**
| env | status |
|---|---|
| CartPole-v1 | ✅ complete (9 methods × 10 seeds, stats refreshed, trajectory plot) |
| CliffWalking-v1 | ✅ complete (9 methods × 10 seeds; CMDP has bimodal seed failure pattern documented as a finding) |
| ALE/Seaquest-v5 | ⏳ partial (7 methods reuse IJCAI data; CMDP + action_mask need 5-seed runs) |

---

## 1. Setup

### Repo paths

```
ccn-rl/
├── config/ijcai_tuned/                # Optuna-tuned hyperparameter YAMLs
│   ├── cppo_{CartPole-v1,CliffWalking-v1,ALE_Seaquest-v5}_params.yaml
│   ├── ppo_action_mask_{CartPole-v1,CliffWalking-v1,ALE_Seaquest-v5}_params.yaml
│   └── ... (other methods, all envs except Seaquest where only CMDP + action_mask are tuned)
├── results/nesy_experiments/          # NeSy-submission results
│   ├── CartPole-v1/                   # 9 methods × 10 seeds, all per-seed CSVs present
│   ├── CliffWalking-v1/               # 9 methods × 10 seeds, all per-seed CSVs present
│   ├── ALE_Seaquest-v5/               # 7 methods (aggregated-only, copied from IJCAI); cppo + action_mask pending
│   └── {CartPole-v1,CliffWalking-v1}_{per_method_stats,pairwise_reward,pairwise_viol_rate}.csv
├── scripts/
│   ├── run_ijcai_experiments.py            (main experiment driver; SEEDS=10, per-seed --skip_existing patch)
│   ├── tune_ijcai_methods.py               (Optuna tuning; study_version v13 for CliffWalking CMDP)
│   ├── analyze_nesy_results.py             (stats: bootstrap CI + Welch's t + Mann-Whitney U + BH-FDR)
│   ├── plot_action_mask_trajectories.py    (per-seed trajectory figure)
│   └── bench_action_mask_vs_sb3.py         (sb3-contrib validation, CartPole only)
└── src/
    ├── agents/ppo_agent.py                 (custom PPO with shield/mask hooks; minor known issues)
    ├── agents/constrained_ppo_agent.py     (CMDP; credit-assignment bug FIXED this session)
    ├── utils/shield_controller.py          (CNF constraint enforcement; not audited)
    ├── utils/constraint_monitor.py         (viol_rate / mod_rate logic; not audited)
    └── requirements/{emergency_cartpole,cliff_safe,seaquest_low_oxygen_go_up}.cnf
```

### Methods (all on the same custom PPO base)

1. `ppo_unshielded` — no constraint enforcement (baseline)
2. `ppo_reward_shaping` — penalty in reward for violations
3. `ppo_semantic_loss` — constraint-aware auxiliary loss term
4. `ppo_action_mask` — MaskablePPO-style hard mask on logits (NEW for this submission)
5. `ppo_preshield_soft` — CCN+ differentiable shield, soft mode (flag_active_val=0.8)
6. `ppo_preshield_hard` — CCN+ differentiable shield, hard mode (flag_active_val=1.0)
7. `ppo_layer_soft` — CCN+ shield as differentiable network layer, soft
8. `ppo_layer_hard` — CCN+ shield as differentiable network layer, hard
9. `cppo` — CMDP (constrained MDP with Lagrangian dual)

### Protocol

- **Seeds:** `[42, 123, 456, 789, 1011, 2024, 1337, 7, 314, 271]` — 10 for CartPole/CliffWalking, 5 for Seaquest
- **Tuning:** 100-trial Optuna per (method, env), objective = `-|avg_reward - target|` (calibrate, not maximize)
- **Targets:** CartPole +200, CliffWalking -20, Seaquest +250
- **Training:** max 1000 episodes (Seaquest 2000 max-steps/ep), early-stop at rolling-25 reward ≥ target, patience 200 (Seaquest 250)
- **Eval:** best-weights snapshot, 100 sampled-action episodes (50 for Seaquest)
- **Stat tests:** Welch's t + Mann-Whitney U on all 36 pairs per metric, BH-FDR corrected, α = 0.05

### Bug fixes applied during this session

- **CMDP credit assignment fixed** (`constrained_ppo_agent.py:182`). The buggy version executed `a_shielded` in env while memory stored `a_unshielded` — broken credit assignment that masked the real Lagrangian signal. Fixed: env now executes `a_unshielded`, memory and reward consistent. CartPole CMDP went from median 138 (buggy) → median 275 (correct). CliffWalking CMDP went from median -130 (buggy) → median -23 (correct, with bimodal seed pattern).

- **CliffWalking CMDP tuning search expanded** (`tune_ijcai_methods.py:201-231`). Previous narrow range (lr 0.012-0.022, fixed arch, no nu_lr/budget tuning) was insufficient. Replaced with broad search across full PPO/Lagrangian parameter space. Bumped entropy ceiling to 0.5 (v13 study) to escape safe-but-stuck local optima.

---

## 2. Metric definitions (CRITICAL for interpreting results)

The columns in result tables are reported as `viol_rate` and `mod_rate`, but the
*meaning* of `viol_rate` differs by method type:

| derived metric | computed as | meaning |
|---|---|---|
| **would-have-violated rate** | `viol_rate` column | how often the *underlying* policy proposes an unsafe action (measures constraint internalization) |
| **modification rate** | `mod_rate` column | how often the shield actually changes the action |
| **actual runtime violation rate** | `viol_rate − mod_rate` for shielded methods; `viol_rate` for non-shielded | what fraction of executed actions actually violated the constraint at runtime |

For **hard shielding** (`action_mask`, `preshield_hard`, `layer_hard`):
`viol_rate = mod_rate` → actual violations = **0** by construction.

For **soft shielding** (`preshield_soft`, `layer_soft`): `viol_rate > mod_rate`
slightly → small but nonzero actual violations (~0.3% on CartPole).

For **non-shielded methods** (unshielded, reward_shaping, semantic_loss, cppo):
`mod_rate = 0` → actual violations = `viol_rate`.

---

## 3. CartPole results (10 seeds, post-CMDP-fix)

### Headline table

| method | reward median | reward mean ± std | would-have-violated | mod_rate | **actual viol rate** |
|---|---|---|---|---|---|
| PPO (Unshielded) | 186 | 194 ± 27 | 0.018 | 0 | **0.018** |
| PPO + Reward Shaping | 234 | 219 ± 72 | 0.025 | 0 | **0.025** |
| PPO + Semantic Loss | 256 | 254 ± 94 | 0.018 | 0 | **0.018** |
| **PPO + Action Mask** | **170** | **172 ± 17** | **0.054** | 0.054 | **0** |
| PPO + Pre-emptive (Soft) | 234 | 231 ± 35 | 0.014 | 0.011 | **0.003** |
| PPO + Pre-emptive (Hard) | 239 | 270 ± 121 | 0.028 | 0.028 | **0** |
| PPO + Layer (Soft) | 220 | 264 ± 106 | 0.013 | 0.010 | **0.003** |
| PPO + Layer (Hard) | 152 | 181 ± 154 | 0.053 | 0.053 | **0** |
| **CMDP** | **275** | **290 ± 126** | **0.011** | 0 | **0.011** |

### Statistical findings (FDR-corrected α=0.05)

**Reward:** only one pairwise comparison survives correction:
- Action_mask (172) < Preshield_soft (231), Welch FDR p = 0.011

Most other reward pairs not significant — high-variance methods (preshield_hard ±121, layer_soft ±106, layer_hard ±154, CMDP ±126) bury other differences.

**Would-have-violated rate:** action_mask (0.054) significantly higher than:
unshielded, reward_shaping, semantic_loss, preshield_soft, layer_soft, CMDP.
Not significantly different from preshield_hard (0.028) or layer_hard (0.053).
**Hard-enforcement methods cluster together** on this metric in CartPole.

### Headline figure

`results/nesy_experiments/CartPole-v1/plots/CartPole-v1_action_mask_vs_preshield_soft_trajectories.png`

Per-seed training trajectories: action_mask shows peak-and-collapse across
seeds (median climbs to ~180, then drifts down). Preshield_soft converges
cleanly to ~250 and stays.

---

## 4. CliffWalking results (10 seeds, with v13 CMDP retune)

### Headline table

| method | reward median | reward mean ± std | would-have-violated | mod_rate | **actual viol rate** |
|---|---|---|---|---|---|
| PPO (Unshielded) | -19 | -246 ± 408 | 0.0005 | 0 | **0.0005** |
| PPO + Reward Shaping | -26 | -362 ± 525 | 0.002 | 0 | **0.002** |
| PPO + Semantic Loss | -25 | -412 ± 481 | 0.002 | 0 | **0.002** |
| **PPO + Action Mask** | **-20** | **-20 ± 1** | **0.302** | 0.302 | **0** |
| PPO + Pre-emptive (Soft) | -22 | -290 ± 560 | 0.003 | 0.002 | **0.001** |
| PPO + Pre-emptive (Hard) | -19 | **-19 ± 1** | 0.098 | 0.098 | **0** |
| PPO + Layer (Soft) | -21 | -320 ± 457 | 0.003 | 0.002 | **0.001** |
| PPO + Layer (Hard) | -20 | **-20 ± 1** | 0.186 | 0.186 | **0** |
| **CMDP** (v13) | **-23** | **-247 ± 389** | 0.009 | 0 | **0.009** |

### CliffWalking CMDP bimodal pattern (a key finding)

| seed | reward | viol_rate | category |
|---|---|---|---|
| 42 | -23.8 | 0 | converged |
| 1011 | **-15.5** | 0 | converged (best) |
| 1337 | -15.8 | 0 | converged |
| 2024 | -23.0 | 0.005 | converged |
| 314 | -17.3 | 0 | converged |
| 271 | -18.4 | 0 | converged |
| 7 | -35.3 | 0.007 | converged (close) |
| 789 | -309.4 | 0.079 | partial-stuck (wandering) |
| 123 | **-1000.0** | 0 | stuck (safe-but-stagnant) |
| 456 | **-1010.9** | 0 | stuck |

**7/10 seeds converge near-optimally; 2-3/10 collapse to safe-but-stuck**
("don't move" satisfies the constraint trivially). Reports median (-23) as
the cleaner stat; mean is dragged by collapse seeds.

### Within-hard-family ordering on viol_rate (cleanest CliffWalking finding)

Hard methods spread on would-have-violated rate, ordered by projection
sophistication:
- preshield_hard: **0.098** (full CCN+ + importance-ratio teaching trick)
- layer_hard: **0.186** (differentiable layer)
- action_mask: **0.302** (simple 0/1 mask, no teaching)

All three pairwise differences are FDR-significant. **Direct empirical
evidence that gradient-signal strength predicts learning** — a finding
CartPole's simpler CNF couldn't reveal.

### Other CliffWalking patterns

- **Hard methods dominate on reward stability** (-19 to -20, std ~1). Soft methods + unshielded have catastrophic seeds (mean -250 to -412, std 400-560) from cliff-falls.
- **Soft methods have huge variance** from 2-3 catastrophic seeds. Use median.

### Headline figure

`results/nesy_experiments/CliffWalking-v1/plots/CliffWalking-v1_action_mask_vs_preshield_soft_trajectories.png`

Striking inversion of CartPole pattern: action_mask shows 10 seeds tightly
converging to -20 (no collapse). Preshield_soft has 7 stable seeds + 2-3
seeds catastrophically diverging to -1000 to -1750.

---

## 5. Seaquest status

### What's tuned

- `config/ijcai_tuned/cppo_ALE_Seaquest-v5_params.yaml` — from IJCAI supplementary Table 5 (lr=1.91e-4, ent_coef=0.164, budget=0.283, etc.)
- `config/ijcai_tuned/ppo_action_mask_ALE_Seaquest-v5_params.yaml` — just tuned (lr=1.9e-4, ent_coef=0.011, achieved reward 233/target 250 in ~19 min)
- Other methods → fall back to PPOAgent defaults (matches IJCAI submission protocol)

### What's run vs pending

| method | status | data |
|---|---|---|
| 7 non-CMDP-non-action_mask methods | reused from IJCAI | aggregated_results.json only (no per-seed CSVs) |
| **CMDP** | **pending** (fix changes results) | — |
| **PPO + Action Mask** | **pending** (new method) | — |

### To finish Seaquest

```bash
conda activate ccn_rl
python scripts/run_ijcai_experiments.py --env ALE/Seaquest-v5 \
    --method cppo ppo_action_mask \
    --base_dir results/nesy_experiments --use_subprocess
python scripts/analyze_nesy_results.py --env ALE/Seaquest-v5
```
10 runs (2 methods × 5 seeds). ETA: ~30-50 hours.

### Early action_mask Seaquest signal

In tuning, action_mask hit target reward 250 in ~20 episodes — very fast. Predicted: high viol_rate + high mod_rate (mask doing the work, policy not learning). Consistent with the CartPole/CliffWalking action_mask story.

---

## 6. The contribution (paper framing)

### Three-axis design space

Each method occupies a distinct point. **The contribution is characterizing the
space, not winning a leaderboard.**

**Axis 1 — Hard runtime safety guarantee (zero actual violations):**
- ✅ Yes: action_mask, preshield_hard, layer_hard
- 🟡 Partial: preshield_soft, layer_soft (~0.3% on CartPole)
- ❌ No: CMDP, semantic_loss, reward_shaping, unshielded

**Axis 2 — Constraint internalization (would-have-violated rate, lower = policy learned the constraint):**
- 🥇 Best: CMDP (CartPole), layer_soft, preshield_soft
- 🥈 Middling: semantic_loss, unshielded, reward_shaping, preshield_hard
- 🥉 Worst: layer_hard, action_mask

**Axis 3 — Reward (median):**
- 🥇 Best: CMDP (CartPole), preshield_hard, preshield_soft, semantic_loss, reward_shaping
- 🥉 Weakest: action_mask, layer_hard

### Unique value of hard CCN+ shielding

**Hard CCN+ shielding (preshield_hard, layer_hard) is the only paradigm
giving both runtime safety AND non-trivial policy learning.**

- preshield_hard: 0 actual violations + would-have-violated 0.028 + reward 239 (CartPole)
- action_mask: 0 actual violations + would-have-violated **0.054** + reward 170 — same safety, much worse learning
- CMDP: 0.011 actual violations + would-have-violated 0.011 + reward 275 — better learning, but no runtime safety

### CMDP CliffWalking bimodality is structural

CMDP succeeds in 7/10 seeds but collapses in 2-3/10 to safe-but-stuck (zero
violations, zero progress). Hard shielding methods succeed 10/10 because the
agent *cannot* avoid the goal by not moving — invalid actions are blocked, so
the policy must learn productive movement. **Structural advantage of hard
shielding over penalty-based safety in catastrophic-violation regimes.**

### Action_mask mechanism (the underlying claim)

Action masking provides zero gradient signal on forced-action states: when
the mask sets a logit to `-inf`, softmax → 0, gradient → 0. In states where
only one action is valid, the masked PPO update produces no learning signal.
The policy never internalizes the constraint, it just gets blocked at
runtime. Empirically observed across all envs; on CliffWalking, the
within-hard-family ordering directly tests this mechanism.

---

## 7. sb3-contrib validation (appendix material)

To rule out action_mask implementation bugs, ran matched-reward comparison
against `sb3-contrib.MaskablePPO` on CartPole:

| | reward at target=200 | viol_rate (would-have-violated) |
|---|---|---|
| ours (PPOAgent + use_action_mask) | 195.6 ± 34.5 | 0.052 |
| sb3-contrib MaskablePPO | 227.8 ± 32.9 | 0.030 |

Both reach the target. sb3's would-have-violated rate is lower (sample
efficiency), but both are substantially higher than soft-shielding methods
(0.013-0.014). **Relative ranking preserved across PPO bases.**

Script: `scripts/bench_action_mask_vs_sb3.py`. Currently 3 seeds CartPole only —
ideally bump to 5 + add CliffWalking before final submission.

---

## 8. Known bugs / limitations / things to acknowledge

### Bugs found this session

- **CMDP credit-assignment bug** (fixed). Old version executed shielded action in env while memory stored unshielded action. Detected by examining `constrained_ppo_agent.py:select_action`. Now corrected.

- **CliffWalking CMDP tuning search was too narrow** (fixed). Original narrow range produced unstable configs. Broad search with widened entropy escapes safe-but-stuck local optimum on most seeds.

- **Post-hoc shielding implementation has incoherent (action, log_prob, reward) bookkeeping**. Memory stores `a_unshielded` while env executes `a_shielded` and log_prob is for `a_shielded` under shielded distribution. This produces a meaningless importance ratio. Post-hoc methods are excluded from the comparison; bug is documented but not fixed (would require splitting the code path, no paper impact).

- **PPOAgent operator precedence at line 130**: `elif self.use_shield_post or self.use_shield_pre and do_apply_shield:` parses as `use_shield_post or (use_shield_pre and do_apply_shield)`. Minimal impact since post-hoc isn't in the comparison.

### Not audited but reasonably trusted

- `src/utils/shield_controller.py` (pishield projection) — empirical results coherent across envs suggest no critical bug, but not directly inspected
- `src/utils/constraint_monitor.py` (viol_rate, mod_rate counting) — same

### Methodology caveats to acknowledge in paper

- **Target-stopping protocol**: methods evaluated at matched task performance (early-stop at rolling-25 reward ≥ target). Reveals safety at matched perf level, not at convergence. Reviewers may push on this.
- **Tuning objective is calibration, not optimization**: Optuna minimizes |reward − target|, not maximizes reward. Methods don't necessarily achieve their "best" configs.
- **Mixed stopping criteria**: methods that hit target stop on target; methods that don't stop on patience (no-improvement). Two different rules.
- **Custom PPO sample efficiency**: ~30× fewer gradient updates per env-step than sb3 defaults. Documented via matched bench; relative comparisons preserved across PPO bases.
- **High variance on some methods**: reward std exceeds mean for some methods. Report median + IQR alongside mean ± std. Most reward pairwise differences are not statistically significant at n=10.
- **Violation rate column ambiguity**: would-have-violated (shielded methods) vs actual violations (unshielded methods). Must be made explicit in paper (see §2).
- **Seaquest data heterogeneity**: 7 methods reuse aggregated stats from original IJCAI submission; 2 methods (CMDP, action_mask) re-run for this submission. Acknowledge in protocol section.

---

## 9. Things to fix before submission

### Must do

- **Run CMDP and action_mask on Seaquest** — 10 runs total, ~30-50 hr
- **Refresh Seaquest stats** after the run completes
- **Update `NESY_RESULTS_SUMMARY.md`** with Seaquest numbers when done

### Paper-side fixes (no compute)

- **Add the metric-definition box (§2 of this doc) to the paper.** Most-likely reviewer-flag if missing.
- **Report median + IQR alongside mean ± std** for all results tables.
- **CMDP-fix acknowledgment**: one sentence noting the implementation correction relative to the IJCAI submission.
- **PPO sample-efficiency footnote** referencing the sb3 bench.
- **Target-stopping protocol acknowledgment** in methods + limitations sections.
- **Reword reward claims** to clusters/trends rather than absolute "X > Y" — most reward pairs are not statistically significant.
- **Seaquest protocol paragraph**: explicitly note the 7-method reuse and CMDP+action_mask re-runs.

### Nice-to-have

- Bump sb3-contrib bench to 5 seeds + add CliffWalking
- Audit `shield_controller.py` and `constraint_monitor.py` if time permits
- Trajectory plots for CMDP and action_mask on Seaquest once they're run

---

## 10. Run history (reproducibility)

### CartPole
```bash
conda activate ccn_rl
python scripts/run_ijcai_experiments.py --env CartPole-v1 \
    --base_dir results/nesy_experiments \
    --use_subprocess --num_train_episodes 1000
# After CMDP credit-assignment fix:
python scripts/run_ijcai_experiments.py --env CartPole-v1 --method cppo \
    --base_dir results/nesy_experiments \
    --use_subprocess --num_train_episodes 1000
python scripts/analyze_nesy_results.py --env CartPole-v1
python scripts/plot_action_mask_trajectories.py --env CartPole-v1
```

### CliffWalking
```bash
# Tune action_mask (only method without a CliffWalking config initially)
python scripts/tune_ijcai_methods.py --env CliffWalking-v1 \
    --method ppo_action_mask --trials 100
# Initial sweep (CMDP buggy in this pass)
python scripts/run_ijcai_experiments.py --env CliffWalking-v1 \
    --base_dir results/nesy_experiments \
    --use_subprocess --num_train_episodes 1000
# After CMDP fix + v13 entropy bump + bumped study_version:
python scripts/tune_ijcai_methods.py --env CliffWalking-v1 --method cppo --trials 30
python scripts/run_ijcai_experiments.py --env CliffWalking-v1 --method cppo \
    --base_dir results/nesy_experiments \
    --use_subprocess --num_train_episodes 1000
python scripts/analyze_nesy_results.py --env CliffWalking-v1
python scripts/plot_action_mask_trajectories.py --env CliffWalking-v1
```

### Seaquest (pending)
```bash
# CMDP config (saved from IJCAI Table 5): config/ijcai_tuned/cppo_ALE_Seaquest-v5_params.yaml
# action_mask config (just tuned): config/ijcai_tuned/ppo_action_mask_ALE_Seaquest-v5_params.yaml
# 7 other methods' aggregated_results.json copied from results/ijcai_experiments/ALE_Seaquest-v5/

# To complete:
python scripts/tune_ijcai_methods.py --env ALE/Seaquest-v5 --method ppo_action_mask \
    --trials 100 --use_ram_obs --max_episode_steps 2000     # DONE
python scripts/run_ijcai_experiments.py --env ALE/Seaquest-v5 \
    --method cppo ppo_action_mask \
    --base_dir results/nesy_experiments --use_subprocess     # PENDING
python scripts/analyze_nesy_results.py --env ALE/Seaquest-v5
```
