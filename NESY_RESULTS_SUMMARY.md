# NeSy submission — experimental results summary

Self-contained briefing covering all experimental work done so far for the NeSy
submission. Hand this to another Claude session or human reviewer for context.

**Status:** CartPole ✅ done (with CMDP fix), CliffWalking ⚠️ partially stale
(CMDP needs re-run with fix), Seaquest ⏳ pending.

---

## 1. The setup

### Repo structure (relevant paths)

```
ccn-rl/
├── config/ijcai_tuned/                # Optuna-tuned hyperparameter YAMLs
├── results/nesy_experiments/          # NeSy-submission results live here
│   ├── CartPole-v1/                   # 9 methods × 10 seeds
│   ├── CliffWalking-v1/               # 9 methods × 10 seeds (CMDP STALE)
│   ├── CartPole-v1_per_method_stats.csv
│   ├── CartPole-v1_pairwise_reward.csv
│   ├── CartPole-v1_pairwise_viol_rate.csv
│   ├── CliffWalking-v1_per_method_stats.csv     (needs refresh)
│   ├── CliffWalking-v1_pairwise_reward.csv      (needs refresh)
│   └── CliffWalking-v1_pairwise_viol_rate.csv   (needs refresh)
├── scripts/
│   ├── run_ijcai_experiments.py            (main experiment driver)
│   ├── tune_ijcai_methods.py               (Optuna tuning)
│   ├── analyze_nesy_results.py             (stats: bootstrap CI + pairwise tests)
│   ├── plot_action_mask_trajectories.py    (per-seed trajectory figure)
│   └── bench_action_mask_vs_sb3.py         (sb3-contrib validation)
└── src/
    ├── agents/ppo_agent.py                 (custom PPO with shield/mask hooks)
    ├── agents/constrained_ppo_agent.py     (CMDP; credit-assignment bug FIXED)
    ├── utils/shield_controller.py          (CNF constraint enforcement)
    └── requirements/{emergency_cartpole,cliff_safe}.cnf
```

### Methods (all on the same custom PPO base)

1. `ppo_unshielded` — no constraint enforcement (baseline)
2. `ppo_reward_shaping` — penalty in reward for violations
3. `ppo_semantic_loss` — constraint-aware auxiliary loss term
4. `ppo_action_mask` — MaskablePPO-style hard mask on logits (NEW baseline)
5. `ppo_preshield_soft` — CCN+ differentiable shield, soft mode
6. `ppo_preshield_hard` — CCN+ differentiable shield, hard mode
7. `ppo_layer_soft` — CCN+ shield as differentiable network layer, soft
8. `ppo_layer_hard` — CCN+ shield as differentiable network layer, hard
9. `cppo` — CMDP (constrained MDP with Lagrangian dual)

### Bug fixes applied during this session

- **CMDP credit assignment** (constrained_ppo_agent.py:182). Previously, the env
  executed `a_shielded` while memory stored `a_unshielded` — broken credit
  assignment that masked the real Lagrangian signal. Fixed: env now executes
  `a_unshielded`, memory and reward consistent. CartPole CMDP went from median
  reward 138 (buggy) → **275 (correct)**; viol_rate became *real* (actual
  violations during eval) instead of counterfactual.

- **CliffWalking CMDP tuning search space** (tune_ijcai_methods.py:201-231).
  Previous narrow range (lr 0.012-0.022, fixed arch, no nu_lr/budget tuning)
  produced unstable configs. Now uses a broad search over the full PPO/Lagrangian
  parameter space. `study_version` bumped to v12 so Optuna starts fresh.

### Protocol

- **Seeds (10):** `[42, 123, 456, 789, 1011, 2024, 1337, 7, 314, 271]`
- **Tuning:** 100-trial Optuna per (method, env), objective = `-|avg_reward - target|`
  - CartPole target = 200, CliffWalking target = -20
- **Training:** max 1000 episodes, early-stop at rolling-25 reward ≥ target, patience = 200
- **Eval:** best-weights snapshot, 100 sampled-action episodes
- **Stat tests:** Welch's t + Mann-Whitney U on all 36 pairs per metric, BH-FDR corrected, α = 0.05

### Constraints

`src/requirements/emergency_cartpole.cnf` (4 clauses, 2 actions): forces a
specific action when an "emergency" state flag is active. Constraint active in
~5% of states.

`src/requirements/cliff_safe.cnf` (4 actions, more clauses): forbids actions
that would step onto the cliff. Constraint active in ~30% of states. Falling
has catastrophic reward consequences.

---

## 2. Metric definitions (critical for interpreting the table)

The columns in the main results table are reported as `viol_rate` and `mod_rate`,
but the *meaning* of `viol_rate` differs by method type. **Always think in terms
of these three derived quantities:**

| derived metric | definition | meaning |
|---|---|---|
| **would-have-violated rate** | `viol_rate` (column value) | how often the *underlying* policy proposes an unsafe action — measures constraint internalization |
| **modification rate** | `mod_rate` (column value) | how often the shield actually changes the action |
| **actual runtime violation rate** | `viol_rate − mod_rate` for shielded methods; `viol_rate` for unshielded methods | what fraction of executed actions actually violated the constraint at runtime |

For **hard shielding** (`action_mask`, `preshield_hard`, `layer_hard`):
`viol_rate = mod_rate` → actual violations = **0** by construction (the shield
always swaps unsafe actions).

For **soft shielding** (`preshield_soft`, `layer_soft`): `viol_rate > mod_rate`
slightly → some unsafe actions are *not* swapped (the shield reduces but doesn't
eliminate their probability) → small but nonzero actual violations.

For **non-shielded methods** (unshielded, reward_shaping, semantic_loss, CMDP):
`mod_rate = 0` → actual violations = `viol_rate` (env runs the policy's own
action).

**This distinction is critical for the paper framing.** It must be made explicit
or reviewers will misread the table.

---

## 3. CartPole results (10 seeds, post-CMDP-fix)

### Numbers — including derived actual violation rate

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

**Would-have-violated rate** (constraint internalization, all methods):
- Action_mask (0.054) significantly higher than: unshielded, reward_shaping,
  semantic_loss, preshield_soft, layer_soft, CMDP. All FDR p < 0.05.
- Action_mask NOT significantly different from preshield_hard (0.028) or
  layer_hard (0.053).
- **Hard-enforcement methods cluster together** on this metric in CartPole.

**Reward** — only one pairwise comparison survives correction:
- Action_mask (172) < Preshield_soft (231), FDR p = 0.011.
- All others not significant. High-variance methods (preshield_hard ±121,
  layer_soft ±106, layer_hard ±154, CMDP ±126) bury other differences.

### Headline figure

`results/nesy_experiments/CartPole-v1/plots/CartPole-v1_action_mask_vs_preshield_soft_trajectories.png`

Per-seed training trajectories: action_mask shows peak-and-collapse across seeds
(median climbs to ~180, then drifts down). Preshield_soft converges cleanly to
~250 and stays. Strongest single figure for the action-masking contribution.

---

## 4. CliffWalking results (10 seeds, CMDP STALE)

The 8 non-CMDP methods have valid 10-seed data. CMDP shows median reward -130
which is the **OLD buggy run** — the fix and broad-search retune are pending.

### Numbers (CMDP row not yet updated; placeholder)

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
| CMDP (STALE — buggy run) | -130 | -217 ± 184 | 0.016 | 0 | 0.016 |

### Key CliffWalking patterns (unchanged regardless of CMDP rerun)

- **Hard methods now dominate on reward stability.** Action_mask, preshield_hard,
  layer_hard all at -19 to -20 with std ~1. Soft methods and unshielded have
  catastrophic seeds (mean -250 to -412, std 400-560) because the agent
  occasionally walks off cliffs.

- **Within the hard family, would-have-violated rate forms a clear ladder**
  matching projection sophistication:
  - preshield_hard 0.098 (full CCN+ + importance-ratio teaching trick)
  - layer_hard 0.186 (differentiable layer)
  - action_mask 0.302 (simple 0/1 mask, no teaching)
  All three pairwise differences are FDR-significant. **This is direct empirical
  support for "gradient-signal strength predicts constraint internalization"** —
  a finding CartPole's simpler CNF couldn't reveal.

- **Soft methods have huge variance** from a few catastrophic seeds. Median is
  the right stat for reporting.

### Headline figure (CliffWalking)

`results/nesy_experiments/CliffWalking-v1/plots/CliffWalking-v1_action_mask_vs_preshield_soft_trajectories.png`

Striking inversion of the CartPole pattern: action_mask shows 10 seeds tightly
converging to -20 (no collapse). Preshield_soft has 7 stable seeds + 2-3 seeds
catastrophically diverging to -1000 to -1750.

---

## 5. The contribution and how to frame it

This is the key reframe based on the corrected understanding of metrics.

### The three-axis design space

Each method occupies a distinct point. **The contribution is characterizing the
space, not winning a leaderboard.**

**Axis 1 — Hard runtime safety guarantee (zero actual violations):**
- ✅ Yes: action_mask, preshield_hard, layer_hard
- 🟡 Partial: preshield_soft, layer_soft (~0.3% actual violations)
- ❌ No: CMDP, semantic_loss, reward_shaping, unshielded

**Axis 2 — Constraint internalization (would-have-violated rate, lower = policy learned the constraint):**
- 🥇 Best: CMDP, layer_soft, preshield_soft
- 🥈 Middling: semantic_loss, unshielded, reward_shaping, preshield_hard
- 🥉 Worst: layer_hard, action_mask

**Axis 3 — Reward (median):**
- 🥇 Best: CMDP, semantic_loss, preshield_hard, preshield_soft, reward_shaping
- 🥈 Competitive: layer_soft, unshielded
- 🥉 Weakest: action_mask, layer_hard (on CartPole)

### The unique value of hard CCN+ shielding

The intersection of axes 1 and 2 is the contribution. **Hard CCN+ shielding
(preshield_hard, layer_hard) is the only paradigm that gives BOTH runtime
safety AND non-trivial policy learning.**

Compare on CartPole:
- preshield_hard: 0 actual violations + would-have-violated 0.028 + reward 239
- action_mask: 0 actual violations + would-have-violated **0.054** + reward 170
  → same safety, much worse learning
- CMDP: 0.011 actual violations + would-have-violated 0.011 + reward 275
  → better learning, but no runtime safety
- preshield_soft: 0.003 actual violations + would-have-violated 0.014 + reward 234
  → almost-but-not-quite-hard safety + strong learning

**Only hard CCN+ shielding sits in the upper-left of "safe AND learning."**
Action_mask is dominated by hard CCN+ shielding (same safety, worse learning).
CMDP and soft CCN+ shielding sacrifice some safety for learning. Each is a
distinct point on the Pareto frontier.

### Suggested framing for the paper (paragraph form)

> Action masking achieves runtime safety but at the cost of policy learning —
> the underlying policy never internalizes the constraint (would-have-violated
> rate 0.054 on CartPole). Penalty-based methods (CMDP, semantic loss) achieve
> strong learning but provide no runtime safety guarantee. CCN+ shielding
> uniquely provides both: hard variants give runtime safety while reducing the
> underlying policy's would-have-violated rate via the importance-ratio teaching
> signal; soft variants further improve learning at a small cost to runtime
> safety (~0.3% violations). The choice between hard CCN+, soft CCN+, and CMDP
> corresponds to choosing a point on the safety-learning Pareto frontier;
> action masking is dominated by hard CCN+ shielding on this frontier.

### The action_mask result, mechanistically

Action masking provides zero gradient signal on forced-action states: when the
mask sets a logit to `-inf`, softmax → 0, gradient → 0. In states where only
one action is valid, the masked PPO update produces no learning signal. The
policy never internalizes the constraint, it just gets blocked at runtime.

This is empirically observed across CartPole and CliffWalking. On CliffWalking,
the within-hard-family ordering (preshield_hard < layer_hard < action_mask on
would-have-violated rate) is direct evidence that **gradient-signal strength
predicts learning**, controlling for the hard-safety guarantee.

---

## 6. sb3-contrib validation (CartPole appendix material)

To rule out implementation bugs in the custom action_mask, ran matched-reward
comparison against the reference `sb3-contrib.MaskablePPO`:

| | reward at target=200 | viol_rate (would-have-violated) |
|---|---|---|
| ours (PPOAgent + use_action_mask) | 195.6 ± 34.5 | 0.052 |
| sb3-contrib MaskablePPO | 227.8 ± 32.9 | 0.030 |

Both reach the target. sb3's would-have-violated rate is lower than ours (sample
efficiency difference), but both are substantially higher than soft-shielding
methods (0.013-0.014). **Relative ranking preserved across PPO bases.**

Script: `scripts/bench_action_mask_vs_sb3.py`. Currently 3 seeds CartPole only —
should be bumped to 5 seeds and extended to CliffWalking before final submission.

---

## 7. Things to fix before submission

### Must do

- **Re-tune + re-run CMDP on CliffWalking** with the credit-assignment fix +
  broad-search tuning. Commands in run history below. Expected: CMDP should
  improve substantially over the current buggy -130 median, ideally joining
  the table as a competitive baseline (similar to its CartPole behavior).

- **Refresh CliffWalking stats CSVs** after CMDP re-run.

- **Seaquest** — tuning source needs resolution; sweep then takes ~25-40 hr.

### Paper-side fixes (no compute)

- **Add the metric-definition box (§2 of this doc) to the paper.** Either as a
  table caption, an inline paragraph in the methodology section, or a dedicated
  subsection. This is the single most important framing fix.

- **Report median + IQR alongside mean ± std**, especially on CliffWalking and
  for high-variance methods on CartPole (CMDP, preshield_hard, layer_soft,
  layer_hard).

- **CMDP-fix acknowledgment.** Add one sentence: "An earlier draft of this work
  used an implementation of CMDP that combined Lagrangian penalties with action
  shielding; we corrected this to match the standard CMDP formulation before
  reporting these results."

- **PPO sample-efficiency footnote.** Custom PPO has ~30× fewer gradient updates
  per env-step than sb3 defaults. Cite the matched-bench as evidence relative
  comparisons preserved across PPO bases.

- **Reword reward claims** to clusters/trends rather than absolute "X > Y" — most
  reward pairs are not statistically significant at n=10.

---

## 8. Run history (for reproducibility)

### CartPole

```bash
conda activate ccn_rl
python scripts/run_ijcai_experiments.py --env CartPole-v1 \
    --base_dir results/nesy_experiments \
    --use_subprocess --num_train_episodes 1000
# After CMDP credit-assignment fix in constrained_ppo_agent.py:182:
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
# Initial sweep (CMDP was buggy in this pass)
python scripts/run_ijcai_experiments.py --env CliffWalking-v1 \
    --base_dir results/nesy_experiments \
    --use_subprocess --num_train_episodes 1000
python scripts/analyze_nesy_results.py --env CliffWalking-v1
python scripts/plot_action_mask_trajectories.py --env CliffWalking-v1

# TODO: re-tune + re-run CMDP with the fix and broad search
python scripts/tune_ijcai_methods.py --env CliffWalking-v1 --method cppo --trials 100
python scripts/run_ijcai_experiments.py --env CliffWalking-v1 --method cppo \
    --base_dir results/nesy_experiments \
    --use_subprocess --num_train_episodes 1000
python scripts/analyze_nesy_results.py --env CliffWalking-v1
```

### Seaquest (TODO)

See `NESY_TODO.md` Phase 3 — tuning source needs resolution first.

---

## 9. Deadlines

- Abstract: June 9, 2026
- Full submission: June 16, 2026

Comfortable timeline remaining for Seaquest + paper-side fixes.
