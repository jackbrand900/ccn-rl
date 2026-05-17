# NeSy submission — experiments todo

**Deadlines:** abstract June 9, full submission June 16.
**Today:** May 17. ~30 days.

**Approach:** clean fresh runs of every method × every env into `results/nesy_experiments/`. Tuned hyperparameters reused from `config/ijcai_tuned/` where they exist; new ones tuned where they don't.

**Bumped training budget:** `--num_train_episodes 1000` + `early_stop_patience=200` (vs IJCAI's 500/100). Methods that hit target early-stop the same; struggling seeds get more time.

---

## Phase 1 — CartPole ✅ DONE

All 9 methods × 10 seeds completed in `results/nesy_experiments/CartPole-v1/`.

**Key results** (mean ± std, viol_rate):
- Action mask: 172 ± 17 reward, **0.054** viol_rate (highest among shielded)
- Preshield soft: 231 ± 35 reward, 0.014 viol_rate (best safety)
- Layer soft: 264 ± 106 reward, 0.013 viol_rate (best safety, high variance)
- Unshielded: 194 ± 27 reward, 0.018 viol_rate

**Statistical analysis** (FDR-corrected α=0.05):
- Viol_rate: action_mask significantly higher than all soft methods + unshielded + semantic_loss + CMDP (***)
- Viol_rate: action_mask NOT distinguishable from preshield_hard or layer_hard (the other hard methods cluster together)
- Reward: only action_mask vs preshield_soft is significant after correction; high-variance methods bury other differences
- Outputs: `results/nesy_experiments/CartPole-v1_per_method_stats.csv`, `_pairwise_reward.csv`, `_pairwise_viol_rate.csv`

**Figures generated:**
- `results/nesy_experiments/CartPole-v1/plots/CartPole-v1_action_mask_vs_preshield_soft_trajectories.png` (the per-seed peak-and-collapse vs converge-and-plateau contrast)
- Standard per-method plots (violation rate, reward, etc.) also in `plots/`

---

## Phase 2 — CliffWalking (next up)

`ppo_action_mask` is the only method without a CliffWalking tuned config. Tune it first, then run.

### 2a. Tune action_mask

```bash
conda activate ccn_rl
python scripts/tune_ijcai_methods.py --env CliffWalking-v1 --method ppo_action_mask --trials 100
```
~1-2 hr. Writes `config/ijcai_tuned/ppo_action_mask_CliffWalking-v1_params.yaml`.

### 2b. Full sweep

```bash
python scripts/run_ijcai_experiments.py --env CliffWalking-v1 --base_dir results/nesy_experiments --use_subprocess --num_train_episodes 1000
```
- 9 methods × 10 seeds = 90 runs
- ~7-10 hr unattended (the bumped budget may extend this)

### 2c. Analyze

```bash
python scripts/analyze_nesy_results.py --env CliffWalking-v1
python scripts/plot_action_mask_trajectories.py --env CliffWalking-v1
```

---

## Phase 3 — Seaquest

### 3a. Resolve tuning approach

No Seaquest configs in `config/ijcai_tuned/`. Pick one path:

- **(A) Find existing tuning state.** Check `results/ijcai_experiments/ALE_Seaquest-v5/*/results.json` for an `agent_kwargs` field, and the repo for any non-ijcai_tuned config files / Optuna study DBs. If found, port hyperparams into proper YAML files.
- **(B) Tune from scratch.** All 9 methods × 100 trials. Each trial is slow (long Seaquest episodes). Realistic: ~20-40 hr per method = several hundred hours total. Compromise option: reduce to 30-50 trials per method.

Resolve before scheduling anything Seaquest-heavy. **Status: not investigated yet.**

### 3b. Tune (if needed)

```bash
# Per method, e.g.:
python scripts/tune_ijcai_methods.py --env ALE/Seaquest-v5 --method ppo_action_mask \
    --trials 100 --use_ram_obs --max_episode_steps 2000
```

### 3c. Full sweep at 5 seeds

```bash
python scripts/run_ijcai_experiments.py --env ALE/Seaquest-v5 \
    --base_dir results/nesy_experiments --use_subprocess --num_train_episodes 1000
```
- 9 methods × 5 seeds = 45 runs
- ~25-40 hr unattended
- Stay at 5 seeds; 10 seeds is ~10 days of compute. Justify in paper protocol section.

### 3d. Analyze

```bash
python scripts/analyze_nesy_results.py --env ALE/Seaquest-v5
python scripts/plot_action_mask_trajectories.py --env ALE/Seaquest-v5
```

---

## Phase 4 — sb3-contrib validation appendix

Currently 3 seeds CartPole only. Bump to 5 seeds + add CliffWalking. (Seaquest matched-bench may need Atari adaptation — check before running.)

```bash
python scripts/bench_action_mask_vs_sb3.py --env CartPole-v1 --mode matched \
    --target-reward 200 --seeds 42 123 456 789 1011 --timesteps 200000
python scripts/bench_action_mask_vs_sb3.py --env CliffWalking-v1 --mode matched \
    --target-reward -20 --seeds 42 123 456 789 1011 --timesteps 200000
```
~30-60 min total.

---

## Paper-side fixes (no compute, do anytime)

- **"Violation rate" column ambiguity.** The column conflates "would-have-violated by raw policy" (shielded methods) with "actual violations" (unshielded, reward_shaping, semantic_loss, cppo). Rename per-row or split into two columns. **Single most-likely reviewer flag.**
- **PPO sample-efficiency footnote.** Acknowledge custom PPO has ~30× fewer grad updates per env-step than sb3 defaults (rollout 64 vs 2048, 1 vs 32 minibatches). Note that the matched-bench shows relative comparisons preserved.
- **Median + IQR in tables.** Already computed; add to the main results table for outlier robustness, especially on layer_hard (152 median vs 181 mean), cppo (138 vs 162), preshield_hard (239 vs 270).
- **Reword reward claims.** Only one pairwise reward comparison survives FDR correction at n=10. Phrase reward findings around clusters/trends, not "X > Y" claims, except for the action_mask vs preshield_soft pair where it's significant.

---

## Estimated remaining compute

| phase | wall time |
|---|---|
| 1 — CartPole | ✅ done |
| 2a — CliffWalking action_mask tune | 1-2 hr |
| 2b — CliffWalking sweep | 7-10 hr |
| 3a — Seaquest tuning resolution | 5 min lookup vs days of tuning |
| 3c — Seaquest sweep | 25-40 hr (assuming 3a finds existing configs) |
| 4 — sb3 bench expansion | 30-60 min |
| **Total remaining** | **~35-55 hr** unattended + Seaquest tuning TBD |

Fits comfortably in the 30 days remaining.

---

## Suggested order

1. **Now**: Phase 2a (CliffWalking action_mask tune) — fire and forget
2. **In parallel**: Phase 3a (Seaquest tuning lookup) — 5 min that gates the long pole
3. **After 2a finishes**: Phase 2b (CliffWalking sweep) — overnight
4. **After 2b**: Phase 2c analysis
5. **Then**: Phase 3 once tuning approach decided
6. **Anywhere**: Phase 4 (bench expansion)
7. **In parallel with everything**: paper-side fixes
