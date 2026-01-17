# Running IJCAI Experiments

## Quick Start

### 1. Run CartPole Experiments
```bash
conda activate thesis_env
python scripts/run_ijcai_experiments.py --env CartPole-v1
```

### 2. What Happens
- **Loads tuned hyperparameters** from `config/ijcai_tuned/` automatically
- **Runs 7 methods** × **5 seeds** = 35 total runs
- **Tracks metrics per episode**: violation_rate, modification_rate, reward
- **Aggregates across 5 runs**: computes mean ± std
- **Generates graphs automatically** after completion

### 3. Output Structure
```
results/ijcai_experiments/
└── CartPole-v1/
    ├── cppo/
    │   ├── train_metrics_run1.csv  (per-episode metrics)
    │   ├── train_metrics_run2.csv
    │   ├── ...
    │   └── aggregated_results.json
    ├── ppo_postshield_hard/
    │   └── ...
    └── plots/
        ├── CartPole-v1_violation_rates_over_time.png
        ├── CartPole-v1_modification_rates_over_time.png
        └── CartPole-v1_*_rates.png (individual methods)
```

## Generated Graphs

1. **Violation Rates Over Time** - All methods compared (mean ± std across 5 runs)
2. **Modification Rates Over Time** - All methods compared (mean ± std across 5 runs)
3. **Individual Method Plots** - Violation + modification rates for each method

All graphs are:
- Publication quality (300 DPI, Times New Roman font)
- Include confidence intervals (mean ± std)
- Ready for submission

## Options

```bash
# Run specific methods only
python scripts/run_ijcai_experiments.py --env CartPole-v1 --method cppo ppo_preshield_hard

# Run multiple environments
python scripts/run_ijcai_experiments.py --env CartPole-v1 CliffWalking-v1

# Skip already completed runs
python scripts/run_ijcai_experiments.py --env CartPole-v1 --skip_existing

# Show summary without running
python scripts/run_ijcai_experiments.py --show_summary
```

## Metrics Tracked

- **Violation Rate**: Per-step rate of constraint violations (primary metric)
- **Modification Rate**: Per-step rate of shield modifications (primary metric)
- **Reward**: Average reward per episode (secondary metric, should be similar across methods)

All metrics are saved per-episode in CSV files for detailed analysis.
