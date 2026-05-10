# Constraint-Satisfying Reinforcement Learning via a Neuro-Symbolic Projection Layer

This repository contains the implementation and experimental code for our submission on safe reinforcement learning using constraint-based shielding methods. The core idea is to train RL agents that not only maximize rewards but also satisfy logical safety constraints throughout training.

## What's This About?

In traditional RL, agents learn to maximize rewards but often violate important safety constraints during training (and sometimes even after). This is a problem in real-world applications where safety matters. We explore several approaches to enforce constraints during both training and execution:

- **Shielding methods** that actively prevent unsafe actions
- **Loss-based approaches** that penalize constraint violations
- **Constrained optimization** techniques that balance performance and safety

The key challenge is maintaining good task performance while minimizing constraint violations during the entire learning process, not just at convergence.

## Methods Compared

We evaluate 8 different approaches across three benchmark environments:

1. **PPO (Unshielded)** - Standard PPO baseline with no safety mechanisms
2. **PPO + Reward Shaping** - Adds penalty terms to the reward for constraint violations
3. **PPO + Semantic Loss** - Incorporates constraint satisfaction into the loss function
4. **PPO + Pre-emptive Shield (Soft)** - Modifies unsafe actions before execution, allows partial violations
5. **PPO + Pre-emptive Shield (Hard)** - Strictly blocks all constraint-violating actions
6. **PPO + Layer Shield (Soft)** - Integrates the shield as a differentiable network layer
7. **PPO + Layer Shield (Hard)** - Shield layer with strict enforcement
8. **CMDP (Constrained MDP)** - Constrained policy optimization baseline

The "soft" vs "hard" distinction is important: soft shields gently guide the agent away from violations while hard shields completely block them. Each has tradeoffs in terms of learning efficiency and constraint satisfaction.

## Environments

We test on three environments with different characteristics:

- **CartPole-v1** - Classic control task with simple safety constraints (keep the pole centered)
- **CliffWalking-v1** - Grid world where the agent must avoid falling off cliffs
- **Seaquest (Atari)** - Atari game using RAM observations with oxygen management constraints

Each environment has hand-crafted logical constraints defined in CNF (Conjunctive Normal Form) under `src/requirements/`.

## Setup

### Requirements

We use Python 3.11 with PyTorch and several RL libraries. The easiest way to get started is with conda:

```bash
conda env create -f environment.yml
conda activate ccn_rl
```

If you prefer pip, you can install dependencies directly:

```bash
pip install -r requirements.txt
```

Note: Some environments (like Atari) require accepting ROM licenses. The AutoROM package should handle this automatically, but if you run into issues, you may need to run `AutoROM --accept-license` manually.

### Project Structure

```
ccn-rl/
├── config/
│   └── ijcai_tuned/          # Hyperparameters for each method and environment
├── results/
│   └── ijcai_experiments/    # Experimental results, plots, and analysis
├── scripts/
│   ├── run_ijcai_experiments.py         # Main experiment runner
│   ├── tune_ijcai_methods.py            # Hyperparameter tuning
│   └── generate_training_violation_curves.py  # Analysis scripts
├── src/
│   ├── agents/               # PPO, A2C, DQN, CPPO implementations
│   ├── envs/                 # Custom environment wrappers
│   ├── models/               # Neural network architectures
│   ├── requirements/         # Logical constraints (CNF/linear files)
│   └── utils/                # Shield controller, monitoring, helpers
└── environment.yml          # Conda environment specification
```

## Running Experiments

### Quick Start

To reproduce our IJCAI results, run:

```bash
python scripts/run_ijcai_experiments.py --env CartPole-v1
```

This will:
- Load the tuned hyperparameters from `config/ijcai_tuned/`
- Train all 8 methods for 5 seeds each (40 runs total)
- Track violation rates, modification rates, and rewards per episode
- Generate comparison plots automatically

### Running on Multiple Environments

```bash
python scripts/run_ijcai_experiments.py --env CartPole-v1 CliffWalking-v1 ALE/Seaquest-v5
```

### Running Specific Methods

If you only want to test certain approaches:

```bash
python scripts/run_ijcai_experiments.py --env CartPole-v1 \
    --method ppo_unshielded cppo ppo_preshield_soft
```

### Important Flags

The experiment runner supports several useful flags:

- `--env` - Specify which environments to run (default: all three)
- `--method` - Specify which methods to run (default: all eight)
- `--skip_existing` - Skip runs that already have results
- `--test` - Quick test mode (1 episode only)
- `--use_subprocess` - Run in separate processes to free memory between runs

**Note on Seaquest:** The experiment runner automatically uses RAM observations for Seaquest (no flag needed). However, if you're running the tuning scripts manually, you need to add `--use_ram_obs`:

```bash
python scripts/tune_ijcai_methods.py --env ALE/Seaquest-v5 --use_ram_obs --trials 100
```

### Viewing Results

After running experiments, check out:
- `results/ijcai_experiments/summary_table.txt` - Overall performance summary
- `results/ijcai_experiments/{env}/plots/` - Training curves and comparisons
- `results/ijcai_experiments/training_violation_plots/` - Violation analysis

## Hyperparameter Tuning

We use Optuna for hyperparameter optimization. Each method and environment combination gets tuned separately:

```bash
python scripts/tune_ijcai_methods.py --env CartPole-v1 \
    --method ppo_unshielded --trials 100
```

The tuning process optimizes for a balance between task performance and constraint satisfaction. Results are saved in `config/ijcai_tuned/` and automatically used by the experiment runner.

For parallel tuning across multiple methods:

```bash
python scripts/tune_ijcai_methods_parallel.py --env CartPole-v1 --trials 100
```

## Key Metrics

We track three primary metrics during training:

- **Violation Rate**: Proportion of steps where constraints are violated (lower is better)
- **Modification Rate**: How often the shield changes the agent's actions (lower means more natural behavior)
- **Reward**: Task performance (should stay high despite safety enforcement)

The ideal method achieves high rewards with minimal violations and minimal modifications. See `results/ijcai_experiments/summary_table.txt` for complete experimental results.

## Implementation Details

### Shields

The shield implementation uses the `pishield` library (the Python implementation of CCN+) to convert logical constraints into differentiable neural network layers.

The `ShieldController` (`src/utils/shield_controller.py`) handles:
- Loading constraint specifications
- Computing context-dependent flags from observations
- Applying shields to action distributions (both hard and soft modes)

### Training

We use PPO as the base algorithm, implemented in `src/agents/ppo_agent.py`. The agent supports:
- Pre-emptive shielding (modify actions before environment step)
- Layer-integrated shielding (shield as part of policy network)
- Semantic loss (constraint violations added to training loss)
- Reward shaping (penalties for violations)

Training loop is in `src/train.py` with comprehensive logging and monitoring.

## Troubleshooting

**Out of memory during training:**
- Reduce batch size or network size in the config files
- Use CPU instead of GPU for smaller environments (CartPole, CliffWalking)

**Atari ROM issues:**
- Run `AutoROM --accept-license` to download ROMs
- Make sure `ale-py` is installed correctly
- Note: We use RAM observations for Seaquest, not visual input

**Slow training:**
- Seaquest takes longer due to longer episodes (~4-6 hours per run)
- CartPole and CliffWalking are much faster (minutes to hours)
- Consider reducing `--train-episodes` for quick tests

## License

This project is part of academic research. Please contact the authors for licensing information.

## Acknowledgments

This work builds on the `pishield` library for differentiable shielding. Thanks to the RL and safe learning communities for the tools and benchmarks.
