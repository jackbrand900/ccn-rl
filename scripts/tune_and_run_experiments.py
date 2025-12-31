#!/usr/bin/env python3
"""
Convenience script to tune hyperparameters and then immediately run experiments.
"""

import subprocess
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters then run experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tune and run CartPole
  python scripts/tune_and_run_experiments.py --env CartPole-v1 --trials 20
  
  # Tune specific methods only, then run experiments
  python scripts/tune_and_run_experiments.py --env CartPole-v1 --method cppo ppo_vanilla --trials 20
  
  # Skip tuning if configs exist, just run experiments
  python scripts/tune_and_run_experiments.py --env CartPole-v1 --skip_tuning
        """
    )
    parser.add_argument('--env', type=str, required=True,
                       choices=['CartPole-v1', 'CliffWalking-v1', 'MiniGrid-DoorKey-5x5-v0', 'ALE/Seaquest-v5'],
                       help='Environment to tune and run')
    parser.add_argument('--trials', type=int, default=20,
                       help='Number of Optuna trials per method (default: 20)')
    parser.add_argument('--method', type=str, nargs='+', default=None,
                       help='Specific methods to tune (default: all)')
    parser.add_argument('--skip_tuning', action='store_true',
                       help='Skip tuning step, just run experiments')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip already completed experiment runs')
    parser.add_argument('--train_episodes', type=int, default=500,
                       help='Number of training episodes (default: 500)')
    parser.add_argument('--eval_episodes', type=int, default=100,
                       help='Number of evaluation episodes (default: 100)')
    
    args = parser.parse_args()
    
    # Step 1: Tune hyperparameters
    if not args.skip_tuning:
        print(f"\n{'='*80}")
        print("STEP 1: Tuning Hyperparameters")
        print(f"{'='*80}\n")
        
        tune_cmd = [
            sys.executable,
            str(Path(__file__).parent / "tune_ijcai_methods_parallel.py"),
            "--env", args.env,
            "--trials", str(args.trials),
            "--train_episodes", str(args.train_episodes),
            "--eval_episodes", str(args.eval_episodes),
        ]
        
        if args.method:
            for method in args.method:
                tune_cmd.extend(["--method", method])
        
        result = subprocess.run(tune_cmd, check=False)
        if result.returncode != 0:
            print(f"\n[ERROR] Tuning failed with exit code {result.returncode}")
            print("Continuing to experiments anyway...")
        else:
            print(f"\n[✓] Tuning completed successfully")
    else:
        print(f"\n[ℹ] Skipping tuning step (--skip_tuning)")
    
    # Step 2: Run experiments
    print(f"\n{'='*80}")
    print("STEP 2: Running Experiments")
    print(f"{'='*80}\n")
    
    exp_cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_ijcai_experiments.py"),
        "--env", args.env,
        "--num_train_episodes", str(args.train_episodes),
        "--num_eval_episodes", str(args.eval_episodes),
    ]
    
    if args.method:
        exp_cmd.extend(["--method"] + args.method)
    
    if args.skip_existing:
        exp_cmd.append("--skip_existing")
    
    result = subprocess.run(exp_cmd, check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] Experiments failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print("COMPLETE: Tuning and Experiments Finished")
    print(f"{'='*80}")
    print(f"Results saved to: results/ijcai_experiments/{args.env.replace('/', '_')}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

