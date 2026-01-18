#!/usr/bin/env python3
"""
Wrapper script to run tuning for each method independently in separate processes.
This prevents memory accumulation by running each method in its own process.
"""

import subprocess
import sys
import argparse
import time
import gc
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.tune_ijcai_methods import METHODS, TARGET_REWARDS, TRAINING_TARGET_REWARDS

def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning for each method independently (separate processes)")
    parser.add_argument('--env', type=str, required=True,
                       choices=['CartPole-v1', 'CliffWalking-v1', 'MiniGrid-DoorKey-5x5-v0', 'ALE/Seaquest-v5'],
                       help='Environment to tune')
    parser.add_argument('--trials', type=int, default=15,
                       help='Number of Optuna trials per method (default: 15)')
    parser.add_argument('--train_episodes', type=int, default=500,
                       help='Number of training episodes during tuning (default: 500)')
    parser.add_argument('--eval_episodes', type=int, default=50,
                       help='Number of evaluation episodes during tuning')
    parser.add_argument('--target_reward', type=float, default=None,
                       help='Target reward (default: use predefined)')
    parser.add_argument('--start_from', type=str, default=None,
                       help='Start from this method (skip previous methods)')
    parser.add_argument('--skip_completed', action='store_true',
                       help='Skip methods that already have config files')
    parser.add_argument('--method', type=str, default=None,
                       help='Tune only this specific method (default: all)')
    parser.add_argument('--max_episode_steps', type=int, default=None,
                       help='Maximum steps per episode (default: env-specific)')
    parser.add_argument('--use_ram_obs', action='store_true',
                       help='Use RAM observations instead of image observations (for Atari games)')
    
    args = parser.parse_args()
    
    # Filter out CPO (commented out)
    methods_to_run = [m for m in METHODS if m['name'] != 'cpo']
    
    # Filter to specific method if requested
    if args.method:
        methods_to_run = [m for m in methods_to_run if m['name'] == args.method]
        if not methods_to_run:
            print(f"Error: Method '{args.method}' not found")
            print("Available methods:")
            for m in METHODS:
                if m['name'] != 'cpo':
                    print(f"  - {m['name']} ({m['display_name']})")
            sys.exit(1)
    
    # Skip completed methods if requested
    if args.skip_completed:
        config_dir = Path("config/ijcai_tuned")
        completed_methods = set()
        if config_dir.exists():
            for config_file in config_dir.glob(f"*_{args.env.replace('/', '_')}_params.yaml"):
                method_name = config_file.stem.replace(f"_{args.env.replace('/', '_')}_params", "")
                completed_methods.add(method_name)
        
        original_count = len(methods_to_run)
        methods_to_run = [m for m in methods_to_run if m['name'] not in completed_methods]
        skipped_count = original_count - len(methods_to_run)
        if skipped_count > 0:
            print(f"Skipping {skipped_count} already-completed method(s)")
    
    # Start from a specific method if requested
    if args.start_from:
        start_index = None
        for i, method in enumerate(methods_to_run):
            if method['name'] == args.start_from:
                start_index = i
                break
        
        if start_index is None:
            print(f"Warning: Method '{args.start_from}' not found. Available methods:")
            for m in methods_to_run:
                print(f"  - {m['name']} ({m['display_name']})")
            sys.exit(1)
        
        methods_to_run = methods_to_run[start_index:]
        print(f"Starting from method: {methods_to_run[0]['display_name']}")
    
    print(f"\n{'='*80}")
    print(f"Running Tuning for {args.env}")
    print(f"Methods: {len(methods_to_run)}")
    print(f"Each method will run in a separate process to prevent memory accumulation")
    print(f"{'='*80}\n")
    
    results = []
    start_time = time.time()
    
    for i, method in enumerate(methods_to_run, 1):
        method_start = time.time()
        print(f"\n{'#'*80}")
        print(f"# Method {i}/{len(methods_to_run)}: {method['display_name']}")
        print(f"{'#'*80}\n")
        
        # Build command (training target is handled internally via TRAINING_TARGET_REWARDS)
        base_target = args.target_reward if args.target_reward is not None else TARGET_REWARDS.get(args.env)
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "tune_ijcai_methods.py"),
            "--env", args.env,
            "--method", method['name'],
            "--trials", str(args.trials),
            "--train_episodes", str(args.train_episodes),
            "--eval_episodes", str(args.eval_episodes),
        ]
        
        if args.target_reward is not None:
            cmd.extend(["--target_reward", str(args.target_reward)])
        
        if args.max_episode_steps is not None:
            cmd.extend(["--max_episode_steps", str(args.max_episode_steps)])
        
        if args.use_ram_obs:
            cmd.append("--use_ram_obs")
        
        # Run in separate process
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            method_time = time.time() - method_start
            method_time_str = f"{int(method_time // 60)}m {int(method_time % 60)}s"
            
            print(f"\n[✓] {method['display_name']} completed in {method_time_str}")
            
            # Estimate remaining time
            elapsed_total = time.time() - start_time
            avg_time_per_method = elapsed_total / i
            remaining_methods = len(methods_to_run) - i
            estimated_remaining = avg_time_per_method * remaining_methods
            
            if remaining_methods > 0:
                remaining_str = f"{int(estimated_remaining // 60)}m {int(estimated_remaining % 60)}s"
                print(f"[⏱] Estimated time remaining: {remaining_str} ({remaining_methods} methods left)")
            
            results.append({
                'method': method['display_name'],
                'status': 'success',
                'time': method_time_str
            })
            
        except subprocess.CalledProcessError as e:
            print(f"\n[✗] {method['display_name']} failed with exit code {e.returncode}")
            results.append({
                'method': method['display_name'],
                'status': 'failed',
                'time': 'N/A'
            })
        except KeyboardInterrupt:
            print(f"\n[!] Interrupted by user")
            print(f"Completed {i-1}/{len(methods_to_run)} methods")
            sys.exit(1)
        
        # Aggressive memory cleanup between methods
        gc.collect()
        # Longer delay to ensure process cleanup and memory release
        time.sleep(5)
        
        # Force Python to release memory
        import os
        if hasattr(os, 'system'):
            # Try to hint to OS to release memory (works on Linux/Mac)
            try:
                import ctypes
                if sys.platform == 'darwin':  # macOS
                    libc = ctypes.CDLL('libc.dylib')
                    libc.malloc_trim(0)
            except:
                pass
    
    # Final summary
    total_time = time.time() - start_time
    total_time_str = f"{int(total_time // 3600)}h {int((total_time % 3600) // 60)}m"
    
    print(f"\n{'='*80}")
    print(f"TUNING COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time_str}")
    print(f"\nResults:")
    for r in results:
        status_icon = "✓" if r['status'] == 'success' else "✗"
        print(f"  {status_icon} {r['method']:<30} {r['time']:<10} {r['status']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

