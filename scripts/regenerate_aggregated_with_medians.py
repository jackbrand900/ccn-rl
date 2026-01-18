#!/usr/bin/env python3
"""
Regenerate aggregated_results.json files with median calculations.
Reads from existing experiment_summary.csv or individual run results.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.run_ijcai_experiments import aggregate_results, ENVIRONMENTS, METHODS

def regenerate_from_csv(base_dir='results/ijcai_experiments'):
    """Regenerate aggregated results from experiment_summary.csv"""
    
    csv_path = os.path.join(base_dir, 'experiment_summary.csv')
    if not os.path.exists(csv_path):
        print(f"❌ {csv_path} not found. Cannot regenerate.")
        return
    
    print(f"Reading from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    for env_name in ENVIRONMENTS:
        env_safe = env_name.replace('/', '_')
        env_dir = os.path.join(base_dir, env_safe)
        
        if not os.path.exists(env_dir):
            continue
        
        env_df = df[df['env'] == env_name]
        
        for method in METHODS:
            method_dir = os.path.join(env_dir, method['name'])
            method_df = env_df[env_df['method'] == method['name']]
            
            if method_df.empty:
                continue
            
            # Convert DataFrame rows to result dictionaries
            method_results = []
            for _, row in method_df.iterrows():
                result = {
                    'avg_reward': row['avg_reward'],
                    'std_reward': row['std_reward'],
                    'max_reward': row['max_reward'],
                    'min_reward': row['min_reward'],
                    'avg_shield_mod_rate': row['avg_shield_mod_rate'],
                    'avg_violations_per_step': row['avg_violations_per_step'],
                    'avg_violations_per_episode': row['avg_violations_per_episode'],
                    'avg_modifications_per_episode': row['avg_modifications_per_episode'],
                    'total_violations': row['total_violations'],
                    'total_modifications': row['total_modifications'],
                    'total_steps': row['total_steps'],
                    'train_episodes': row['train_episodes'],
                    'best_avg_reward': row['best_avg_reward'],
                    'final_reward': row['final_reward'],
                    'seed': row['seed'],
                    'method': row['method'],
                    'env': row['env'],
                }
                method_results.append(result)
            
            # Re-aggregate with median calculation
            aggregated = aggregate_results(method_results)
            if aggregated:
                os.makedirs(method_dir, exist_ok=True)
                aggregated_path = os.path.join(method_dir, 'aggregated_results.json')
                with open(aggregated_path, 'w') as f:
                    json.dump(aggregated, f, indent=2)
                print(f"✓ Regenerated {method['display_name']} on {env_name} (median: {aggregated.get('avg_reward', {}).get('median', 'N/A'):.2f})")


if __name__ == "__main__":
    base_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/ijcai_experiments'
    regenerate_from_csv(base_dir)
    print("\n✓ Done! Summary tables will now show medians.")

