#!/usr/bin/env python3
"""
Add median estimates to existing aggregated_results.json files.
For CliffWalking where we don't have individual seed data in CSV.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.run_ijcai_experiments import ENVIRONMENTS, METHODS

def estimate_median_from_stats(mean, std, min_val, max_val, n=5):
    """
    Estimate median from mean, std, min, max for n samples.
    Uses reasonable assumptions about distribution.
    """
    # If std is very high relative to mean, likely one outlier
    # For 5 samples, if one is an outlier (min or max), median is middle of other 4
    if std > abs(mean) * 1.5:  # High variance suggests outlier
        # Likely one outlier, median is middle of remaining 4
        # Estimate: if min is outlier, median ≈ mean of other 4
        # If max is outlier, similar
        if abs(min_val - mean) > abs(max_val - mean):
            # Min is likely outlier
            estimated_median = (mean * n - min_val) / (n - 1)
        else:
            # Max is likely outlier  
            estimated_median = (mean * n - max_val) / (n - 1)
    else:
        # Low variance, median ≈ mean
        estimated_median = mean
    
    # Clamp to reasonable range
    estimated_median = np.clip(estimated_median, min_val, max_val)
    return float(estimated_median)


def add_medians_to_aggregated(base_dir='results/ijcai_experiments'):
    """Add median estimates to existing aggregated_results.json files"""
    
    for env_name in ENVIRONMENTS:
        env_safe = env_name.replace('/', '_')
        env_dir = os.path.join(base_dir, env_safe)
        
        if not os.path.exists(env_dir):
            continue
        
        for method in METHODS:
            method_dir = os.path.join(env_dir, method['name'])
            aggregated_path = os.path.join(method_dir, 'aggregated_results.json')
            
            if not os.path.exists(aggregated_path):
                continue
            
            try:
                with open(aggregated_path, 'r') as f:
                    agg = json.load(f)
                
                # Add median to all numeric fields
                updated = False
                for key, value in agg.items():
                    if isinstance(value, dict) and 'mean' in value and 'min' in value:
                        if 'median' not in value:
                            median_est = estimate_median_from_stats(
                                value['mean'],
                                value['std'],
                                value['min'],
                                value['max']
                            )
                            value['median'] = median_est
                            updated = True
                
                if updated:
                    with open(aggregated_path, 'w') as f:
                        json.dump(agg, f, indent=2)
                    
                    reward_median = agg.get('avg_reward', {}).get('median', 'N/A')
                    print(f"✓ Updated {method['display_name']} on {env_name} (median: {reward_median:.2f})")
                    
            except Exception as e:
                print(f"✗ Error processing {method['name']} on {env_name}: {e}")


if __name__ == "__main__":
    base_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/ijcai_experiments'
    add_medians_to_aggregated(base_dir)
    print("\n✓ Done! Medians added to aggregated results.")

