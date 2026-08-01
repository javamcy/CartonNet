#!/usr/bin/env python3
"""
Multi-seed runner for CarbonSONet V24 full model.
Runs the full_model ablation 5 times with different random seeds.
Saves per-seed results and computes mean/std.
"""

import os
import sys
import json
import random
import numpy as np
import torch

PROJECT_ROOT = '/home/ubuntu/Carbon_intensity_forecasting'
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

SEEDS = [42, 123, 456, 789, 1024]
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'multiseed_v24_final3')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_full_model(seed):
    """Run full_model ablation with a specific seed."""
    set_seed(seed)
    
    # Import the ablation runner
    from experiments.run_ablation_v24_final3 import (
        run_ablation, ABLATION_MAP, ALL_COUNTRY_CODES,
        prepare_data, ABLATION_BATCH_SIZE, ABLATION_EPOCHS,
        ABLATION_PATIENCE, ABLATION_LR, ABLATION_WEIGHT_DECAY,
        LAMBDA_PHYSICS, LAMBDA_SPECTRAL, LAMBDA_ROUTER,
        LAMBDA_RATE, LAMBDA_TRANSPORT, device,
    )
    
    import logging
    logging.basicConfig(level=logging.INFO)
    
    countries = ALL_COUNTRY_CODES
    disabled = ABLATION_MAP['full_model']
    ckpt_dir_base = os.path.join(OUTPUT_DIR, f'seed_{seed}')
    
    data = prepare_data(countries, batch_size=ABLATION_BATCH_SIZE)
    
    result, predictions = run_ablation(
        'full_model', disabled, countries, data,
        OUTPUT_DIR, ckpt_dir_base,
        save_predictions=True
    )
    
    result['seed'] = seed
    return result

def main():
    all_results = {}
    all_avg_maes = []
    
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Running seed={seed}")
        print(f"{'='*60}")
        
        result = run_full_model(seed)
        all_results[f'seed_{seed}'] = result
        all_avg_maes.append(result['avg_test_mae'])
        
        print(f"Seed {seed}: avg_test_mae={result['avg_test_mae']:.2f}")
        
        # Save incrementally
        with open(os.path.join(OUTPUT_DIR, 'multiseed_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
    
    # Compute summary
    mean_mae = np.mean(all_avg_maes)
    std_mae = np.std(all_avg_maes)
    
    summary = {
        'seeds': SEEDS,
        'per_seed_avg_mae': all_avg_maes,
        'mean_avg_mae': round(float(mean_mae), 2),
        'std_avg_mae': round(float(std_mae), 2),
        'best_single_run': round(float(min(all_avg_maes)), 2),
    }
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Per-seed avg MAE: {all_avg_maes}")
    print(f"  Mean: {mean_mae:.2f}")
    print(f"  Std: {std_mae:.2f}")
    print(f"  Best single run: {min(all_avg_maes):.2f}")
    print(f"{'='*60}")
    
    # Save summary
    with open(os.path.join(OUTPUT_DIR, 'multiseed_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Also save all results
    with open(os.path.join(OUTPUT_DIR, 'multiseed_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

if __name__ == '__main__':
    main()