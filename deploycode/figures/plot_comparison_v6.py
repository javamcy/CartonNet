#!/usr/bin/env python3
"""
Generate 16-country comparison figures for CarbonSONet V24 Final3 vs SOTA baselines.
Format: Each country one figure, each figure has multiple subplots (one per row).
- First subplot: groundtruth + all methods (CarbonSONet highlighted)
- Subsequent subplots: groundtruth + one method
"""

import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============ Configuration ============
RESULTS_DIR = '/home/ubuntu/Carbon_intensity_forecasting/results'
FIGURES_DIR = '/home/ubuntu/Carbon_intensity_forecasting/figures/v6_comparison'
COUNTRIES = ['AT', 'BE', 'CZ', 'DE', 'DK', 'ES', 'FI', 'FR', 'GB', 'GR', 'IT', 'NL', 'NO', 'PL', 'PT', 'SE']

# All methods to display (CarbonSONet first, then baselines)
BASELINE_MODELS = [
    'Persistence', 'iTransformer', 'DLinear', 'FreTS', 'TimesNet',
    'MICN', 'PatchTST', 'FiLM', 'LSTM', 'Transformer',
    'Informer', 'FEDformer', 'Autoformer'
]

# Color scheme - CarbonSONet in red, others in distinct colors
COLORS = {
    'CarbonSONet': '#E63946',      # Bold red
    'Persistence': '#457B9D',      # Steel blue
    'iTransformer': '#2A9D8F',     # Teal
    'DLinear': '#E9C46A',          # Yellow
    'FreTS': '#F4A261',            # Orange
    'TimesNet': '#264653',         # Dark blue
    'MICN': '#8338EC',             # Purple
    'PatchTST': '#FF006E',         # Pink
    'FiLM': '#3A86FF',             # Blue
    'LSTM': '#06D6A0',             # Green
    'Transformer': '#118AB2',      # Cyan
    'Informer': '#073B4C',         # Dark teal
    'FEDformer': '#FB5607',        # Bright orange
    'Autoformer': '#8AC926',       # Lime green
}

GROUNDTRUTH_COLOR = '#1D3557'  # Dark navy for groundtruth

# Figure settings
SAMPLE_STEP = 24       # Show every 24th point (daily) for clarity
MAX_POINTS = 200       # Maximum number of points to show
FIG_DPI = 150
SUBPLOT_HEIGHT = 2.0   # Height per subplot in inches
FIG_WIDTH = 14         # Width in inches


def load_carbonsonet_predictions():
    """Load CarbonSONet V24 Final3 predictions."""
    path = os.path.join(RESULTS_DIR, 'v24_final3_predictions.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f"CarbonSONet predictions not found: {path}")
    data = np.load(path, allow_pickle=True)
    predictions = {}
    for country in COUNTRIES:
        pred_key = f'{country}_predictions'
        tgt_key = f'{country}_targets'
        if pred_key in data and tgt_key in data:
            predictions[country] = {
                'predictions': data[pred_key],
                'targets': data[tgt_key]
            }
    return predictions


def load_sota_predictions():
    """Load SOTA baseline predictions."""
    sota_dir = os.path.join(RESULTS_DIR, 'sota_predictions')
    predictions = {}
    for model in BASELINE_MODELS:
        if model == 'Persistence':
            continue  # Computed separately
        predictions[model] = {}
        for country in COUNTRIES:
            path = os.path.join(sota_dir, f'{model}_{country}_predictions.npz')
            if os.path.exists(path):
                data = np.load(path, allow_pickle=True)
                predictions[model][country] = {
                    'predictions': data['predictions'],
                    'targets': data['targets']
                }
    return predictions


def compute_persistence_predictions(cs_preds):
    """Compute Persistence baseline: predict last input value for all horizons.
    Since we don't have raw input data, we use the last known target value
    as the persistence prediction (which is the standard approach).
    """
    persistence = {}
    for country in COUNTRIES:
        if country in cs_preds:
            targets = cs_preds[country]['targets']
            # Persistence: use the value at t-1 (last column of shifted targets)
            # Actually, for carbon intensity, persistence = last observed value
            # We approximate by using the first target value repeated
            # More accurately: shift targets by 1 step
            preds = np.roll(targets, 1, axis=0)
            preds[0] = targets[0]  # First row has no previous
            persistence[country] = {
                'predictions': preds,
                'targets': targets
            }
    return persistence


def compute_mae(pred, target):
    """Compute MAE between predictions and targets."""
    return np.mean(np.abs(pred - target))


def generate_country_figure(country, cs_preds, sota_preds, persistence_preds):
    """Generate comparison figure for a single country."""
    # Get groundtruth
    targets = cs_preds[country]['targets']
    cs_pred = cs_preds[country]['predictions']

    # Select a subset of time points for visualization
    n_total = targets.shape[0]
    # Use the first 24-hour forecast horizon (h=0)
    gt = targets[:, 0]  # 1-step ahead forecast
    cs = cs_pred[:, 0]

    # Sample for clarity
    step = max(1, n_total // MAX_POINTS)
    indices = np.arange(0, n_total, step)
    gt = gt[indices]
    cs = cs[indices]
    x = np.arange(len(gt))

    # All methods for this country
    all_methods = {'CarbonSONet': cs}
    for model in BASELINE_MODELS:
        if model == 'Persistence':
            if country in persistence_preds:
                all_methods['Persistence'] = persistence_preds[country]['predictions'][indices, 0]
        elif model in sota_preds and country in sota_preds[model]:
            all_methods[model] = sota_preds[model][country]['predictions'][indices, 0]

    # Compute MAE for display
    cs_mae = compute_mae(cs_preds[country]['predictions'], cs_preds[country]['targets'])

    # Create figure with subplots
    n_subplots = 1 + len(all_methods) - 1  # 1 overview + (n_methods - 1) individual
    # Limit to CarbonSONet + key baselines to keep figure manageable
    key_baselines = ['Persistence', 'iTransformer', 'DLinear', 'FreTS', 'TimesNet',
                     'LSTM', 'Transformer', 'PatchTST']
    display_methods = [m for m in key_baselines if m in all_methods]

    n_subplots = 1 + len(display_methods)  # Overview + individual
    fig_height = SUBPLOT_HEIGHT * n_subplots
    fig, axes = plt.subplots(n_subplots, 1, figsize=(FIG_WIDTH, fig_height),
                             sharex=True)
    if n_subplots == 1:
        axes = [axes]

    fig.suptitle(f'Carbon Intensity Forecasting - {country} (CarbonSONet avg MAE={cs_mae:.1f})',
                 fontsize=14, fontweight='bold', y=0.98)

    # ---- Subplot 1: Overview with all methods ----
    ax = axes[0]
    ax.plot(x, gt, color=GROUNDTRUTH_COLOR, linewidth=1.5, alpha=0.8, label='Ground Truth')
    ax.plot(x, cs, color=COLORS['CarbonSONet'], linewidth=1.8, alpha=0.9, label='CarbonSONet')

    # Plot other methods with thinner lines
    for model in display_methods:
        if model in all_methods:
            ax.plot(x, all_methods[model], color=COLORS.get(model, '#999999'),
                    linewidth=0.8, alpha=0.6, label=model)

    ax.set_ylabel('gCO$_2$/kWh', fontsize=9)
    ax.set_title('All Methods Overview', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, ncol=4, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(gt))

    # ---- Subplots 2+: Individual method comparison ----
    for idx, model in enumerate(display_methods):
        ax = axes[idx + 1]
        ax.plot(x, gt, color=GROUNDTRUTH_COLOR, linewidth=1.5, alpha=0.8,
                label='Ground Truth')
        if model in all_methods:
            ax.plot(x, all_methods[model], color=COLORS.get(model, '#999999'),
                    linewidth=1.2, alpha=0.9, label=model)

        # Compute MAE for this method
        if model == 'Persistence' and country in persistence_preds:
            model_mae = compute_mae(persistence_preds[country]['predictions'],
                                    persistence_preds[country]['targets'])
        elif model in sota_preds and country in sota_preds[model]:
            model_mae = compute_mae(sota_preds[model][country]['predictions'],
                                    sota_preds[model][country]['targets'])
        else:
            model_mae = float('nan')

        ax.set_ylabel('gCO$_2$/kWh', fontsize=9)
        ax.set_title(f'{model} (MAE={model_mae:.1f}) vs CarbonSONet (MAE={cs_mae:.1f})',
                     fontsize=9, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(gt))

    axes[-1].set_xlabel('Time (hours)', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    os.makedirs(FIGURES_DIR, exist_ok=True)
    pdf_path = os.path.join(FIGURES_DIR, f'comparison_{country}.pdf')
    png_path = os.path.join(FIGURES_DIR, f'comparison_{country}.png')
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    return cs_mae


def main():
    print("Loading CarbonSONet V24 Final3 predictions...")
    cs_preds = load_carbonsonet_predictions()
    print(f"  Loaded {len(cs_preds)} countries")

    print("Loading SOTA baseline predictions...")
    sota_preds = load_sota_predictions()
    print(f"  Loaded {len(sota_preds)} models")

    print("Computing Persistence baseline...")
    persistence_preds = compute_persistence_predictions(cs_preds)
    print(f"  Computed for {len(persistence_preds)} countries")

    # Generate figures
    print(f"\nGenerating figures in {FIGURES_DIR}...")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    mae_summary = {}
    for country in COUNTRIES:
        print(f"  {country}...", end=' ', flush=True)
        cs_mae = generate_country_figure(country, cs_preds, sota_preds, persistence_preds)
        mae_summary[country] = cs_mae
        print(f"MAE={cs_mae:.2f}")

    avg_mae = np.mean(list(mae_summary.values()))
    print(f"\n  Average MAE: {avg_mae:.2f}")

    # Save MAE summary
    summary_path = os.path.join(FIGURES_DIR, 'mae_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'carbonsonet_v24_final3': {k: float(v) for k, v in mae_summary.items()},
            'avg_mae': round(float(avg_mae), 2)
        }, f, indent=2)

    print(f"\nDone! Figures saved to {FIGURES_DIR}")


if __name__ == '__main__':
    main()
