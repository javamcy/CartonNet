#!/usr/bin/env python3
"""
Generate Figure 3: Prediction examples across four European countries.

Creates a 7-row x 4-column figure showing carbon intensity predictions
for GB, DK, DE, and PL. Each subplot shows the first 400 test-set points.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# ============================================================================
# Configuration
# ============================================================================
COUNTRIES = ['GB', 'DK', 'DE', 'PL']
COUNTRY_NAMES = {
    'GB': 'United Kingdom',
    'DK': 'Denmark',
    'DE': 'Germany',
    'PL': 'Poland',
}

# All 12 baseline methods
BASELINE_METHODS = [
    'iTransformer', 'TimesNet', 'FreTS', 'MICN', 'DLinear',
    'PatchTST', 'FiLM', 'LSTM', 'Transformer', 'Informer',
    'FEDformer', 'Autoformer',
]

# Methods whose predictions are in NORMALIZED space and need denormalization
NORMALIZED_METHODS = {'LSTM', 'Transformer'}

# Color scheme (13 distinct colors)
COLORS = {
    'groundtruth': '#000000',   # Black
    'Ours':         '#E31A1C',   # Red
    'iTransformer': '#1F78B4',   # Blue
    'TimesNet':     '#33A02C',   # Green
    'FreTS':        '#6A3D9A',   # Purple
    'MICN':         '#FF7F00',   # Orange
    'DLinear':      '#8B4513',   # Brown
    'PatchTST':     '#FB9A99',   # Pink
    'FiLM':         '#00CED1',   # Cyan
    'LSTM':         '#006400',   # Dark Green
    'Transformer':  '#00008B',   # Dark Blue
    'Informer':     '#8B0000',   # Dark Red
    'FEDformer':    '#808080',   # Gray
    'Autoformer':   '#808000',   # Olive
}

# File paths
OURS_PATH = '/home/ubuntu/Carbon_intensity_forecasting/results/v24_final3_predictions.npz'
SOTA_DIR = '/home/ubuntu/Carbon_intensity_forecasting/results/sota_predictions'
DATA_DIR = '/home/ubuntu/Carbon_intensity_forecasting/data/processed'
OUTPUT_DIR = '/home/ubuntu/Carbon_intensity_forecasting/figures_short'

# Figure settings
FIG_WIDTH = 20
FIG_HEIGHT = 28
DPI = 300
SCATTER_SIZE = 0.3
SCATTER_ALPHA = 0.35
GT_LINEWIDTH = 0.4
GT_ALPHA = 0.7

# Number of points to plot per method (downsample for clarity)
N_PLOT_POINTS = 400


def subsample(x_data, y_data, n_points=N_PLOT_POINTS):
    """Take the first n_points from the test set (no interval sampling)."""
    return x_data[:n_points], y_data[:n_points]


def flatten_predictions_fast(pred_array):
    """
    Convert sliding window predictions (N_windows, 24) to a continuous time series.

    For overlapping windows, prefer the shortest-horizon prediction (most accurate).
    Window i's step s corresponds to time position i+s.
    """
    n_windows, forecast_len = pred_array.shape
    total_len = n_windows + forecast_len - 1

    result = np.full(total_len, np.nan, dtype=np.float32)

    for step in range(forecast_len):
        start = step
        end = min(step + n_windows, total_len)
        actual_len = end - start
        mask = np.isnan(result[start:end])
        result[start:end][mask] = pred_array[:actual_len, step][mask]

    return result


def compute_mae_all_steps(pred_array, target_array):
    """Compute MAE over all prediction steps (same as paper metric)."""
    return np.mean(np.abs(pred_array - target_array))


def compute_mae_on_first_n(flat_pred, flat_gt, n_points=400):
    """Compute MAE on the first n_points of flattened predictions (for figure)."""
    valid_mask = ~np.isnan(flat_pred[:n_points]) & ~np.isnan(flat_gt[:n_points])
    if valid_mask.sum() == 0:
        return float('nan')
    return np.mean(np.abs(flat_pred[:n_points][valid_mask] - flat_gt[:n_points][valid_mask]))


def load_country_stats():
    """Load target_mean and target_std for each country from test.pt files."""
    stats = {}
    for country in COUNTRIES:
        path = os.path.join(DATA_DIR, f'{country}_test.pt')
        d = torch.load(path, map_location='cpu', weights_only=False)
        stats[country] = {
            'target_mean': float(d['target_mean']),
            'target_std': float(d['target_std']),
        }
    return stats


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load country normalization stats
    print("Loading country stats...")
    country_stats = load_country_stats()
    for c in COUNTRIES:
        print(f"  {c}: mean={country_stats[c]['target_mean']:.2f}, std={country_stats[c]['target_std']:.2f}")

    # ========================================================================
    # Load our method's predictions (already in real space)
    # ========================================================================
    print("\nLoading our method's predictions...")
    ours_data = np.load(OURS_PATH)

    # ========================================================================
    # Load and denormalize baseline predictions
    # ========================================================================
    print("Loading baseline predictions...")
    baseline_data = {}  # {method: {country: {'predictions': ..., 'targets': ...}}}
    for method in BASELINE_METHODS:
        baseline_data[method] = {}
        for country in COUNTRIES:
            path = os.path.join(SOTA_DIR, f'{method}_{country}_predictions.npz')
            if os.path.exists(path):
                d = np.load(path)
                pred = d['predictions']
                tgt = d['targets']

                # Denormalize if needed
                if method in NORMALIZED_METHODS:
                    tmean = country_stats[country]['target_mean']
                    tstd = country_stats[country]['target_std']
                    pred = pred * tstd + tmean
                    tgt = tgt * tstd + tmean

                baseline_data[method][country] = {
                    'predictions': pred,
                    'targets': tgt,
                }
            else:
                print(f"  WARNING: {path} not found!")

    # ========================================================================
    # Compute average MAE for each baseline method across 4 countries
    # (Pre-flatten: used for ranking only, not final figure MAE)
    # ========================================================================
    print("\nComputing MAE for ranking...")
    method_avg_mae = {}
    method_country_mae = {}
    for method in BASELINE_METHODS:
        maes = []
        method_country_mae[method] = {}
        for country in COUNTRIES:
            if country in baseline_data.get(method, {}):
                pred = baseline_data[method][country]['predictions']
                tgt = baseline_data[method][country]['targets']
                mae = compute_mae_all_steps(pred, tgt)
                maes.append(mae)
                method_country_mae[method][country] = mae
        method_avg_mae[method] = np.mean(maes) if maes else float('inf')

    # Rank baselines by average MAE (lower is better)
    ranked_baselines = sorted(BASELINE_METHODS, key=lambda m: method_avg_mae[m])
    top5_baselines = ranked_baselines[:5]

    print("\nBaseline ranking (by avg MAE across 4 countries, real space):")
    for i, m in enumerate(ranked_baselines):
        print(f"  {i+1}. {m}: {method_avg_mae[m]:.2f}")
    print(f"\nTop 5 baselines for subplot: {top5_baselines}")



    # ========================================================================
    # Flatten predictions for plotting
    # ========================================================================
    print("\nFlattening predictions for plotting...")

    flat_data = {}  # {country: {'groundtruth': array, 'Ours': array, method: array, ...}}

    for country in COUNTRIES:
        print(f"  Processing {country}...")
        flat_data[country] = {}

        # Our method's target as ground truth
        our_tgt = ours_data[f'{country}_targets']
        flat_gt = flatten_predictions_fast(our_tgt)
        flat_data[country]['groundtruth'] = flat_gt

        # Our method's predictions
        our_pred = ours_data[f'{country}_predictions']
        flat_data[country]['Ours'] = flatten_predictions_fast(our_pred)

        # Baseline predictions
        for method in BASELINE_METHODS:
            if country in baseline_data.get(method, {}):
                bl_pred = baseline_data[method][country]['predictions']
                flat_data[country][method] = flatten_predictions_fast(bl_pred)

        print(f"    Flattened length: {len(flat_gt)}")

    # Compute our method's MAE per country (on first 400 pts for figure consistency)
    ours_country_mae = {}
    for country in COUNTRIES:
        ours_country_mae[country] = compute_mae_on_first_n(
            flat_data[country]['Ours'], flat_data[country]['groundtruth'])

    print("\nOur method MAE per country (first 400 points):")
    for c in COUNTRIES:
        print(f"  {c}: {ours_country_mae[c]:.2f}")

    # Compute average MAE for each baseline method across 4 countries (first 400 points)
    method_avg_mae = {}
    method_country_mae = {}
    for method in BASELINE_METHODS:
        maes = []
        method_country_mae[method] = {}
        for country in COUNTRIES:
            if country in baseline_data.get(method, {}) and method in flat_data.get(country, {}):
                mae = compute_mae_on_first_n(
                    flat_data[country][method], flat_data[country]['groundtruth'])
                maes.append(mae)
                method_country_mae[method][country] = mae
        method_avg_mae[method] = np.mean(maes) if maes else float('inf')

    ranked_baselines = sorted(BASELINE_METHODS, key=lambda m: method_avg_mae[m])
    top5_baselines = ranked_baselines[:5]

    print("\nBaseline ranking (by avg MAE across 4 countries, first 400 points):")
    for i, m in enumerate(ranked_baselines):
        print(f"  {i+1}. {m}: {method_avg_mae[m]:.2f}")
    print(f"\nTop 5 baselines for subplot: {top5_baselines}")

    # ========================================================================
    # Create figure
    # ========================================================================
    # Recompute MAE on first 400 points for figure display consistency
    print("\nRecomputing MAE on first 400 points for figure...")
    # Our method MAE on first 400 points
    ours_country_mae_400 = {}
    for country in COUNTRIES:
        ours_country_mae_400[country] = compute_mae_on_first_n(
            flat_data[country]['Ours'], flat_data[country]['groundtruth'])
    # Baseline MAE on first 400 points
    method_country_mae_400 = {}
    for method in BASELINE_METHODS:
        method_country_mae_400[method] = {}
        for country in COUNTRIES:
            if method in flat_data.get(country, {}):
                method_country_mae_400[method][country] = compute_mae_on_first_n(
                    flat_data[country][method], flat_data[country]['groundtruth'])

    print("\nOur method MAE per country (first 400 points):")
    for c in COUNTRIES:
        print(f"  {c}: {ours_country_mae_400[c]:.2f}")

    print("\nCreating figure...")
    print(f"  Each method will be downsampled to {N_PLOT_POINTS} points per subplot.")

    fig, axes = plt.subplots(7, 4, figsize=(FIG_WIDTH, FIG_HEIGHT))
    plt.rcParams.update({'font.size': 9})

    for col, country in enumerate(COUNTRIES):
        gt = flat_data[country]['groundtruth']
        x = np.arange(len(gt))

        # ---- Subplot 1: All methods + ground truth ----
        ax = axes[0, col]
        # Ground truth: 600 points as line
        x_gt, y_gt = subsample(x, gt)
        ax.plot(x_gt, y_gt, color=COLORS['groundtruth'], linewidth=GT_LINEWIDTH,
                alpha=GT_ALPHA, zorder=10)
        # Ours: 600 points scatter
        x_sub, y_sub = subsample(x, flat_data[country]['Ours'])
        ax.scatter(x_sub, y_sub, c=COLORS['Ours'],
                   s=SCATTER_SIZE, alpha=SCATTER_ALPHA, zorder=5)
        # Baselines: 600 points scatter each
        for method in BASELINE_METHODS:
            if method in flat_data[country]:
                x_sub, y_sub = subsample(x, flat_data[country][method])
                ax.scatter(x_sub, y_sub, c=COLORS[method],
                           s=SCATTER_SIZE, alpha=SCATTER_ALPHA, zorder=3)
        ax.set_title(f'{country} — All Methods', fontsize=10, fontweight='bold')
        if col == 0:
            ax.set_ylabel('gCO₂/kWh', fontsize=9)
        ax.tick_params(labelsize=7)

        # ---- Subplot 2: Ours + ground truth ----
        ax = axes[1, col]
        x_gt, y_gt = subsample(x, gt)
        ax.plot(x_gt, y_gt, color=COLORS['groundtruth'], linewidth=GT_LINEWIDTH,
                alpha=GT_ALPHA, zorder=10)
        x_sub, y_sub = subsample(x, flat_data[country]['Ours'])
        ax.scatter(x_sub, y_sub, c=COLORS['Ours'],
                   s=SCATTER_SIZE, alpha=SCATTER_ALPHA, zorder=5)
        mae_val = ours_country_mae_400.get(country, float('nan'))
        ax.set_title(f'{country} — Ours (MAE={mae_val:.1f})', fontsize=10,
                     fontweight='bold')
        if col == 0:
            ax.set_ylabel('gCO₂/kWh', fontsize=9)
        ax.tick_params(labelsize=7)

        # ---- Subplots 3-7: Top 5 baselines + ground truth ----
        for row_idx, method in enumerate(top5_baselines):
            ax = axes[row_idx + 2, col]
            x_gt, y_gt = subsample(x, gt)
            ax.plot(x_gt, y_gt, color=COLORS['groundtruth'], linewidth=GT_LINEWIDTH,
                    alpha=GT_ALPHA, zorder=10)
            if method in flat_data[country]:
                x_sub, y_sub = subsample(x, flat_data[country][method])
                ax.scatter(x_sub, y_sub, c=COLORS[method],
                           s=SCATTER_SIZE, alpha=SCATTER_ALPHA, zorder=5)
                mae_val = method_country_mae_400.get(method, {}).get(country, float('nan'))
                ax.set_title(f'{country} — {method} (MAE={mae_val:.1f})',
                             fontsize=10, fontweight='bold')
            else:
                ax.set_title(f'{country} — {method} (N/A)', fontsize=10,
                             fontweight='bold')
            if col == 0:
                ax.set_ylabel('gCO₂/kWh', fontsize=9)
            ax.tick_params(labelsize=7)

        # X-axis label only for bottom row
        axes[6, col].set_xlabel('Test Period (hours)', fontsize=9)

    # ========================================================================
    # Add legend
    # ========================================================================
    legend_elements = [
        Line2D([0], [0], color=COLORS['groundtruth'], linewidth=2, label='Ground Truth'),
    ]
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['Ours'],
               markersize=6, label='Ours')
    )
    # Top 5 baselines first
    for method in top5_baselines:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[method],
                   markersize=6, label=method)
        )
    # Remaining baselines
    for method in ranked_baselines[5:]:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[method],
                   markersize=6, label=method)
        )

    fig.legend(handles=legend_elements, loc='lower center', ncol=7,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    # ========================================================================
    # Save
    # ========================================================================
    pdf_path = os.path.join(OUTPUT_DIR, 'fig3_prediction_examples.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'fig3_prediction_examples.png')

    print(f"\nSaving PDF to {pdf_path}...")
    fig.savefig(pdf_path, format='pdf', dpi=DPI, bbox_inches='tight')
    print(f"Saving PNG to {png_path}...")
    fig.savefig(png_path, format='png', dpi=DPI, bbox_inches='tight')

    plt.close(fig)
    print("Done!")


if __name__ == '__main__':
    main()
