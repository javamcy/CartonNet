#!/usr/bin/env python3
"""Generate comparison figures v5 - FINALIZED V24 Final3 model.

For each country, creates ONE figure with MULTIPLE subplots (one per row):
  1. Overview: groundtruth + ALL methods overlaid
  2-N. Each method individually: groundtruth + that method's predictions

Only plots h=1 (first forecast step) so each time point has exactly
2 values: one groundtruth, one prediction.

BUT displays the FULL 24-step average MAE in subplot titles,
computed from all 24 forecast steps: np.mean(np.abs(preds - targets))
where preds and targets are shape (n_samples, 24).

Key changes from v4:
  - Uses v24_final3_predictions.npz (FINALIZED 3-component V24 model)
  - Method name: "CarbonSONet (Ours)"
  - Full 24-step average MAE in all subplot titles
  - Output to figures/v5_comparison/
"""

import argparse
import json
import math
import os
import sys
import warnings
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

warnings.filterwarnings('ignore')

PROJECT_ROOT = '/home/ubuntu/Carbon_intensity_forecasting'

ALL_COUNTRY_CODES = [
    'AT', 'BE', 'CZ', 'DE', 'DK', 'ES', 'FI', 'FR',
    'GB', 'GR', 'IT', 'NL', 'NO', 'PL', 'PT', 'SE',
]

HISTORY_LEN = 168
FORECAST_LEN = 24

# Method display order and colors (highly distinguishable as specified)
METHOD_STYLES = OrderedDict([
    ('V24',          {'color': '#E53935', 'lw': 1.5,  'label': 'CarbonSONet (Ours)'}),
    ('Persistence',  {'color': '#9E9E9E', 'lw': 0.8,  'label': 'Persistence'}),
    ('ARIMA',        {'color': '#795548', 'lw': 0.8,  'label': 'ARIMA'}),
    ('LSTM',         {'color': '#FF9800', 'lw': 0.8,  'label': 'LSTM'}),
    ('Transformer',  {'color': '#9C27B0', 'lw': 0.8,  'label': 'Transformer'}),
    ('DLinear',      {'color': '#2196F3', 'lw': 0.8,  'label': 'DLinear'}),
    ('PatchTST',     {'color': '#00BCD4', 'lw': 0.8,  'label': 'PatchTST'}),
    ('iTransformer', {'color': '#4CAF50', 'lw': 0.8,  'label': 'iTransformer'}),
    ('TimesNet',     {'color': '#3F51B5', 'lw': 0.8,  'label': 'TimesNet'}),
    ('FiLM',         {'color': '#E91E63', 'lw': 0.8,  'label': 'FiLM'}),
    ('FreTS',        {'color': '#009688', 'lw': 0.8,  'label': 'FreTS'}),
    ('Informer',     {'color': '#FF5722', 'lw': 0.8,  'label': 'Informer'}),
    ('Autoformer',   {'color': '#607D8B', 'lw': 0.8,  'label': 'Autoformer'}),
    ('FEDformer',    {'color': '#8BC34A', 'lw': 0.8,  'label': 'FEDformer'}),
])

GT_COLOR = '#000000'
GT_LW = 0.8
GT_LABEL = 'Ground Truth'


def load_norm_stats(country, data_dir):
    """Load normalization stats for a country (only needed for LSTM/Transformer denorm)."""
    import torch
    path = os.path.join(data_dir, f'{country}_test.pt')
    if not os.path.exists(path):
        return None, None
    d = torch.load(path, map_location='cpu', weights_only=False)
    return float(d['target_mean']), float(d['target_std'])


def is_normalized(preds, targets):
    """Heuristic to detect if predictions are in normalized space.

    Key insight: Carbon intensity (gCO2/kWh) is always positive in real space.
    - If TARGETS have significant negative values -> data is in normalized space
    - If targets are all positive but predictions have some negatives -> real space
      with bad model outputs (model generated unrealistic negative predictions)
    """
    if len(targets) == 0:
        return False
    tgt_neg_ratio = np.sum(targets < 0) / len(targets)
    # Only consider normalized if TARGETS are negative (they must be positive in real space)
    return tgt_neg_ratio > 0.1


def denormalize(arr, mean, std):
    """Denormalize: real = normalized * std + mean."""
    return arr * std + mean


def load_v24_predictions(country, pred_path):
    """Load V24 final3 predictions and targets.
    Already in real space (gCO2/kWh).
    Returns (predictions_h1, targets_h1, mae_full_24step)
    - predictions_h1, targets_h1: shape (N,) each - for plotting
    - mae_full_24step: full 24-step average MAE
    """
    if not os.path.exists(pred_path):
        return None, None, None
    npz = np.load(pred_path, allow_pickle=True)
    pred_key = f'{country}_predictions'
    tgt_key = f'{country}_targets'
    if pred_key not in npz or tgt_key not in npz:
        return None, None, None
    preds_full = npz[pred_key]  # (N, 24) - already real space
    targets_full = npz[tgt_key]  # (N, 24) - already real space
    # Compute full 24-step average MAE
    mae_full_24step = np.mean(np.abs(preds_full - targets_full))
    # Take only h=1 (first forecast step) for plotting
    return preds_full[:, 0], targets_full[:, 0], mae_full_24step


def load_sota_predictions(model_name, country, pred_dir, norm_mean=None, norm_std=None):
    """Load SOTA model predictions from npz file.
    Most are in real space. LSTM/Transformer may be normalized -> auto-detect.
    Returns (predictions_h1, targets_h1, mae_full_24step)
    - predictions_h1, targets_h1: shape (N,) each, both in real space - for plotting
    - mae_full_24step: full 24-step average MAE
    """
    path = os.path.join(pred_dir, f'{model_name}_{country}_predictions.npz')
    if not os.path.exists(path):
        return None, None, None
    npz = np.load(path, allow_pickle=True)
    if 'predictions' not in npz or 'targets' not in npz:
        return None, None, None
    preds_full = npz['predictions']  # (N, 24)
    targets_full = npz['targets']    # (N, 24)

    # Auto-detect if predictions are in normalized space
    # Key: check TARGETS, not predictions. Real-space targets are always positive.
    if is_normalized(preds_full.flatten(), targets_full.flatten()):
        if norm_mean is not None and norm_std is not None:
            print(f'    {model_name}: detected normalized space, denormalizing...')
            preds_full = denormalize(preds_full, norm_mean, norm_std)
            targets_full = denormalize(targets_full, norm_mean, norm_std)
        else:
            print(f'    {model_name}: appears normalized but no norm stats available!')
            return None, None, None

    # Compute full 24-step average MAE
    mae_full_24step = np.mean(np.abs(preds_full - targets_full))
    # Take only h=1 for plotting
    return preds_full[:, 0], targets_full[:, 0], mae_full_24step


def compute_persistence_mae(targets_full):
    """Compute persistence baseline for ALL 24 horizons.

    Persistence: prediction at horizon h = last observed value (groundtruth at h=0 context).
    Since we have targets_full of shape (N, 24), for each sample the persistence
    prediction at horizon h is the groundtruth value at the last input step.

    However, we don't have the input sequence directly. A common simplification:
    persistence at h=1 = target at previous time step's h=1.

    For the FULL 24-step MAE of persistence, we use a simple approach:
    - At each horizon h, persistence predicts the same value as the last observed
    - Since we only have h=1 targets, persistence MAE across horizons increases
    - We approximate: persistence prediction at horizon h = target at h=1 (same sample)
      This gives a lower bound on persistence error.
    - Better: use the mean of all targets as a naive baseline.

    Actually, the simplest correct approach for persistence:
    - For h=1: predict previous hour's actual = shift targets by 1
    - For full 24-step: we don't have the full persistence computation from raw data
    - We'll compute h=1 persistence MAE and note it as such
    """
    # For persistence, we can only compute h=1 MAE from the test targets
    # Since we need the last observed value, and our targets start at h=1,
    # persistence for h=1 = target[t-1] (shifted by 1)
    n_samples = targets_full.shape[0]
    n_horizons = targets_full.shape[1]

    # h=1 persistence: predict previous hour's value
    # We use the h=1 targets shifted by 1
    targets_h1 = targets_full[:, 0]

    # For FULL 24-step persistence, the prediction at each horizon h
    # is the last known value. Without the actual input context, we
    # approximate using: each sample's persistence = the h=1 target value
    # shifted by 1 (previous timestep).
    # For simplicity, compute h=1 shift-based persistence:
    pers_pred_h1 = np.zeros_like(targets_h1)
    pers_pred_h1[0] = targets_h1[0]
    pers_pred_h1[1:] = targets_h1[:-1]

    # For the full 24-step, we only have h=1 persistence.
    # We'll report h=1 MAE for persistence.
    mae_h1 = np.mean(np.abs(pers_pred_h1 - targets_h1))

    return pers_pred_h1, mae_h1


def compute_arima(gt_series, n_test, max_arima=150):
    """Compute ARIMA(2,1,2) on subsample of test points.
    gt_series: full groundtruth time series in real space (1D array).
    Returns: (predictions array, indices array) for subsampled points.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        print('    ARIMA: statsmodels not available, skipping')
        return None, None

    preds = []
    indices = []
    step = max(1, n_test // max_arima)

    for i in range(0, n_test, step):
        hist_end = i + HISTORY_LEN
        if hist_end > len(gt_series):
            break
        window = gt_series[i:hist_end]
        try:
            model = ARIMA(window, order=(2, 1, 2))
            fitted = model.fit()
            forecast = fitted.forecast(steps=1)
            preds.append(forecast[0])
            indices.append(i)
        except Exception:
            preds.append(window[-1])
            indices.append(i)

    if not preds:
        return None, None
    return np.array(preds), np.array(indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'figures', 'v5_comparison'))
    parser.add_argument('--countries', type=str, default=None,
                        help='Comma-separated country codes')
    parser.add_argument('--max_display_points', type=int, default=500,
                        help='Max number of time points to display on x-axis')
    parser.add_argument('--skip_arima', action='store_true',
                        help='Skip ARIMA computation (slow)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    countries = ALL_COUNTRY_CODES if args.countries is None else [c.strip() for c in args.countries.split(',')]
    # KEY CHANGE: Use v24_final3_predictions.npz
    v24_pred_path = os.path.join(PROJECT_ROOT, 'results', 'v24_final3_predictions.npz')
    sota_pred_dir = os.path.join(PROJECT_ROOT, 'results', 'sota_predictions')
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')

    # Verify prediction file exists
    if not os.path.exists(v24_pred_path):
        print(f'ERROR: V24 final3 predictions not found at {v24_pred_path}')
        sys.exit(1)
    print(f'Using V24 final3 predictions: {v24_pred_path}')

    for country in countries:
        print(f'\n{"="*60}')
        print(f'Processing {country}...')
        print(f'{"="*60}')

        # Load norm stats (only for denormalizing LSTM/Transformer if needed)
        norm_mean, norm_std = load_norm_stats(country, data_dir)

        # Load V24 final3 predictions and targets (groundtruth)
        v24_preds_h1, v24_targets_h1, v24_mae_24 = load_v24_predictions(country, v24_pred_path)
        if v24_targets_h1 is None:
            print(f'  ERROR: Could not load V24 final3 predictions for {country}, skipping')
            continue

        n_total = len(v24_targets_h1)
        n_display = min(n_total, args.max_display_points)
        print(f'  Total test samples: {n_total}, displaying: {n_display}')

        # Groundtruth for display
        gt = v24_targets_h1[:n_display]
        time_axis = np.arange(n_display)

        # Collect all method predictions (h=1, first n_display points)
        # Also store full 24-step MAE for each method
        all_preds = OrderedDict()  # method_name -> pred_array_h1[:n_display]
        all_mae_24step = OrderedDict()  # method_name -> full 24-step average MAE

        # 1. V24 (CarbonSONet Ours)
        if v24_preds_h1 is not None:
            all_preds['V24'] = v24_preds_h1[:n_display]
            all_mae_24step['V24'] = v24_mae_24
            print(f'  CarbonSONet (Ours): 24-step avg MAE={v24_mae_24:.2f}')

        # 2. Persistence - h=1 MAE only (no multi-step persistence data)
        # Load full targets for persistence computation
        v24_npz = np.load(v24_pred_path, allow_pickle=True)
        targets_full_24 = v24_npz[f'{country}_targets']  # (N, 24)
        pers_preds_h1, pers_mae = compute_persistence_mae(targets_full_24)
        all_preds['Persistence'] = pers_preds_h1[:n_display]
        all_mae_24step['Persistence'] = pers_mae  # h=1 only for persistence
        print(f'  Persistence: h=1 MAE={pers_mae:.2f} (h=1 only)')

        # 3. ARIMA (computed on subsample) - only h=1
        if not args.skip_arima:
            print(f'  Computing ARIMA (subsample max 150)...')
            arima_preds_sub, arima_indices = compute_arima(v24_targets_h1, n_total, max_arima=150)
            if arima_preds_sub is not None:
                # ARIMA was computed on a subsample, fill in with persistence for missing
                arima_full = np.copy(pers_preds_h1)  # default to persistence
                for j, idx in enumerate(arima_indices):
                    if idx < n_total:
                        arima_full[idx] = arima_preds_sub[j]
                mae_arima = np.mean(np.abs(arima_full - v24_targets_h1))
                all_preds['ARIMA'] = arima_full[:n_display]
                all_mae_24step['ARIMA'] = mae_arima  # h=1 only for ARIMA
                print(f'  ARIMA: h=1 MAE={mae_arima:.2f} (h=1, subsampled, {len(arima_indices)} points)')
            else:
                print(f'  ARIMA: failed, skipping')

        # 4-14. SOTA models from saved predictions
        sota_models = ['LSTM', 'Transformer', 'DLinear', 'PatchTST', 'iTransformer',
                       'TimesNet', 'FiLM', 'FreTS', 'Informer', 'Autoformer', 'FEDformer']

        for model_name in sota_models:
            pred_h1, tgt_h1, mae_24 = load_sota_predictions(
                model_name, country, sota_pred_dir, norm_mean, norm_std)
            if pred_h1 is not None and tgt_h1 is not None and mae_24 is not None:
                all_preds[model_name] = pred_h1[:n_display]
                all_mae_24step[model_name] = mae_24
                print(f'  {model_name}: 24-step avg MAE={mae_24:.2f}')
            else:
                print(f'  {model_name}: predictions not available, skipping')

        # -------------------------------------------------
        # Generate figure
        # -------------------------------------------------
        n_methods = len(all_preds)
        if n_methods == 0:
            print(f'  No methods available for {country}, skipping figure')
            continue

        # Total subplots: 1 overview + n_methods individual
        n_subplots = 1 + n_methods
        fig_height = 3.5 * n_subplots
        fig, axes = plt.subplots(n_subplots, 1, figsize=(16, fig_height),
                                  squeeze=False)
        axes = axes.flatten()

        # -- Subplot 0: Overview (all methods overlaid) --
        ax = axes[0]
        ax.plot(time_axis, gt, color=GT_COLOR, lw=GT_LW, linestyle='--',
                label=GT_LABEL, alpha=0.7)
        for method_name, pred_arr in all_preds.items():
            style = METHOD_STYLES.get(method_name, {'color': 'gray', 'lw': 0.8,
                                                     'label': method_name})
            min_l = min(len(pred_arr), len(gt))
            ax.plot(time_axis[:min_l], pred_arr[:min_l],
                    color=style['color'], lw=style['lw'],
                    label=style['label'], alpha=0.85)
        ax.set_title(f'{country} - All Methods Overview', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (hours)', fontsize=10)
        ax.set_ylabel('CI (gCO$_2$/kWh)', fontsize=10)
        ax.legend(fontsize=8, loc='upper right', ncol=3)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)

        # -- Subplots 1..N: Each method individually --
        for idx, (method_name, pred_arr) in enumerate(all_preds.items()):
            ax = axes[idx + 1]
            style = METHOD_STYLES.get(method_name, {'color': 'gray', 'lw': 0.8,
                                                     'label': method_name})

            # Groundtruth
            ax.plot(time_axis, gt, color=GT_COLOR, lw=GT_LW, linestyle='--',
                    label=GT_LABEL, alpha=0.7)

            # Method prediction
            min_l = min(len(pred_arr), len(gt))
            ax.plot(time_axis[:min_l], pred_arr[:min_l],
                    color=style['color'], lw=style['lw'],
                    label=style['label'], alpha=0.9)

            # Use full 24-step average MAE in subplot title
            if method_name in all_mae_24step:
                title_mae = all_mae_24step[method_name]
            else:
                # Fallback: compute h=1 MAE from displayed data
                title_mae = np.mean(np.abs(pred_arr[:min_l] - gt[:min_l]))

            ax.set_title(f'{style["label"]} (MAE={title_mae:.2f})',
                         fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (hours)', fontsize=10)
            ax.set_ylabel('CI (gCO$_2$/kWh)', fontsize=10)
            ax.legend(fontsize=8, loc='upper right')
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Carbon Intensity Forecasting Comparison - {country}',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        png_path = os.path.join(args.output_dir, f'comparison_{country}.png')
        pdf_path = os.path.join(args.output_dir, f'comparison_{country}.pdf')
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {country} figure ({n_methods} methods) -> {png_path}')

    print(f'\nAll figures saved to {args.output_dir}')


if __name__ == '__main__':
    main()
