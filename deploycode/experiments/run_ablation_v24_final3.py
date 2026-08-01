#!/usr/bin/env python3
"""
Ablation experiment runner for CarbonSONet.

Runs the full model and ablation variants (removing one component at a time)
across 16 European countries. Components: PICFP (physics-informed carbon flow
prior), RCPH (rate-of-change prediction head), DSF-HDR (dual-space fusion with
horizon-dependent reweighting).
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, ConcatDataset, Dataset

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.carbon_sonet_v24 import (
    CarbonSONetV24,
    LAMBDA_PHYSICS,
    LAMBDA_SPECTRAL,
    LAMBDA_ROUTER,
    LAMBDA_RATE,
    LAMBDA_TRANSPORT,
    NEIGHBOR_GRID,
    MSSD_N_PERIODS,
)

# Constants
ALL_COUNTRY_CODES = [
    'AT', 'BE', 'CZ', 'DE', 'DK', 'ES', 'FI', 'FR',
    'GB', 'GR', 'IT', 'NL', 'NO', 'PL', 'PT', 'SE',
]
CODE_TO_ID = {c: i for i, c in enumerate(ALL_COUNTRY_CODES)}

REORDER_INDICES = [9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]

HISTORY_LEN = 168
FORECAST_LEN = 24
N_NEIGHBOR_FEATURES = 3
N_TOTAL_FEATURES = 19
N_GEN_FEATURES = 9

# Base disabled set: all components NOT in the final 3 retained (PICFP, DSF-HDR, RCPH)
# PCSH added because it had negative contribution in 4-component ablation
BASE_DISABLED = {'MDIE', 'CCSR', 'MSSD', 'HAPF', 'DSA', 'FiLM', 'SATC', 'FGCRH', 'PCSH'}

# Ablation definitions: name -> full disabled_components set
ABLATION_MAP = {
    'full_model': BASE_DISABLED,
    'no_picfp':   BASE_DISABLED | {'PICFP'},
    'no_dsf_hdr': BASE_DISABLED | {'DSF_HDR'},
    'no_rcph':    BASE_DISABLED | {'RCPH'},
}

ABLATION_DESCRIPTIONS = {
    'full_model': 'Full V24 final3 (PICFP + DSF-HDR + RCPH)',
    'no_picfp':   'w/o PICFP - Remove Physics-Informed Carbon Flow Prior',
    'no_dsf_hdr': 'w/o DSF-HDR - Remove Dual-Space Fusion',
    'no_rcph':    'w/o RCPH - Remove Rate-of-Change Prediction Head',
}

# Training hyperparameters (matching final V24 training)
ABLATION_EPOCHS = 25
ABLATION_LR = 3e-4
ABLATION_WEIGHT_DECAY = 0.01
ABLATION_BATCH_SIZE = 512
ABLATION_PATIENCE = 5
DSA_PROB = 0.0  # DSA disabled
USE_AMP = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_MC_SAMPLES = 5

# Prediction save path
PREDICTION_SAVE_PATH = os.path.join(PROJECT_ROOT, 'results', 'v24_final3_predictions.npz')


# ============================================================================
# Dataset (same as train_carbon_sonet_v24.py)
# ============================================================================
class CarbonDataset(Dataset):
    def __init__(self, features, targets, history_len=168, forecast_len=24,
                 region_id=0, n_sources=9):
        super().__init__()
        self.features = np.asarray(features, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.region_id = region_id
        self.n_sources = n_sources
        self.n_samples = max(len(features) - history_len - forecast_len + 1, 0)
        self.features = np.nan_to_num(self.features, nan=0.0)
        self.targets = np.nan_to_num(self.targets, nan=0.0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        h_start = idx
        h_end = idx + self.history_len
        f_end = h_end + self.forecast_len
        x = self.features[h_start:h_end]
        y = self.targets[h_end:f_end]
        sample = {
            'x': torch.tensor(x, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32),
            'region_id': torch.tensor(self.region_id, dtype=torch.long),
        }
        if self.n_sources > 0 and self.features.shape[1] >= 1 + self.n_sources:
            future_gen = self.features[h_end:f_end, 1:1 + self.n_sources]
            sample['future_gen'] = torch.tensor(future_gen, dtype=torch.float32)
        return sample


# ============================================================================
# Data Loading
# ============================================================================
def load_country_data(country_code, split, data_dir):
    path = os.path.join(data_dir, f'{country_code}_{split}.pt')
    return torch.load(path, map_location='cpu', weights_only=False)


def compute_neighbor_features_for_split(country_data_split, country_codes,
                                         norm_mean=None, norm_std=None):
    WINDOW = 24
    x_full = np.arange(WINDOW, dtype=np.float64)
    x_full_c = x_full - x_full.mean()
    slope_denom = (x_full_c ** 2).sum()

    ci = {}
    for code in country_codes:
        d = country_data_split[code]
        if 'targets_raw' in d:
            ci[code] = d['targets_raw'].numpy().astype(np.float64)
        else:
            tm = float(d['target_mean'])
            ts = float(d['target_std'])
            ci[code] = d['targets'].numpy().astype(np.float64) * ts + tm

    raw_feats = {}
    for code in country_codes:
        n = len(ci[code])
        neighbors = [nb for nb in NEIGHBOR_GRID.get(code, []) if nb in ci]
        result = np.zeros((n, 3), dtype=np.float64)
        if not neighbors:
            raw_feats[code] = result.astype(np.float32)
            continue
        nb_arrays = []
        for nb in neighbors:
            arr = ci[nb]
            if len(arr) < n:
                arr = np.pad(arr, (0, n - len(arr)), mode='edge')
            elif len(arr) > n:
                arr = arr[:n]
            nb_arrays.append(arr)
        nb_ci = np.stack(nb_arrays, axis=0)
        nb_mean = nb_ci.mean(axis=0)
        if n >= WINDOW:
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(nb_mean, WINDOW)
            result[WINDOW - 1:, 0] = windows.mean(axis=1)
            result[WINDOW - 1:, 1] = windows @ x_full_c / slope_denom
            result[WINDOW - 1:, 2] = windows.std(axis=1, ddof=0)
            for t in range(WINDOW - 1):
                w = nb_mean[:t + 1]
                result[t, 0] = w.mean()
                if len(w) >= 2:
                    xw = np.arange(len(w), dtype=np.float64)
                    xw_c = xw - xw.mean()
                    d = (xw_c ** 2).sum()
                    if d > 1e-8:
                        result[t, 1] = (xw_c * (w - w.mean())).sum() / d
                    result[t, 2] = w.std(ddof=0) if len(w) > 1 else 0.0
        else:
            for t in range(n):
                w = nb_mean[:t + 1]
                result[t, 0] = w.mean()
                if len(w) >= 2:
                    xw = np.arange(len(w), dtype=np.float64)
                    xw_c = xw - xw.mean()
                    d = (xw_c ** 2).sum()
                    if d > 1e-8:
                        result[t, 1] = (xw_c * (w - w.mean())).sum() / d
                    result[t, 2] = w.std(ddof=0) if len(w) > 1 else 0.0
        raw_feats[code] = result.astype(np.float32)

    if norm_mean is None or norm_std is None:
        all_raw = np.concatenate([raw_feats[c] for c in country_codes], axis=0)
        norm_mean = all_raw.mean(axis=0).astype(np.float32)
        norm_std = all_raw.std(axis=0).astype(np.float32)
        norm_std = np.maximum(norm_std, 1e-8)

    norm_feats = {}
    for code in country_codes:
        normed = (raw_feats[code] - norm_mean) / norm_std
        norm_feats[code] = torch.from_numpy(normed.astype(np.float32))
    return norm_feats, norm_mean, norm_std


def prepare_data(countries, batch_size=512):
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    all_data = {}
    for code in countries:
        all_data[code] = {}
        for split in ['train', 'val', 'test']:
            all_data[code][split] = load_country_data(code, split, data_dir)

    train_nf, norm_mean, norm_std = compute_neighbor_features_for_split(
        {c: all_data[c]['train'] for c in countries}, countries)
    val_nf, _, _ = compute_neighbor_features_for_split(
        {c: all_data[c]['val'] for c in countries}, countries,
        norm_mean=norm_mean, norm_std=norm_std)
    test_nf, _, _ = compute_neighbor_features_for_split(
        {c: all_data[c]['test'] for c in countries}, countries,
        norm_mean=norm_mean, norm_std=norm_std)

    neighbor_feats = {'train': train_nf, 'val': val_nf, 'test': test_nf}
    country_stats = {}
    country_id_to_code = {}
    all_train_datasets = []
    country_val_loaders = {}
    country_test_loaders = {}

    for code in countries:
        rid = CODE_TO_ID[code]
        country_id_to_code[rid] = code
        for split in ['train', 'val', 'test']:
            d = all_data[code][split]
            features = torch.as_tensor(d['features'], dtype=torch.float32)
            targets = torch.as_tensor(d['targets'], dtype=torch.float32)
            feat_mean = torch.as_tensor(d['feat_mean'], dtype=torch.float32)
            feat_std = torch.as_tensor(d['feat_std'], dtype=torch.float32)
            features = features[:, REORDER_INDICES]
            feat_mean = feat_mean[REORDER_INDICES]
            feat_std = feat_std[REORDER_INDICES]
            nf = neighbor_feats[split][code]
            min_len = min(len(features), len(nf))
            features = features[:min_len]
            nf = nf[:min_len]
            targets = targets[:min_len]
            features_ext = torch.cat([features, nf], dim=1)
            feat_mean_19 = torch.cat([feat_mean, torch.zeros(N_NEIGHBOR_FEATURES)])
            feat_std_19 = torch.cat([feat_std, torch.ones(N_NEIGHBOR_FEATURES)])
            if split == 'train':
                country_stats[code] = {
                    'target_mean': float(d['target_mean']),
                    'target_std': float(d['target_std']),
                    'feat_mean_19': feat_mean_19,
                    'feat_std_19': feat_std_19,
                }
                ds = CarbonDataset(features_ext.numpy(), targets.numpy(),
                                   HISTORY_LEN, FORECAST_LEN, rid, N_GEN_FEATURES)
                all_train_datasets.append(ds)
            elif split == 'val':
                ds = CarbonDataset(features_ext.numpy(), targets.numpy(),
                                   HISTORY_LEN, FORECAST_LEN, rid, N_GEN_FEATURES)
                country_val_loaders[code] = DataLoader(
                    ds, batch_size=batch_size * 4, shuffle=False,
                    num_workers=0, pin_memory=True)
            elif split == 'test':
                ds = CarbonDataset(features_ext.numpy(), targets.numpy(),
                                   HISTORY_LEN, FORECAST_LEN, rid, N_GEN_FEATURES)
                country_test_loaders[code] = DataLoader(
                    ds, batch_size=batch_size * 4, shuffle=False,
                    num_workers=0, pin_memory=True)

    joint_train = ConcatDataset(all_train_datasets)
    train_loader = DataLoader(
        joint_train, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
        persistent_workers=True)

    return {
        'train_loader': train_loader,
        'country_val_loaders': country_val_loaders,
        'country_test_loaders': country_test_loaders,
        'country_stats': country_stats,
        'country_id_to_code': country_id_to_code,
    }


# ============================================================================
# Helpers
# ============================================================================
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def evaluate(model, val_loaders, country_stats, mc_dropout=False):
    model.eval()
    if mc_dropout:
        enable_dropout(model)
    results = {}
    with torch.no_grad():
        for country, loader in val_loaders.items():
            tstd = country_stats[country]['target_std']
            tmean = country_stats[country]['target_mean']
            total_mae = 0.0
            n = 0
            for batch in loader:
                x = batch['x'].to(device, non_blocking=True)
                y = batch['y'].to(device, non_blocking=True)
                rid = batch['region_id'].to(device, non_blocking=True)
                fg = batch.get('future_gen')
                if fg is not None:
                    fg = fg.to(device, non_blocking=True)
                if mc_dropout:
                    mc_preds = []
                    for _ in range(N_MC_SAMPLES):
                        pred = model(x, rid, future_gen=fg)['point_pred']
                        mc_preds.append(pred)
                    pred = torch.stack(mc_preds).mean(dim=0)
                else:
                    pred = model(x, rid, future_gen=fg)['point_pred']
                y_real = y * tstd + tmean
                total_mae += F.l1_loss(pred, y_real, reduction='sum').item()
                n += y.numel()
            results[country] = total_mae / max(n, 1)
    model.eval()
    return results


def evaluate_with_predictions(model, test_loaders, country_stats, mc_dropout=True):
    """Evaluate and collect predictions in real space (gCO2/kWh)."""
    model.eval()
    if mc_dropout:
        enable_dropout(model)
    results = {}
    predictions = {}
    with torch.no_grad():
        for country, loader in test_loaders.items():
            tstd = country_stats[country]['target_std']
            tmean = country_stats[country]['target_mean']
            total_mae = 0.0
            n = 0
            all_preds = []
            all_targets = []
            for batch in loader:
                x = batch['x'].to(device, non_blocking=True)
                y = batch['y'].to(device, non_blocking=True)
                rid = batch['region_id'].to(device, non_blocking=True)
                fg = batch.get('future_gen')
                if fg is not None:
                    fg = fg.to(device, non_blocking=True)
                if mc_dropout:
                    mc_preds = []
                    for _ in range(N_MC_SAMPLES):
                        pred = model(x, rid, future_gen=fg)['point_pred']
                        mc_preds.append(pred)
                    pred = torch.stack(mc_preds).mean(dim=0)
                else:
                    pred = model(x, rid, future_gen=fg)['point_pred']
                y_real = y * tstd + tmean
                total_mae += F.l1_loss(pred, y_real, reduction='sum').item()
                n += y.numel()
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y_real.cpu().numpy())
            results[country] = total_mae / max(n, 1)
            predictions[country] = {
                'preds': np.concatenate(all_preds, axis=0),
                'targets': np.concatenate(all_targets, axis=0),
            }
    model.eval()
    return results, predictions


# ============================================================================
# Training
# ============================================================================
def train_one_epoch(model, train_loader, optimizer, scheduler,
                    lambda_physics, lambda_spectral, lambda_router,
                    lambda_rate, lambda_transport, grad_accum=1):
    model.train()
    total_loss = 0.0
    total_main = 0.0
    n_batch = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        x = batch['x'].to(device, non_blocking=True)
        y = batch['y'].to(device, non_blocking=True)
        rid = batch['region_id'].to(device, non_blocking=True)
        fg = batch.get('future_gen')
        if fg is not None:
            fg = fg.to(device, non_blocking=True)

        # DSA prob = 0.0, so no augmentation
        _, _, target_mean, target_std = model._get_stats(rid)
        target_real = y * target_std.unsqueeze(1) + target_mean.unsqueeze(1)

        out = model(x, rid, future_gen=fg)
        point_pred = out['point_pred']
        rate_pred = out['rate_pred']
        satc_calibration = out['satc_calibration']
        physics_residual = out['physics_residual']
        delta_spec = out['delta_spec']
        router_weights = out['router_weights']
        ci_current_real = out['ci_current_real']
        window_std = out['window_std']

        main_loss = F.smooth_l1_loss(point_pred, target_real)
        target_rate = ((target_real - ci_current_real.unsqueeze(-1)) /
                       window_std.unsqueeze(-1).clamp(min=1.0)).float()
        rate_loss = F.mse_loss(rate_pred.float(), target_rate.detach())
        transport_loss = (satc_calibration ** 2).mean()
        phys_loss = (physics_residual ** 2).mean()
        spec_loss = (delta_spec ** 2).mean()
        eps = 1e-8
        router_entropy = -(router_weights * torch.log(router_weights + eps)).sum(dim=-1).mean()

        loss = (main_loss
                + lambda_rate * rate_loss
                + lambda_transport * transport_loss
                + lambda_physics * phys_loss
                + lambda_spectral * spec_loss
                + lambda_router * (-router_entropy)) / grad_accum

        loss.backward()

        if (batch_idx + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * grad_accum
        total_main += main_loss.item()
        n_batch += 1

    return {'loss': total_loss / max(n_batch, 1),
            'main_loss': total_main / max(n_batch, 1)}


# ============================================================================
# Run single ablation experiment
# ============================================================================
def run_ablation(ablation_name, disabled_components, countries, data,
                 output_dir, ckpt_dir_base, save_predictions=False):
    """Run a single ablation experiment (Phase 1 only, no Phase 2)."""
    log.info(f"\n{'=' * 70}")
    log.info(f"ABLATION: {ablation_name}")
    log.info(f"  {ABLATION_DESCRIPTIONS.get(ablation_name, '')}")
    log.info(f"  Disabled components: {disabled_components}")
    log.info(f"{'=' * 70}")

    # Create model
    n_regions = max(CODE_TO_ID[c] for c in countries) + 1
    class _Cfg:
        d_model = 192
        n_layers = 3
        n_heads = 4
        seq_len = HISTORY_LEN
        fore_len = FORECAST_LEN
        n_features = N_TOTAL_FEATURES
        n_gen_features = N_GEN_FEATURES
        dropout = 0.25
        lambda_physics = LAMBDA_PHYSICS
        lambda_spectral = LAMBDA_SPECTRAL
        lambda_router = LAMBDA_ROUTER
        lambda_rate = LAMBDA_RATE
        lambda_transport = LAMBDA_TRANSPORT
    _Cfg.n_regions = n_regions

    model = CarbonSONetV24(_Cfg, disabled_components=list(disabled_components)).to(device)
    model.init_all_from_volatility(data['country_id_to_code'])

    for code in countries:
        rid = CODE_TO_ID[code]
        stats = data['country_stats'][code]
        model.set_country_stats(
            rid, stats['target_mean'], stats['target_std'],
            stats['feat_mean_19'].to(device),
            stats['feat_std_19'].to(device),
        )

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Parameters: {n_params:,}  (disabled: {disabled_components})")

    # Training setup
    ckpt_dir = os.path.join(ckpt_dir_base, f'ablation_{ablation_name}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=ABLATION_LR,
                                  weight_decay=ABLATION_WEIGHT_DECAY)
    total_steps = ABLATION_EPOCHS * len(data['train_loader'])
    warmup_steps = min(200, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Phase 1: Joint Training only (NO Phase 2)
    best_avg_mae = float('inf')
    best_epoch = 0
    no_improve = 0
    t_start = time.time()

    for epoch in range(ABLATION_EPOCHS):
        losses = train_one_epoch(
            model, data['train_loader'], optimizer, scheduler,
            lambda_physics=LAMBDA_PHYSICS,
            lambda_spectral=LAMBDA_SPECTRAL,
            lambda_router=LAMBDA_ROUTER,
            lambda_rate=LAMBDA_RATE,
            lambda_transport=LAMBDA_TRANSPORT,
        )

        val_results = evaluate(model, data['country_val_loaders'], data['country_stats'])
        avg_val = float(np.mean(list(val_results.values())))

        lr_now = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else ABLATION_LR
        elapsed = time.time() - t_start
        log.info(f"  E{epoch}: loss={losses['loss']:.4f}  avg_val={avg_val:.1f}  "
                 f"best_val={best_avg_mae:.1f}  lr={lr_now:.2e}  [{elapsed:.0f}s]")

        if avg_val < best_avg_mae:
            best_avg_mae = avg_val
            best_epoch = epoch
            no_improve = 0
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_mae': avg_val,
                'epoch': epoch,
                'ablation': ablation_name,
            }, os.path.join(ckpt_dir, 'best_val.pt'))
            log.info(f"  * Best VAL saved (avg_val={avg_val:.1f})")
        else:
            no_improve += 1

        if no_improve >= ABLATION_PATIENCE:
            log.info(f"  Early stopping at epoch {epoch} (patience={ABLATION_PATIENCE})")
            break

    # Load best VAL checkpoint
    best_val_path = os.path.join(ckpt_dir, 'best_val.pt')
    if os.path.exists(best_val_path):
        ckpt = torch.load(best_val_path, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        log.info(f"  Loaded best VAL checkpoint (epoch {ckpt['epoch']}, val_mae={ckpt['val_mae']:.1f})")

    # NO Phase 2 - go straight to test evaluation

    # Test Evaluation (MC Dropout)
    log.info(f"\n  Test Evaluation (MC Dropout, N={N_MC_SAMPLES})")

    if save_predictions:
        test_results, predictions = evaluate_with_predictions(
            model, data['country_test_loaders'], data['country_stats'], mc_dropout=True)
    else:
        test_results = {}
        predictions = None
        for code in countries:
            test_res = evaluate(model, {code: data['country_test_loaders'][code]},
                                {code: data['country_stats'][code]}, mc_dropout=True)
            test_results[code] = test_res[code]

    for code in countries:
        log.info(f"    {code}: test={test_results[code]:.1f}")

    avg_test = float(np.mean(list(test_results.values())))
    gb_test = test_results.get('GB', float('inf'))
    log.info(f"  AVG test={avg_test:.1f}  GB={gb_test:.1f}")

    result = {
        'ablation_name': ablation_name,
        'disabled_components': sorted(list(disabled_components)),
        'description': ABLATION_DESCRIPTIONS.get(ablation_name, ablation_name),
        'avg_test_mae': round(avg_test, 2),
        'gb_test_mae': round(gb_test, 2),
        'best_phase1_epoch': best_epoch,
        'best_phase1_val_mae': round(best_avg_mae, 2),
        'per_country_test_mae': {c: round(v, 2) for c, v in test_results.items()},
        'n_params': n_params,
    }

    return result, predictions


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Ablation Study for CarbonSONet V24 Final (3 components)')
    parser.add_argument('--specific', type=str, default=None,
                        help='Run only a specific ablation (e.g., no_picfp)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        PROJECT_ROOT, 'results', 'ablation_v24_final3.json')
    ckpt_dir_base = os.path.join(PROJECT_ROOT, 'results', 'ablation_v24_final3_checkpoints')

    countries = ALL_COUNTRY_CODES

    log.info("=" * 70)
    log.info("CarbonSONet V24 Final - Ablation Study (3 retained components)")
    log.info("  Retained: PICFP, DSF-HDR, RCPH")
    log.info(f"  Base disabled: {sorted(BASE_DISABLED)}")
    log.info(f"  Countries: {len(countries)}")
    log.info("  Phase 1 only (no Phase 2)")
    log.info(f"  Epochs: {ABLATION_EPOCHS}, patience: {ABLATION_PATIENCE}")
    log.info(f"  lr={ABLATION_LR}  weight_decay={ABLATION_WEIGHT_DECAY}  batch_size={ABLATION_BATCH_SIZE}")
    log.info("  DSA prob=0.0  AMP=False")
    log.info("  d_model=192  n_layers=3  n_heads=4")
    log.info(f"  LAMBDA_PHYSICS={LAMBDA_PHYSICS}  LAMBDA_SPECTRAL={LAMBDA_SPECTRAL}")
    log.info(f"  LAMBDA_ROUTER={LAMBDA_ROUTER}  LAMBDA_RATE={LAMBDA_RATE}  LAMBDA_TRANSPORT={LAMBDA_TRANSPORT}")
    log.info(f"  Output: {output_path}")
    log.info(f"  Predictions: {PREDICTION_SAVE_PATH}")
    log.info("=" * 70)

    # Prepare data (shared across all ablations)
    log.info("\nPreparing data...")
    data = prepare_data(countries, batch_size=ABLATION_BATCH_SIZE)
    log.info("Data prepared.")

    # Determine which ablations to run
    if args.specific:
        ablation_keys = [args.specific]
    else:
        ablation_keys = list(ABLATION_MAP.keys())

    all_results = {}

    # Load existing results for resume support
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                existing = json.load(f)
            if isinstance(existing, dict) and 'ablations' in existing:
                all_results = existing
                completed = set(existing.get('ablations', {}).keys())
                log.info(f"Resuming: {len(completed)} ablations already done")
                ablation_keys = [k for k in ablation_keys if k not in completed]
        except Exception:
            pass

    # Track full_model predictions for NPZ saving
    full_model_predictions = None

    for ablation_key in ablation_keys:
        disabled = ABLATION_MAP[ablation_key]
        is_full_model = (ablation_key == 'full_model')
        result, predictions = run_ablation(
            ablation_key, disabled, countries, data,
            os.path.dirname(output_path), ckpt_dir_base,
            save_predictions=is_full_model)

        if is_full_model and predictions is not None:
            full_model_predictions = predictions

        all_results.setdefault('ablations', {})[ablation_key] = result

        # Save incrementally
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        log.info(f"  Results saved to {output_path}")

    # Save full_model predictions as NPZ
    if full_model_predictions is not None:
        npz_dict = {}
        for code in countries:
            npz_dict[f'{code}_predictions'] = full_model_predictions[code]['preds']
            npz_dict[f'{code}_targets'] = full_model_predictions[code]['targets']
        np.savez(PREDICTION_SAVE_PATH, **npz_dict)
        log.info(f"  Full model predictions saved to {PREDICTION_SAVE_PATH}")
    else:
        # Try to load from already completed full_model checkpoint
        log.warning("  No full_model predictions collected (may have been resumed). Running prediction extraction...")
        n_regions = max(CODE_TO_ID[c] for c in countries) + 1
        class _CfgReload:
            d_model = 192
            n_layers = 3
            n_heads = 4
            seq_len = HISTORY_LEN
            fore_len = FORECAST_LEN
            n_features = N_TOTAL_FEATURES
            n_gen_features = N_GEN_FEATURES
            dropout = 0.25
            lambda_physics = LAMBDA_PHYSICS
            lambda_spectral = LAMBDA_SPECTRAL
            lambda_router = LAMBDA_ROUTER
            lambda_rate = LAMBDA_RATE
            lambda_transport = LAMBDA_TRANSPORT
        _CfgReload.n_regions = n_regions
        model = CarbonSONetV24(_CfgReload, disabled_components=list(ABLATION_MAP['full_model'])).to(device)
        model.init_all_from_volatility(data['country_id_to_code'])
        for code in countries:
            rid = CODE_TO_ID[code]
            stats = data['country_stats'][code]
            model.set_country_stats(
                rid, stats['target_mean'], stats['target_std'],
                stats['feat_mean_19'].to(device),
                stats['feat_std_19'].to(device),
            )
        ckpt_dir = os.path.join(ckpt_dir_base, 'ablation_full_model')
        best_val_path = os.path.join(ckpt_dir, 'best_val.pt')
        if os.path.exists(best_val_path):
            ckpt = torch.load(best_val_path, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            _, predictions = evaluate_with_predictions(
                model, data['country_test_loaders'], data['country_stats'], mc_dropout=True)
            npz_dict = {}
            for code in countries:
                npz_dict[f'{code}_predictions'] = predictions[code]['preds']
                npz_dict[f'{code}_targets'] = predictions[code]['targets']
            np.savez(PREDICTION_SAVE_PATH, **npz_dict)
            log.info(f"  Full model predictions saved to {PREDICTION_SAVE_PATH}")

    # Generate summary table with per-country MAE for GB/DK/DE/ES
    log.info(f"\n{'=' * 70}")
    log.info("ABLATION SUMMARY")
    log.info(f"{'=' * 70}")

    ablations = all_results.get('ablations', {})
    full_avg = ablations.get('full_model', {}).get('avg_test_mae', 0)

    SHOW_COUNTRIES = ['GB', 'DK', 'DE', 'ES']
    summary_lines = []
    header = f"{'Ablation':<15} {'Avg':>6} {'GB':>6} {'DK':>6} {'DE':>6} {'ES':>6} {'Delta':>7} {'Delta%':>7}"
    summary_lines.append(header)
    summary_lines.append("-" * len(header))

    for key in ABLATION_MAP.keys():
        if key in ablations:
            r = ablations[key]
            avg = r['avg_test_mae']
            per_country = r.get('per_country_test_mae', {})
            gb = per_country.get('GB', 0)
            dk = per_country.get('DK', 0)
            de = per_country.get('DE', 0)
            es = per_country.get('ES', 0)
            delta = avg - full_avg if key != 'full_model' else 0
            delta_pct = (delta / full_avg * 100) if full_avg > 0 and key != 'full_model' else 0
            summary_lines.append(
                f"{key:<15} {avg:>6.2f} {gb:>6.1f} {dk:>6.1f} {de:>6.1f} {es:>6.1f} {delta:>+7.2f} {delta_pct:>+6.1f}%")

    for line in summary_lines:
        log.info(line)

    # Verify full_model is competitive with previous 4-component no_pcsh result (3.56)
    if full_avg > 0:
        prev_ref = 3.56
        if full_avg <= prev_ref + 0.3:
            log.info(f"\n  VERIFICATION: full_model avg MAE={full_avg:.2f} is close to or better than prev no_pcsh={prev_ref}")
        else:
            log.warning(f"\n  VERIFICATION WARNING: full_model avg MAE={full_avg:.2f} is worse than prev no_pcsh={prev_ref} by {full_avg - prev_ref:.2f}")

    all_results['summary_table'] = summary_lines

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    log.info(f"\nAblation study complete. Results: {output_path}")
    log.info(f"Predictions: {PREDICTION_SAVE_PATH}")


if __name__ == '__main__':
    main()
