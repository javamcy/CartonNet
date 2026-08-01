"""
CarbonSONet: Physics-Informed Carbon Intensity Forecasting

A physics-informed sequence-to-sequence model for real-time carbon intensity
forecasting across European power systems. The model embeds the physical
relationship between electricity generation and carbon emissions directly
into the forecasting architecture as a carbon flow prior (PICFP), combined
with a rate-of-change prediction head (RCPH) and dual-space fusion with
horizon-dependent reweighting (DSF-HDR).

Key components:
  1. PICFP (Physics-Informed Carbon Flow Prior): Uses day-ahead generation
     forecasts and fuel-specific emission factors to compute a physically
     grounded baseline prediction.
  2. RCPH (Rate-of-Change Prediction Head): Predicts normalized carbon
     intensity changes rather than absolute values for level-invariance.
  3. DSF-HDR (Dual-Space Fusion with Horizon-Dependent Reweighting):
     Adaptively combines absolute and rate-of-change prediction paths.

     This is NOT data leakage: only uses INPUT window statistics available
     at test time.

  3. Dual-Space Fusion with Horizon-Dependent Routing (DSF-HDR)
     Inspired by Mixture of Experts (Jacobs et al., 1991), adaptive
     computation (Graves, 2016), and multi-scale forecasting (Liu et al.,
     ICLR 2024).
     Predict in TWO spaces simultaneously: absolute value (V22+ style) and
     rate-of-change (new).  A learned, horizon-dependent router blends them:
       weight_h = sigmoid(g(horizon, shift_mag, country_id))
       pred_final = weight_h * pred_absolute_calibrated
                  + (1 - weight_h) * pred_rate_recovered
     For short horizons (1-6h): trust absolute prediction more.
     For long horizons (7-24h): trust rate prediction more.
     For high-shift countries (GB): trust rate prediction more at all horizons.
     For low-shift countries (FR, ES): trust absolute prediction more.

All V22+ components (PICFP, MDIE, MSSD, PCSH, CCSR, Country-FiLM, HAPF,
Causal Decomposition Heads) are retained.  V24 is purely additive.

Self-contained implementation -- no imports from project modules.
Only depends on: torch, math, numpy, typing.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

# ═══════════════════════════════════════════════════════════════════════════
# Constants (copied from V22+)
# ═══════════════════════════════════════════════════════════════════════════

EMISSION_FACTORS = [230, 820, 490, 12, 41, 11, 24, 500, 300]
# biomass, coal, gas, nuclear, solar, wind_on, wind_off, other, imports

BASELOAD_INDICES = [0, 3, 7, 8]    # biomass, nuclear, other, imports
RENEWABLE_INDICES = [4, 5, 6]       # solar, wind_on, wind_off
FOSSIL_INDICES = [1, 2]             # coal, gas

COUNTRY_VOLATILITY_TIER = {
    'GB': 3, 'DE': 3, 'FR': 2, 'ES': 3, 'IT': 2, 'SE': 1,
    'NL': 2, 'BE': 2, 'AT': 2, 'PL': 4, 'DK': 3, 'FI': 1,
    'PT': 2, 'GR': 3, 'CZ': 2, 'NO': 1,
}

TIER_TO_INIT_ALPHA = {1: -2.0, 2: -1.0, 3: -0.3, 4: 0.5}
TIER_TO_GAMMA_INIT = {1: 0.5, 2: 0.8, 3: 1.2, 4: 1.5}

# MSSD: 5 dominant periods (in hours)
MSSD_PERIODS = [24, 12, 6, 168, 72]
MSSD_N_PERIODS = len(MSSD_PERIODS)

LAMBDA_PHYSICS = 0.05
MDIE_DIM = 4
LAMBDA_SPECTRAL = 0.01
LAMBDA_ROUTER = 0.001

# V24: new loss weights
LAMBDA_RATE = 0.1          # rate-of-change loss weight
LAMBDA_TRANSPORT = 0.01    # transport calibration regularisation weight

DEFAULT_COUNTRY_CODES = [
    'AT', 'BE', 'CZ', 'DE', 'DK', 'ES', 'FI', 'FR',
    'GB', 'GR', 'IT', 'NL', 'NO', 'PL', 'PT', 'SE',
]

NEIGHBOR_GRID = {
    'GB': ['FR', 'NL', 'DK'],
    'DE': ['FR', 'SE', 'NL', 'AT', 'PL', 'DK', 'CZ'],
    'FR': ['GB', 'DE', 'ES', 'IT', 'BE', 'PT'],
    'ES': ['FR', 'PT'],
    'IT': ['FR', 'AT', 'GR'],
    'SE': ['DE', 'PL', 'DK', 'FI', 'NO'],
    'NL': ['GB', 'DE', 'BE', 'DK'],
    'BE': ['FR', 'NL'],
    'AT': ['DE', 'IT', 'CZ'],
    'PL': ['DE', 'SE', 'CZ'],
    'DK': ['GB', 'DE', 'SE', 'NL', 'FI', 'NO'],
    'FI': ['SE', 'DK', 'NO'],
    'PT': ['FR', 'ES'],
    'GR': ['IT'],
    'CZ': ['DE', 'AT', 'PL'],
    'NO': ['SE', 'DK', 'FI'],
}

COUNTRY_PERIOD_PRIOR = {
    'ES': [0.40, 0.15, 0.10, 0.20, 0.15],
    'IT': [0.35, 0.15, 0.10, 0.25, 0.15],
    'PT': [0.40, 0.15, 0.10, 0.20, 0.15],
    'GR': [0.35, 0.15, 0.10, 0.25, 0.15],
    'DK': [0.15, 0.30, 0.30, 0.10, 0.15],
    'DE': [0.20, 0.20, 0.20, 0.20, 0.20],
    'GB': [0.20, 0.25, 0.25, 0.15, 0.15],
    'NL': [0.20, 0.20, 0.20, 0.20, 0.20],
    'FR': [0.20, 0.10, 0.10, 0.45, 0.15],
    'SE': [0.15, 0.10, 0.10, 0.40, 0.25],
    'FI': [0.10, 0.10, 0.20, 0.30, 0.30],
    'NO': [0.15, 0.10, 0.15, 0.35, 0.25],
    'PL': [0.20, 0.15, 0.10, 0.30, 0.25],
    'CZ': [0.20, 0.15, 0.10, 0.35, 0.20],
    'BE': [0.20, 0.15, 0.10, 0.30, 0.25],
    'AT': [0.25, 0.15, 0.10, 0.30, 0.20],
}

CI_FEATURE_IDX = 0
GEN_FEATURE_START = 1
N_BASE_FEATURES = 16
NEIGHBOR_FEATURE_START = 16


# ═══════════════════════════════════════════════════════════════════════════
# V22+ Sub-modules (copied verbatim)
# ═══════════════════════════════════════════════════════════════════════════

class CountryFiLM(nn.Module):
    """Layer-wise Country-FiLM modulation:  gamma * x + beta"""

    def __init__(self, n_regions: int, d_model: int):
        super().__init__()
        self.gamma_raw = nn.Embedding(n_regions, d_model)
        self.beta = nn.Embedding(n_regions, d_model)
        nn.init.zeros_(self.gamma_raw.weight)
        nn.init.zeros_(self.beta.weight)

    def init_from_volatility(self, country_id_to_code: Dict[int, str]):
        with torch.no_grad():
            for cid, code in country_id_to_code.items():
                tier = COUNTRY_VOLATILITY_TIER[code]
                self.gamma_raw.weight[cid].fill_(TIER_TO_GAMMA_INIT[tier])
                self.beta.weight[cid].zero_()

    def forward(self, x: torch.Tensor, region_id: torch.Tensor) -> torch.Tensor:
        gamma = torch.sigmoid(self.gamma_raw(region_id)).unsqueeze(1)
        beta = self.beta(region_id).unsqueeze(1)
        return x * gamma + beta


class RenewableAwarePositionalEncoding(nn.Module):
    """Combined positional encoding:
    1. Standard sinusoidal PE (position-based, 10000 base)
    2. Solar-cycle PE  (24 h diurnal harmonics)
    3. Weekly-cycle PE (168 h weekly harmonics)
    4. Wind PE  (per-country x per-hour-of-day learned)
    """

    def __init__(
        self,
        d_model: int,
        seq_len: int,
        n_regions: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer('pe_standard', pe)

        n_solar_harm = max(d_model // 8, 1)
        pe_solar = torch.zeros(seq_len, d_model)
        hour = torch.arange(seq_len, dtype=torch.float32) % 24.0
        for k in range(n_solar_harm):
            freq = 2.0 * math.pi * (k + 1) / 24.0
            idx_s, idx_c = 2 * k, 2 * k + 1
            if idx_c < d_model:
                pe_solar[:, idx_s] = torch.sin(freq * hour)
                pe_solar[:, idx_c] = torch.cos(freq * hour)
        self.register_buffer('pe_solar', pe_solar)

        n_weekly_harm = max(d_model // 8, 1)
        pe_weekly = torch.zeros(seq_len, d_model)
        t = torch.arange(seq_len, dtype=torch.float32)
        for k in range(n_weekly_harm):
            freq = 2.0 * math.pi * (k + 1) / 168.0
            idx_s, idx_c = 2 * k, 2 * k + 1
            if idx_c < d_model:
                pe_weekly[:, idx_s] = torch.sin(freq * t)
                pe_weekly[:, idx_c] = torch.cos(freq * t)
        self.register_buffer('pe_weekly', pe_weekly)

        self.wind_hour_embed = nn.Embedding(24, d_model)
        self.wind_country_scale = nn.Embedding(n_regions, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, region_id: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        pe = (self.pe_standard[:S]
              + self.pe_solar[:S]
              + self.pe_weekly[:S])
        x = x + pe.unsqueeze(0)

        hours = torch.arange(S, device=x.device) % 24
        wind_base = self.wind_hour_embed(hours)
        country_scale = self.wind_country_scale(region_id)
        x = x + wind_base.unsqueeze(0) * country_scale.unsqueeze(1)

        return self.dropout(x)


class HorizonAdaptivePersistenceFusion(nn.Module):
    """HAPF:  gamma(h) = sigmoid(alpha_country + beta * h)
    CI_final = gamma * CI_model + (1 - gamma) * CI_persistence
    """

    def __init__(self, n_regions: int, fore_len: int):
        super().__init__()
        self.fore_len = fore_len
        self.alpha = nn.Embedding(n_regions, 1)
        self.beta = nn.Parameter(torch.tensor(-0.1))
        nn.init.zeros_(self.alpha.weight)

    def init_from_volatility(self, country_id_to_code: Dict[int, str]):
        with torch.no_grad():
            for cid, code in country_id_to_code.items():
                tier = COUNTRY_VOLATILITY_TIER[code]
                self.alpha.weight[cid].fill_(TIER_TO_INIT_ALPHA[tier])

    def forward(
        self,
        ci_model: torch.Tensor,
        ci_persistence: torch.Tensor,
        region_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = self.alpha(region_id)
        h = torch.arange(self.fore_len, device=ci_model.device,
                          dtype=torch.float32)
        gamma = torch.sigmoid(alpha + self.beta * h.unsqueeze(0))
        ci_final = gamma * ci_model + (1.0 - gamma) * ci_persistence
        return ci_final, gamma


class PerCountryHead(nn.Module):
    """Per-Country Prediction Head.
    Shared backbone -> h [B, d_half] -> per-country weight matrix -> [B, out_dim].
    """

    def __init__(self, n_regions: int, d_half: int, out_dim: int):
        super().__init__()
        self.d_half = d_half
        self.out_dim = out_dim
        self.head_weights = nn.Embedding(n_regions, d_half * out_dim)
        nn.init.zeros_(self.head_weights.weight)

    def forward(self, h: torch.Tensor, region_id: torch.Tensor) -> torch.Tensor:
        W = self.head_weights(region_id).view(-1, self.d_half, self.out_dim)
        return torch.bmm(h.unsqueeze(1), W).squeeze(1)


class MissingDataImputationEmbedding(nn.Module):
    """V22 MDIE: per-country, per-source missing/present embedding."""

    def __init__(self, n_regions: int, n_gen_features: int, dim: int = MDIE_DIM):
        super().__init__()
        self.n_gen_features = n_gen_features
        self.dim = dim
        self.embed_missing = nn.Embedding(n_regions * n_gen_features, dim)
        self.embed_present = nn.Embedding(n_regions * n_gen_features, dim)
        nn.init.zeros_(self.embed_missing.weight)
        nn.init.zeros_(self.embed_present.weight)

    def forward(
        self,
        gen_real: torch.Tensor,
        region_id: torch.Tensor,
    ) -> torch.Tensor:
        B, T, G = gen_real.shape
        gen_abs_max = gen_real.abs().max(dim=1).values
        missing_mask = (gen_abs_max < 1e-3).long()
        base = region_id.unsqueeze(1).expand(-1, G) * G
        src_idx = base + torch.arange(G, device=region_id.device).unsqueeze(0)

        emb_m = self.embed_missing(src_idx)
        emb_p = self.embed_present(src_idx)
        mask = missing_mask.unsqueeze(-1).float()
        emb = emb_p * (1.0 - mask) + emb_m * mask
        return emb.view(B, G * self.dim)


class FutureGenMDIEConditionedHeads(nn.Module):
    """V22 FGCRH + MDIE conditioned causal decomposition heads."""

    BASELOAD_SCALE = 0.5
    RENEWABLE_SCALE = 1.0
    FOSSIL_SCALE = 0.8

    def __init__(self, n_regions: int, d_half: int, fore_len: int,
                 n_gen_features: int = 9, mdie_dim: int = MDIE_DIM):
        super().__init__()
        self.fore_len = fore_len
        self.n_gen_features = n_gen_features
        self.mdie_dim = mdie_dim

        self.baseload_head = PerCountryHead(n_regions, d_half, fore_len)

        self.renewable_head = PerCountryHead(n_regions, d_half, fore_len)
        self.solar_prior_proj = nn.Linear(1, d_half)
        nn.init.zeros_(self.solar_prior_proj.weight)
        nn.init.zeros_(self.solar_prior_proj.bias)

        self.future_gen_renew_proj = nn.Linear(n_gen_features, d_half)
        nn.init.zeros_(self.future_gen_renew_proj.weight)
        nn.init.zeros_(self.future_gen_renew_proj.bias)

        self.mdie_renew_proj = nn.Linear(n_gen_features * mdie_dim, d_half)
        nn.init.zeros_(self.mdie_renew_proj.weight)
        nn.init.zeros_(self.mdie_renew_proj.bias)

        self.fossil_head = PerCountryHead(n_regions, d_half, fore_len)
        self.renewable_proj = nn.Linear(fore_len, d_half)
        nn.init.zeros_(self.renewable_proj.weight)
        nn.init.zeros_(self.renewable_proj.bias)

        self.future_gen_fossil_proj = nn.Linear(len(FOSSIL_INDICES), d_half)
        nn.init.zeros_(self.future_gen_fossil_proj.weight)
        nn.init.zeros_(self.future_gen_fossil_proj.bias)

        self.mdie_fossil_proj = nn.Linear(n_gen_features * mdie_dim, d_half)
        nn.init.zeros_(self.mdie_fossil_proj.weight)
        nn.init.zeros_(self.mdie_fossil_proj.bias)

        self.register_buffer(
            '_fossil_gen_idx',
            torch.tensor(FOSSIL_INDICES, dtype=torch.long),
        )
        self.register_buffer(
            '_renewable_gen_idx',
            torch.tensor(RENEWABLE_INDICES, dtype=torch.long),
        )

    def compute_solar_prior(self, current_hour: int, device: torch.device,
                            batch_size: int) -> torch.Tensor:
        h = torch.arange(self.fore_len, device=device, dtype=torch.float32)
        angles = math.pi * (h + current_hour) / 24.0
        solar_factor = torch.sin(angles).clamp(min=0.0)
        return solar_factor.unsqueeze(0).expand(batch_size, -1)

    def forward(
        self,
        h: torch.Tensor,
        region_id: torch.Tensor,
        current_hour: int,
        future_gen: Optional[torch.Tensor] = None,
        mdie: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = h.size(0)
        device = h.device

        delta_baseload = torch.tanh(self.baseload_head(h, region_id)) * self.BASELOAD_SCALE

        solar_prior = self.compute_solar_prior(current_hour, device, B)
        solar_cond = self.solar_prior_proj(solar_prior.mean(dim=1, keepdim=True))

        if future_gen is not None:
            fg = future_gen.clone()
            fg_abs_max = fg.abs().max(dim=1).values
            missing_mask = fg_abs_max < 1e-6
            fg_mean = fg.mean(dim=1)
            fg_renew_cond = self.future_gen_renew_proj(fg_mean)
            renew_missing = missing_mask[:, self._renewable_gen_idx].any(dim=1)
            fg_renew_cond = fg_renew_cond * (~renew_missing).unsqueeze(1).float()

            mdie_cond = torch.zeros_like(fg_renew_cond)
            if mdie is not None:
                mdie_cond = self.mdie_renew_proj(mdie)

            h_renew = h + solar_cond.squeeze(1) * 0.1 + fg_renew_cond * 0.1 + mdie_cond * 0.1
        else:
            h_renew = h + solar_cond.squeeze(1) * 0.1

        delta_renewable = self.renewable_head(h_renew, region_id) * self.RENEWABLE_SCALE

        renew_cond = self.renewable_proj(delta_renewable.detach())
        if future_gen is not None:
            fg_fossil = future_gen[:, :, self._fossil_gen_idx]
            fg_fossil_mean = fg_fossil.mean(dim=1)
            fg_fossil_cond = self.future_gen_fossil_proj(fg_fossil_mean)
            fossil_missing = missing_mask[:, self._fossil_gen_idx].any(dim=1)
            fg_fossil_cond = fg_fossil_cond * (~fossil_missing).unsqueeze(1).float()

            mdie_fossil_cond = torch.zeros_like(fg_fossil_cond)
            if mdie is not None:
                mdie_fossil_cond = self.mdie_fossil_proj(mdie)

            h_fossil = h + renew_cond * 0.1 + fg_fossil_cond * 0.1 + mdie_fossil_cond * 0.1
        else:
            h_fossil = h + renew_cond * 0.1

        delta_fossil = self.fossil_head(h_fossil, region_id) * self.FOSSIL_SCALE

        return delta_baseload, delta_renewable, delta_fossil


class PhysicsInformedCarbonFlowPrior(nn.Module):
    """V22 PICFP: exact physics-based CI from future_gen."""

    def __init__(self, n_regions: int, fore_len: int, n_gen_features: int = 9):
        super().__init__()
        self.fore_len = fore_len
        self.n_gen_features = n_gen_features

        self.scale_raw = nn.Embedding(n_regions, fore_len)
        nn.init.zeros_(self.scale_raw.weight)
        self.shift = nn.Embedding(n_regions, fore_len)
        nn.init.zeros_(self.shift.weight)

    def forward(
        self,
        future_gen: Optional[torch.Tensor],
        region_id: torch.Tensor,
        ci_persistence: torch.Tensor,
    ) -> torch.Tensor:
        B = future_gen.size(0) if future_gen is not None else ci_persistence.size(0)
        device = region_id.device

        if future_gen is None:
            return ci_persistence

        gen_total = future_gen.sum(dim=-1).clamp(min=0.0) + 1e-6
        gen_frac = future_gen / gen_total.unsqueeze(-1)
        gen_frac = gen_frac.clamp(min=0.0, max=1.0)
        ef = torch.tensor(EMISSION_FACTORS[:self.n_gen_features],
                          dtype=gen_frac.dtype, device=device)
        ci_phys = (gen_frac * ef.view(1, 1, -1)).sum(dim=-1)

        a_raw = self.scale_raw(region_id)
        a = 1.0 + 0.1 * a_raw
        b = self.shift(region_id)
        return a * ci_phys + b


class MultiScaleSpectralDecomposition(nn.Module):
    """V22+ MSSD: multi-scale spectral decomposition."""

    def __init__(self, n_regions: int, d_model: int, n_periods: int = MSSD_N_PERIODS):
        super().__init__()
        self.n_periods = n_periods
        self.d_model = d_model

        self.register_buffer(
            'period_indices',
            torch.tensor(
                [168 // p for p in MSSD_PERIODS],
                dtype=torch.long,
            ),
        )

        self.period_inception = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=(1, 3), padding=(0, 1)),
                nn.GELU(),
                nn.Conv2d(16, 16, kernel_size=(3, 1), padding=(1, 0)),
                nn.GELU(),
            )
            for _ in range(n_periods)
        ])

        self.period_spatial = [7, 14, 28, 1, 2]
        self.period_projs = nn.ModuleList([
            nn.Linear(16 * sp, d_model) for sp in self.period_spatial
        ])

        self.router = nn.Embedding(n_regions, n_periods)

        self.norm = nn.LayerNorm(d_model)

    def init_from_volatility(self, country_id_to_code: Dict[int, str]):
        with torch.no_grad():
            for cid, code in country_id_to_code.items():
                if code in COUNTRY_PERIOD_PRIOR:
                    prior = torch.tensor(
                        COUNTRY_PERIOD_PRIOR[code],
                        dtype=self.router.weight.dtype,
                    )
                    self.router.weight[cid].copy_(prior * 4.0)
                else:
                    self.router.weight[cid].fill_(0.0)

    def forward(
        self,
        x_ci: torch.Tensor,
        region_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S = x_ci.shape
        device = x_ci.device

        X = torch.fft.rfft(x_ci, dim=-1)

        period_feats: List[torch.Tensor] = []
        for k, (idx, sp) in enumerate(zip(self.period_indices.tolist(),
                                          self.period_spatial)):
            idx = max(1, min(idx, X.size(-1) - 1))
            lo = max(0, idx - 1)
            hi = min(X.size(-1), idx + sp // 2 + 1)
            sub = X[:, lo:hi]
            real = sub.real.unsqueeze(1)
            imag = sub.imag.unsqueeze(1)
            twod = torch.cat([real, imag], dim=1)
            if sub.shape[-1] < sp:
                pad = sp - sub.shape[-1]
                twod = F.pad(twod, (0, pad), mode='replicate')
            elif sub.shape[-1] > sp:
                twod = twod[:, :, :sp]
            twod = twod.unsqueeze(2)
            h_k = self.period_inception[k](twod)
            h_k = h_k.squeeze(2).flatten(1)
            h_k = self.period_projs[k](h_k)
            period_feats.append(h_k)

        period_stack = torch.stack(period_feats, dim=1)

        router_logits = self.router(region_id)
        router_weights = F.softmax(router_logits, dim=-1)

        spectral_features = (period_stack * router_weights.unsqueeze(-1)).sum(dim=1)
        spectral_features = self.norm(spectral_features)

        return spectral_features, router_weights


class PhysicsConstrainedSpectralHead(nn.Module):
    """V22+ PCSH: physics-constrained spectral head."""

    def __init__(self, n_regions: int, d_model: int, fore_len: int):
        super().__init__()
        self.fore_len = fore_len
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, fore_len),
        )
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        self.gate = nn.Embedding(n_regions, 1)
        nn.init.zeros_(self.gate.weight)

        self.energy_logit = nn.Embedding(n_regions, 1)
        nn.init.zeros_(self.energy_logit.weight)

    def init_from_volatility(self, country_id_to_code: Dict[int, str]):
        with torch.no_grad():
            for cid, code in country_id_to_code.items():
                tier = COUNTRY_VOLATILITY_TIER[code]
                self.energy_logit.weight[cid].fill_(0.5 * (tier - 2))

    def forward(
        self,
        spectral_features: torch.Tensor,
        region_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.proj(spectral_features)
        energy = F.softplus(self.energy_logit(region_id).squeeze(-1) + 1.0)
        scale = energy.unsqueeze(-1).clamp(min=0.5)
        delta_spec = scale * torch.tanh(raw / scale)
        gate = torch.sigmoid(self.gate(region_id)).expand(-1, self.fore_len)
        return delta_spec * gate, gate, energy


# ═══════════════════════════════════════════════════════════════════════════
# V24 INNOVATION 1: Rate-of-Change Prediction Head (RCPH)
# ═══════════════════════════════════════════════════════════════════════════

class RateOfChangePredictionHead(nn.Module):
    """V24 INNOVATION 1: Rate-of-Change Prediction Head (RCPH)

    Inspired by financial return modeling (Cont, 2001), RevIN (Kim et al.,
    ICLR 2022), and SAN (Liu et al., ICML 2024).

    Instead of predicting absolute CI values, predict the NORMALIZED rate
    of change:
        rate_h = (CI_{t+h} - CI_current) / (window_std + eps)
    Recovery:
        CI_pred = CI_current + rate_h * window_std

    This makes predictions inherently level-invariant.  For GB where test
    level is 65% of training, the rate of change is still consistent.

    Architecture:
        h_backbone [B, d_half] -> PerCountryHead -> [B, fore_len]
        The output is the predicted rate in NORMALIZED space.
        Recovery is done externally using window statistics.
    """

    def __init__(self, n_regions: int, d_half: int, fore_len: int):
        super().__init__()
        self.fore_len = fore_len
        # Per-country rate prediction head, zero-initialised
        self.rate_head = PerCountryHead(n_regions, d_half, fore_len)
        # Small learnable scale to prevent too-aggressive rate predictions
        self.rate_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        h_backbone: torch.Tensor,
        region_id: torch.Tensor,
    ) -> torch.Tensor:
        """Predict normalized rate of change.

        Returns
        -------
        rate_pred : [B, fore_len]  predicted rate in normalized space
        """
        rate_pred = self.rate_head(h_backbone, region_id) * self.rate_scale
        return rate_pred


# ═══════════════════════════════════════════════════════════════════════════
# V24 INNOVATION 2: Shift-Adaptive Transport Calibration (SATC)
# ═══════════════════════════════════════════════════════════════════════════

class ShiftAdaptiveTransportCalibration(nn.Module):
    """V24 INNOVATION 2: Shift-Adaptive Transport Calibration (SATC)

    Inspired by optimal transport for domain adaptation (Courty et al.,
    NeurIPS 2017), normalizing flows (Rezende & Mohamed, ICML 2015), and
    test-time training (Sun et al., NeurIPS 2020).

    Learn a monotone transport function that maps predictions from the
    training distribution to the test distribution.  The transport is
    CONDITIONED on the observed shift magnitude and country ID:

        shift = (window_mean - train_mean) / train_std
        calibration = MLP([shift, country_embedding]) -> scalar per horizon
        pred_calibrated = pred * (1 + calibration)

    Key properties:
      - When shift ~ 0 (no distribution shift), calibration ~ 0 (no-op)
      - When shift > 0 (window above training mean), calibration adjusts
        the scale to match observed level
      - When shift < 0 (window below training mean, e.g. GB), calibration
        scales down predictions to match the lower test level
      - The MLP is small and zero-initialised so it starts as identity

    This is NOT data leakage: only uses INPUT window statistics available
    at test time (the same statistics used for normalisation).
    """

    def __init__(self, n_regions: int, fore_len: int, d_embed: int = 16):
        super().__init__()
        self.fore_len = fore_len
        self.d_embed = d_embed

        # Country embedding for shift-aware calibration
        self.country_embed = nn.Embedding(n_regions, d_embed)
        nn.init.zeros_(self.country_embed.weight)

        # Small MLP: [shift_scalar, country_embed] -> fore_len calibration factors
        # Input dim = 1 (shift) + d_embed (country)
        self.calib_mlp = nn.Sequential(
            nn.Linear(1 + d_embed, 32),
            nn.GELU(),
            nn.Linear(32, fore_len),
        )
        # Zero-init so calibration starts as identity (1 + 0 = 1)
        nn.init.zeros_(self.calib_mlp[-1].weight)
        nn.init.zeros_(self.calib_mlp[-1].bias)

    def forward(
        self,
        pred: torch.Tensor,
        shift_mag: torch.Tensor,
        region_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply shift-adaptive transport calibration.

        Parameters
        ----------
        pred      : [B, fore_len]  absolute predictions in real space
        shift_mag : [B]  normalised shift magnitude
            shift = (window_mean - train_mean) / train_std
        region_id : [B]  country index

        Returns
        -------
        pred_calibrated : [B, fore_len]  calibrated predictions
        calibration     : [B, fore_len]  per-horizon calibration factors
        """
        # Embed country
        c_embed = self.country_embed(region_id)   # [B, d_embed]

        # Concat shift magnitude with country embedding
        shift_input = torch.cat([
            shift_mag.unsqueeze(-1),   # [B, 1]
            c_embed,                    # [B, d_embed]
        ], dim=-1)                      # [B, 1 + d_embed]

        # MLP -> per-horizon calibration
        calibration = self.calib_mlp(shift_input)  # [B, fore_len]

        # Apply monotone transport: pred * (1 + calibration)
        # tanh bounds the calibration to [-1, 1] for stability
        calibration = torch.tanh(calibration) * 0.5  # bounded in [-0.5, 0.5]
        pred_calibrated = pred * (1.0 + calibration)

        return pred_calibrated, calibration


# ═══════════════════════════════════════════════════════════════════════════
# V24 INNOVATION 3: Dual-Space Fusion with Horizon-Dependent Routing (DSF-HDR)
# ═══════════════════════════════════════════════════════════════════════════

class DualSpaceFusionHDR(nn.Module):
    """V24 INNOVATION 3: Dual-Space Fusion with Horizon-Dependent Routing (DSF-HDR)

    Inspired by Mixture of Experts (Jacobs et al., 1991), adaptive
    computation (Graves, 2016), and multi-scale forecasting (Liu et al.,
    ICLR 2024).

    Predict in TWO spaces simultaneously: absolute value (V22+ style) and
    rate-of-change (RCPH).  A learned, horizon-dependent router blends
    the two:

        weight_h = sigmoid(g(horizon, shift_mag, country_id))
        pred_final = weight_h * pred_absolute_calibrated
                   + (1 - weight_h) * pred_rate_recovered

    For short horizons (1-6h): trust absolute prediction more
        (patterns are similar to training)
    For long horizons (7-24h): trust rate prediction more
        (level-invariant)
    For high-shift countries (GB): trust rate prediction more at all horizons
    For low-shift countries (FR, ES): trust absolute prediction more

    The router is a small MLP conditioned on (horizon_encoding,
    shift_mag, country_embedding).
    """

    def __init__(self, n_regions: int, fore_len: int, d_embed: int = 16):
        super().__init__()
        self.fore_len = fore_len
        self.d_embed = d_embed

        # Country embedding for routing
        self.country_embed = nn.Embedding(n_regions, d_embed)
        nn.init.zeros_(self.country_embed.weight)

        # Learnable horizon encoding: one per forecast horizon
        self.horizon_embed = nn.Embedding(fore_len, d_embed)
        nn.init.zeros_(self.horizon_embed.weight)

        # Router MLP: [horizon_embed, shift_scalar, country_embed] -> fore_len weights
        # Input dim = d_embed (horizon) + 1 (shift) + d_embed (country)
        self.router_mlp = nn.Sequential(
            nn.Linear(d_embed * 2 + 1, 32),
            nn.GELU(),
            nn.Linear(32, fore_len),
        )
        # Initialise bias to -1.0 so sigmoid(-1) ~ 0.27
        # This means the model starts by trusting absolute predictions more (weight ~0.73)
        # which is safe since V22+ absolute predictions are already good for most countries
        nn.init.zeros_(self.router_mlp[-1].weight)
        nn.init.constant_(self.router_mlp[-1].bias, -1.0)

    def forward(
        self,
        pred_absolute_calibrated: torch.Tensor,
        pred_rate_recovered: torch.Tensor,
        shift_mag: torch.Tensor,
        region_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse absolute and rate predictions with horizon-dependent routing.

        Parameters
        ----------
        pred_absolute_calibrated : [B, fore_len]  SATC-calibrated absolute pred
        pred_rate_recovered       : [B, fore_len]  RCPH rate pred recovered
        shift_mag                 : [B]  normalised shift magnitude
        region_id                 : [B]  country index

        Returns
        -------
        pred_fused : [B, fore_len]  final fused prediction
        weights    : [B, fore_len]  per-horizon absolute-space weights
        """
        B = pred_absolute_calibrated.size(0)

        # Get embeddings
        c_embed = self.country_embed(region_id)   # [B, d_embed]
        h_embed = self.horizon_embed.weight        # [fore_len, d_embed]
        h_embed_expanded = h_embed.unsqueeze(0).expand(B, -1, -1)  # [B, fore_len, d_embed]
        h_embed_pooled = h_embed.mean(dim=0).unsqueeze(0).expand(B, -1)  # [B, d_embed]

        # Router input: [horizon_embed_pooled, shift, country_embed]
        router_input = torch.cat([
            h_embed_pooled,              # [B, d_embed]
            shift_mag.unsqueeze(-1),     # [B, 1]
            c_embed,                     # [B, d_embed]
        ], dim=-1)                       # [B, 2*d_embed + 1]

        # Compute per-horizon weights
        logits = self.router_mlp(router_input)  # [B, fore_len]
        weights = torch.sigmoid(logits)         # [B, fore_len]

        # Fuse predictions
        pred_fused = (weights * pred_absolute_calibrated
                      + (1.0 - weights) * pred_rate_recovered)

        return pred_fused, weights


# ═══════════════════════════════════════════════════════════════════════════
# Main Model
# ═══════════════════════════════════════════════════════════════════════════

class CarbonSONetV24(nn.Module):
    """CarbonSONet V24: RCPH + SATC + DSF-HDR

    Architecture
    ────────────
    Input: [B, 168, 19] + region_id [B] + future_gen [B, 24, 9] (optional)
      -> Input projections + LayerNorm
      -> Sinusoidal PE + Solar 24h + Weekly 168h + Wind per-country PE
      -> V22+ MSSD: multi-scale spectral features
      -> Concat spectral features to time-domain representation
      -> x N { TransformerEncoderLayer -> Country-FiLM }
      -> Temporal mean pool -> [B, d_model]
      -> Shared backbone (Linear -> GELU) -> h [B, d_model//2]

    V22+ INNOVATIONS (retained):
      1. PICFP: physics-based CI prior
      2. MDIE:  per-country, per-source missing data embedding
      3. MSSD:  multi-scale spectral decomposition
      4. PCSH:  energy-bounded spectral head
      5. CCSR:  country-conditional spectral router

    V24 INNOVATIONS (new):
      6. RCPH:  rate-of-change prediction head
      7. SATC:  shift-adaptive transport calibration
      8. DSF-HDR: dual-space fusion with horizon-dependent routing

    Final prediction flow:
      pred_rate = RCPH(h_backbone)               # normalized rate
      pred_rate_real = CI_current + rate * window_std  # recover to real
      pred_abs = V22+ absolute prediction         # already in real space
      pred_abs_cal = SATC(pred_abs, shift, rid)   # transport calibration
      weights = DSF_HDR(shift, rid, horizon)      # per-horizon weights
      pred_final = w * pred_abs_cal + (1-w) * pred_rate_real  # fusion
    """

    def __init__(self, config, disabled_components=None):
        super().__init__()

        # -- ablation support
        self.disabled = set(disabled_components) if disabled_components else set()

        # -- hyper-parameters
        self.d_model        = getattr(config, 'd_model',       192)
        self.n_layers       = getattr(config, 'n_layers',      3)
        self.n_heads        = getattr(config, 'n_heads',       4)
        self.seq_len        = getattr(config, 'seq_len',       168)
        self.fore_len       = getattr(config, 'fore_len',      24)
        self.n_features     = getattr(config, 'n_features',    19)
        self.n_gen_features = getattr(config, 'n_gen_features', 9)
        self.n_regions      = getattr(config, 'n_regions',     16)
        self.dropout        = getattr(config, 'dropout',       0.25)
        self.lambda_physics = getattr(config, 'lambda_physics', LAMBDA_PHYSICS)
        self.lambda_spectral = getattr(config, 'lambda_spectral', LAMBDA_SPECTRAL)
        self.lambda_router   = getattr(config, 'lambda_router',   LAMBDA_ROUTER)
        self.lambda_rate     = getattr(config, 'lambda_rate',     LAMBDA_RATE)
        self.lambda_transport = getattr(config, 'lambda_transport', LAMBDA_TRANSPORT)

        d      = self.d_model
        d_half = d // 2

        # -- country normalization buffers
        self.register_buffer(
            'country_feat_mean',
            torch.zeros(self.n_regions, self.n_features),
        )
        self.register_buffer(
            'country_feat_std',
            torch.ones(self.n_regions, self.n_features),
        )
        self.register_buffer(
            'country_target_mean',
            torch.zeros(self.n_regions),
        )
        self.register_buffer(
            'country_target_std',
            torch.ones(self.n_regions),
        )
        self.register_buffer(
            'country_initialized',
            torch.zeros(self.n_regions, dtype=torch.bool),
        )

        # -- index tensors
        self.register_buffer(
            '_baseload_idx',
            torch.tensor(BASELOAD_INDICES, dtype=torch.long),
        )
        self.register_buffer(
            '_renewable_idx',
            torch.tensor(RENEWABLE_INDICES, dtype=torch.long),
        )
        self.register_buffer(
            '_fossil_idx',
            torch.tensor(FOSSIL_INDICES, dtype=torch.long),
        )
        self.register_buffer(
            '_gen_feature_indices',
            torch.tensor(
                list(range(GEN_FEATURE_START,
                           GEN_FEATURE_START + self.n_gen_features)),
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            '_emission_factors',
            torch.tensor(EMISSION_FACTORS[:self.n_gen_features],
                         dtype=torch.float32),
        )

        # -- input projection
        self.input_proj = nn.Linear(self.n_features, d)
        self.input_norm = nn.LayerNorm(d)

        # -- positional encoding
        self.pos_encoding = RenewableAwarePositionalEncoding(
            d, self.seq_len, self.n_regions, self.dropout,
        )

        # -- transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=self.n_heads,
                dim_feedforward=d * 4,
                dropout=self.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(self.n_layers)
        ])

        # -- Country-FiLM
        self.film_layers = nn.ModuleList([
            CountryFiLM(self.n_regions, d)
            for _ in range(self.n_layers)
        ])

        # -- shared backbone
        self.shared_backbone = nn.Sequential(
            nn.Linear(d, d_half),
            nn.GELU(),
        )

        # *** V22 INNOVATION 1: Physics-Informed Carbon Flow Prior ***
        self.picfp = PhysicsInformedCarbonFlowPrior(
            self.n_regions, self.fore_len, self.n_gen_features,
        )

        # *** V22 INNOVATION 2: Missing-Data Imputation Embedding ***
        self.mdie = MissingDataImputationEmbedding(
            self.n_regions, self.n_gen_features, MDIE_DIM,
        )

        # *** V22+ INNOVATION 3: Multi-Scale Spectral Decomposition ***
        self.mssd = MultiScaleSpectralDecomposition(
            self.n_regions, d, MSSD_N_PERIODS,
        )

        # *** V22+ INNOVATION 4: Physics-Constrained Spectral Head ***
        self.pcsh = PhysicsConstrainedSpectralHead(
            self.n_regions, d, self.fore_len,
        )

        # *** V22: FGCRH + MDIE -> 3 residual deltas ***
        self.causal_heads = FutureGenMDIEConditionedHeads(
            self.n_regions, d_half, self.fore_len,
            self.n_gen_features, MDIE_DIM,
        )

        # -- V22: per-country, per-horizon learnable residual
        self.physics_residual = nn.Embedding(self.n_regions, self.fore_len)
        nn.init.zeros_(self.physics_residual.weight)

        # -- HAPF
        self.hapf = HorizonAdaptivePersistenceFusion(
            self.n_regions, self.fore_len,
        )

        # -- generation prediction head (auxiliary)
        self.gen_head = nn.Linear(d_half, self.fore_len * self.n_gen_features)
        nn.init.zeros_(self.gen_head.weight)
        nn.init.zeros_(self.gen_head.bias)

        # -- MSSD feature injection projection
        self.mssd_inject = nn.Linear(d, d)
        nn.init.zeros_(self.mssd_inject.weight)
        nn.init.zeros_(self.mssd_inject.bias)

        # ═══════════════════════════════════════════════════════════════════
        # V24 INNOVATIONS
        # ═══════════════════════════════════════════════════════════════════

        # *** V24 INNOVATION 1: Rate-of-Change Prediction Head (RCPH) ***
        self.rcph = RateOfChangePredictionHead(
            self.n_regions, d_half, self.fore_len,
        )

        # *** V24 INNOVATION 2: Shift-Adaptive Transport Calibration (SATC) ***
        self.satc = ShiftAdaptiveTransportCalibration(
            self.n_regions, self.fore_len, d_embed=16,
        )

        # *** V24 INNOVATION 3: Dual-Space Fusion with HDR (DSF-HDR) ***
        self.dsf_hdr = DualSpaceFusionHDR(
            self.n_regions, self.fore_len, d_embed=16,
        )

        # -- default country-id -> code mapping
        self._default_id_to_code: Dict[int, str] = {
            i: c for i, c in enumerate(DEFAULT_COUNTRY_CODES[:self.n_regions])
        }

    # ──────────────────────────────────────────────────────────────────────
    # Country-stats interface
    # ──────────────────────────────────────────────────────────────────────

    def set_country_stats(
        self,
        country_id: int,
        target_mean: float,
        target_std: float,
        feat_mean: torch.Tensor,
        feat_std: torch.Tensor,
    ):
        with torch.no_grad():
            self.country_feat_mean[country_id].copy_(feat_mean)
            self.country_feat_std[country_id].copy_(feat_std)
            self.country_target_mean[country_id].fill_(target_mean)
            self.country_target_std[country_id].fill_(target_std)
            self.country_initialized[country_id].fill_(True)

    def set_source_importance(self, country_id: int, country_code: str):
        """Retained for API compatibility."""
        pass

    def _get_stats(
        self, region_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fm = self.country_feat_mean[region_id]
        fs = self.country_feat_std[region_id]
        tm = self.country_target_mean[region_id]
        ts = self.country_target_std[region_id]
        return fm, fs, tm, ts

    def _denorm_gen(
        self,
        gen_norm: torch.Tensor,
        feat_mean: torch.Tensor,
        feat_std: torch.Tensor,
    ) -> torch.Tensor:
        gen_mean = feat_mean[:, self._gen_feature_indices]
        gen_std  = feat_std[:, self._gen_feature_indices]
        return gen_norm * gen_std.unsqueeze(1) + gen_mean.unsqueeze(1)

    def _denorm_ci(
        self,
        ci_norm: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> torch.Tensor:
        return ci_norm * target_std.unsqueeze(-1) + target_mean.unsqueeze(-1)

    # ──────────────────────────────────────────────────────────────────────
    # Volatility initialisation helpers
    # ──────────────────────────────────────────────────────────────────────

    def init_film_from_volatility(
        self,
        country_id_to_code: Optional[Dict[int, str]] = None,
    ):
        mapping = country_id_to_code or self._default_id_to_code
        for film in self.film_layers:
            film.init_from_volatility(mapping)

    def init_hapf_from_volatility(
        self,
        country_id_to_code: Optional[Dict[int, str]] = None,
    ):
        mapping = country_id_to_code or self._default_id_to_code
        self.hapf.init_from_volatility(mapping)

    def init_mssd_from_volatility(
        self,
        country_id_to_code: Optional[Dict[int, str]] = None,
    ):
        mapping = country_id_to_code or self._default_id_to_code
        self.mssd.init_from_volatility(mapping)

    def init_pcsh_from_volatility(
        self,
        country_id_to_code: Optional[Dict[int, str]] = None,
    ):
        mapping = country_id_to_code or self._default_id_to_code
        self.pcsh.init_from_volatility(mapping)

    def init_all_from_volatility(
        self,
        country_id_to_code: Optional[Dict[int, str]] = None,
    ):
        self.init_film_from_volatility(country_id_to_code)
        self.init_hapf_from_volatility(country_id_to_code)
        self.init_mssd_from_volatility(country_id_to_code)
        self.init_pcsh_from_volatility(country_id_to_code)

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        region_id: torch.Tensor,
        future_gen: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = x.size(0)

        feat_mean, feat_std, target_mean, target_std = self._get_stats(region_id)

        # -- input projection + layer norm
        h = self.input_norm(self.input_proj(x))

        # -- positional encoding
        h = self.pos_encoding(h, region_id)

        # *** V22+ MSSD ***
        x_ci_norm = x[:, :, CI_FEATURE_IDX]
        x_ci_real = self._denorm_ci(x_ci_norm, target_mean, target_std)
        if 'MSSD' not in self.disabled:
            spectral_features, router_weights = self.mssd(x_ci_real, region_id)
        else:
            spectral_features = torch.zeros(B, self.d_model, device=x.device)
            router_weights = torch.ones(B, MSSD_N_PERIODS, device=x.device) / MSSD_N_PERIODS

        # Inject spectral summary
        if 'MSSD' not in self.disabled:
            h = h + self.mssd_inject(spectral_features).unsqueeze(1)

        # CCSR override: use uniform weights when disabled
        if 'CCSR' in self.disabled:
            router_weights = torch.ones(B, MSSD_N_PERIODS, device=x.device) / MSSD_N_PERIODS

        # -- transformer + country-FiLM
        for transformer, film in zip(self.transformer_layers, self.film_layers):
            h = transformer(h)
            if 'FiLM' not in self.disabled:
                h = film(h, region_id)

        # -- temporal mean pool
        h_pool = h.mean(dim=1)

        # -- shared backbone
        h_backbone = self.shared_backbone(h_pool)

        # =====================================================================
        # SHARED CONTEXT
        # =====================================================================

        gen_current_norm = x[:, -1, self._gen_feature_indices]
        gen_current_real = self._denorm_gen(
            gen_current_norm.unsqueeze(1), feat_mean, feat_std,
        ).squeeze(1)

        ci_last_norm = x[:, -1, CI_FEATURE_IDX]
        ci_current_real = ci_last_norm * target_std + target_mean
        ci_current_expanded = ci_current_real.unsqueeze(1).expand(B, self.fore_len)

        future_gen_real = None
        if future_gen is not None:
            future_gen_real = self._denorm_gen(
                future_gen, feat_mean, feat_std,
            )

        mdie = None
        if future_gen_real is not None:
            if 'MDIE' not in self.disabled:
                mdie = self.mdie(future_gen_real, region_id)
            else:
                mdie = torch.zeros(future_gen_real.size(0), self.n_gen_features * MDIE_DIM, device=x.device)

        # =====================================================================
        # V22+ ABSOLUTE PREDICTION PATH
        # =====================================================================

        # PICFP
        if 'PICFP' not in self.disabled:
            ci_phys = self.picfp(future_gen_real, region_id, ci_current_expanded)
        else:
            ci_phys = ci_current_expanded

        # PCSH
        if 'PCSH' not in self.disabled:
            delta_spec, spec_gate, spec_energy = self.pcsh(spectral_features, region_id)
        else:
            delta_spec = torch.zeros(B, self.fore_len, device=x.device)
            spec_gate = torch.zeros(B, self.fore_len, device=x.device)
            spec_energy = torch.ones(B, device=x.device)

        # Causal decomposition heads
        hour_sin_val = x[:, -1, 10].detach().cpu().numpy()
        hour_cos_val = x[:, -1, 11].detach().cpu().numpy()
        current_hour = int(
            (np.arctan2(hour_sin_val.mean(), hour_cos_val.mean())
             / (2.0 * math.pi) * 24.0 + 24.0) % 24.0
        )

        if 'FGCRH' not in self.disabled:
            delta_baseload, delta_renewable, delta_fossil = \
            self.causal_heads(h_backbone, region_id, current_hour,
            future_gen=future_gen, mdie=mdie)
        else:
            # Simple linear head fallback
            if not hasattr(self, '_fgcrh_fallback'):
                d_half = self.d_model // 2
                self._fgcrh_fallback = nn.Linear(d_half, self.fore_len * 3).to(x.device)
                nn.init.zeros_(self._fgcrh_fallback.weight)
                nn.init.zeros_(self._fgcrh_fallback.bias)
            fallback_out = self._fgcrh_fallback(h_backbone)
            delta_baseload = fallback_out[:, :self.fore_len] * 0.5
            delta_renewable = fallback_out[:, self.fore_len:2*self.fore_len] * 1.0
            delta_fossil = fallback_out[:, 2*self.fore_len:] * 0.8

        ci_model = ci_current_expanded + delta_baseload + delta_renewable + delta_fossil

        # Physics residual
        physics_residual = self.physics_residual(region_id)
        ci_pred = ci_phys + physics_residual + delta_spec

        # HAPF
        ci_persistence = ci_current_expanded
        ci_volatility = (
            (x[:, :, CI_FEATURE_IDX] * target_std.unsqueeze(1)
             + target_mean.unsqueeze(1)).std(dim=1)
        )

        if 'HAPF' not in self.disabled:
            point_pred_abs, gamma = self.hapf(ci_pred, ci_persistence, region_id)
        else:
            point_pred_abs = ci_pred
            gamma = torch.ones(B, self.fore_len, device=x.device)

        # =====================================================================
        # V24 INNOVATION 1: RATE-OF-CHANGE PREDICTION (RCPH)
        # =====================================================================

        # Compute window statistics from input CI history (in real space)
        # These are available at test time and reflect the actual level
        window_std = x_ci_real.std(dim=1).clamp(min=1.0)   # [B]  std of CI in real units (clamp≥1 for numerical safety)
        window_mean = x_ci_real.mean(dim=1)  # [B]  mean of CI in real units

        # Predict normalized rate of change
        if 'RCPH' not in self.disabled:
            rate_pred = self.rcph(h_backbone, region_id)   # [B, fore_len]
        else:
            rate_pred = torch.zeros(B, self.fore_len, device=x.device)

        # Recover rate prediction to real space:
        # CI_pred = CI_current + rate * window_std
        pred_rate_real = ci_current_expanded + rate_pred * window_std.unsqueeze(-1)

        # Compute actual rate for loss computation
        # actual_rate = (CI_target - CI_current) / (window_std + eps)
        # (returned for external loss computation; target not available here)

        # =====================================================================
        # V24 INNOVATION 2: SHIFT-ADAPTIVE TRANSPORT CALIBRATION (SATC)
        # =====================================================================

        # Compute shift magnitude from window statistics
        # shift = (window_mean - train_mean) / train_std
        shift_mag = (window_mean - target_mean) / target_std.clamp(min=1e-6)  # [B]

        # Apply transport calibration to absolute prediction
        if 'SATC' not in self.disabled:
            pred_abs_calibrated, satc_calibration = self.satc(
            point_pred_abs, shift_mag, region_id,
            )
        else:
            pred_abs_calibrated = point_pred_abs
            satc_calibration = torch.zeros(B, self.fore_len, device=x.device)

        # =====================================================================
        # V24 INNOVATION 3: DUAL-SPACE FUSION WITH HDR (DSF-HDR)
        # =====================================================================

        if 'DSF_HDR' not in self.disabled:
            pred_final, dsf_weights = self.dsf_hdr(
            pred_abs_calibrated,   # absolute prediction, transport-calibrated
            pred_rate_real,        # rate prediction, recovered to real space
            shift_mag,             # normalised shift magnitude
            region_id,             # country index
            )
        else:
            pred_final = pred_abs_calibrated
            dsf_weights = torch.ones(B, self.fore_len, device=x.device)

        # -- generation prediction (auxiliary)
        gen_delta = self.gen_head(h_backbone)
        gen_delta = gen_delta.view(B, self.fore_len, self.n_gen_features)
        gen_pred_norm = gen_current_norm.unsqueeze(1) + gen_delta
        gen_pred_real = self._denorm_gen(gen_pred_norm, feat_mean, feat_std)

        return {
            # Final V24 prediction
            'point_pred':           pred_final,

            # V24 intermediate predictions
            'pred_rate_real':       pred_rate_real,
            'pred_abs_calibrated':  pred_abs_calibrated,
            'rate_pred':            rate_pred,
            'shift_mag':            shift_mag,
            'satc_calibration':     satc_calibration,
            'dsf_weights':          dsf_weights,

            # V22+ absolute prediction (before SATC)
            'point_pred_abs':       point_pred_abs,

            # V22+ internals
            'ci_model':             ci_model,
            'ci_persistence':       ci_persistence,
            'ci_phys':              ci_phys,
            'gamma':                gamma,
            'delta_baseload':       delta_baseload,
            'delta_renewable':      delta_renewable,
            'delta_fossil':         delta_fossil,
            'physics_residual':     physics_residual,
            'delta_spec':           delta_spec,
            'spec_gate':            spec_gate,
            'spec_energy':          spec_energy,
            'router_weights':       router_weights,
            'ci_volatility':        ci_volatility,
            'gen_pred_real':        gen_pred_real,

            # V24 window statistics (for loss computation)
            'window_std':           window_std,
            'window_mean':          window_mean,
            'ci_current_real':      ci_current_real,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import types

    config = types.SimpleNamespace(
        d_model=192,
        n_layers=3,
        n_heads=4,
        seq_len=168,
        fore_len=24,
        n_features=19,
        n_gen_features=9,
        n_regions=16,
        dropout=0.25,
        lambda_physics=LAMBDA_PHYSICS,
        lambda_spectral=LAMBDA_SPECTRAL,
        lambda_router=LAMBDA_ROUTER,
        lambda_rate=LAMBDA_RATE,
        lambda_transport=LAMBDA_TRANSPORT,
    )

    torch.manual_seed(42)
    device = torch.device('cpu')

    model = CarbonSONetV24(config).to(device)
    model.eval()

    for cid in range(config.n_regions):
        model.set_country_stats(
            country_id=cid,
            target_mean=200.0 + cid * 10.0,
            target_std=50.0,
            feat_mean=torch.randn(config.n_features) * 100.0,
            feat_std=torch.ones(config.n_features) * 20.0,
        )

    model.init_all_from_volatility()

    B = 4
    x = torch.randn(B, config.seq_len, config.n_features)
    region_id = torch.randint(0, config.n_regions, (B,))

    # -- Test 1: forward WITHOUT future_gen
    with torch.no_grad():
        out_no_fg = model(x, region_id, future_gen=None)

    # -- Test 2: forward WITH future_gen
    future_gen = torch.randn(B, config.fore_len, config.n_gen_features)
    with torch.no_grad():
        out_with_fg = model(x, region_id, future_gen=future_gen)

    expected_shapes = {
        'point_pred':          (B, config.fore_len),
        'pred_rate_real':      (B, config.fore_len),
        'pred_abs_calibrated': (B, config.fore_len),
        'rate_pred':           (B, config.fore_len),
        'shift_mag':           (B,),
        'satc_calibration':    (B, config.fore_len),
        'dsf_weights':         (B, config.fore_len),
        'point_pred_abs':      (B, config.fore_len),
        'ci_model':            (B, config.fore_len),
        'ci_persistence':      (B, config.fore_len),
        'ci_phys':             (B, config.fore_len),
        'gamma':               (B, config.fore_len),
        'delta_baseload':      (B, config.fore_len),
        'delta_renewable':     (B, config.fore_len),
        'delta_fossil':        (B, config.fore_len),
        'physics_residual':    (B, config.fore_len),
        'delta_spec':          (B, config.fore_len),
        'spec_gate':           (B, config.fore_len),
        'spec_energy':         (B,),
        'router_weights':      (B, MSSD_N_PERIODS),
        'ci_volatility':       (B,),
        'gen_pred_real':       (B, config.fore_len, config.n_gen_features),
        'window_std':          (B,),
        'window_mean':         (B,),
        'ci_current_real':     (B,),
    }

    print("=" * 70)
    print("CarbonSONet V24 -- Forward Pass Verification")
    print("=" * 70)

    all_ok = True
    for key, expected in expected_shapes.items():
        for label, out in [("no_fg", out_no_fg), ("with_fg", out_with_fg)]:
            actual = tuple(out[key].shape)
            ok = actual == expected
            all_ok = all_ok and ok
            status = "OK" if ok else "MISMATCH"
            print(f"  [{label}] {key:25s}  expected {expected}  actual {actual}  [{status}]")

    print("-" * 70)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")
    print(f"  All shapes correct: {all_ok}")

    # -- verify V24 innovations exist
    print("-" * 70)
    has_rcph    = hasattr(model, 'rcph')
    has_satc    = hasattr(model, 'satc')
    has_dsf_hdr = hasattr(model, 'dsf_hdr')
    has_picfp   = hasattr(model, 'picfp')
    has_mssd    = hasattr(model, 'mssd')
    has_pcsh    = hasattr(model, 'pcsh')
    print(f"  RCPH   module exists: {has_rcph}    (V24 INNOVATION 1)")
    print(f"  SATC   module exists: {has_satc}    (V24 INNOVATION 2)")
    print(f"  DSF-HDR module exists: {has_dsf_hdr}  (V24 INNOVATION 3)")
    print(f"  PICFP  module exists: {has_picfp}    (V22 retained)")
    print(f"  MSSD   module exists: {has_mssd}    (V22+ retained)")
    print(f"  PCSH   module exists: {has_pcsh}    (V22+ retained)")

    # -- verify V24 innovations have effect
    print("-" * 70)
    # DSF weights should be between 0 and 1
    w = out_with_fg['dsf_weights']
    print(f"  DSF weights: min={w.min().item():.4f}  max={w.max().item():.4f}  "
          f"mean={w.mean().item():.4f}  (should be in [0,1])")
    # Shift magnitude
    s = out_with_fg['shift_mag']
    print(f"  Shift magnitude: min={s.min().item():.4f}  max={s.max().item():.4f}  "
          f"mean={s.mean().item():.4f}")
    # Rate prediction
    r = out_with_fg['rate_pred']
    print(f"  Rate prediction: min={r.min().item():.4f}  max={r.max().item():.4f}  "
          f"mean={r.mean().item():.4f}")
    # SATC calibration
    c = out_with_fg['satc_calibration']
    print(f"  SATC calibration: min={c.min().item():.4f}  max={c.max().item():.4f}  "
          f"mean={c.mean().item():.4f}  (should be ~0 at init)")
    # Final vs absolute
    diff = (out_with_fg['point_pred'] - out_with_fg['point_pred_abs']).abs().max().item()
    print(f"  point_pred vs point_pred_abs max diff: {diff:.6f}")

    # -- check router weights
    rw = out_with_fg['router_weights']
    print(f"  router_weights: shape={tuple(rw.shape)}, "
          f"sum_per_row~1.0: {rw.sum(dim=-1).mean().item():.4f}")

    print("=" * 70)

    if (all_ok and has_rcph and has_satc and has_dsf_hdr
            and has_picfp and has_mssd and has_pcsh):
        print("SUCCESS: V24 RCPH + SATC + DSF-HDR architecture verified.")
    else:
        print("FAILURE: V24 architecture verification failed!")
