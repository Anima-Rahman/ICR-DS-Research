#!/usr/bin/env python3
"""STAT: Spectro-temporal Axial Transformer for in-vehicle radar heart rate estimation.

Training implementation of the proposed STAT method on ICR-DS. The model
processes 36-s FMCW radar windows with an 18-s hop, represented by 15 radar
cells and 136 heart-rate frequency bins (45–180 bpm). A radar-only, local
three-window 75th-percentile background estimate is applied before neural
enhancement, using 0.18-Hz Gaussian smoothing and strength 0.90.

STAT uses two factorised axial Transformer blocks, applying multi-head
self-attention first along the frequency axis and then along the temporal
axis, independently for each radar cell. The network predicts a bounded
frequency-dependent attenuation mask rather than directly regressing scalar
heart rate. Its three-window context spans 72 s. Predictions from overlapping
blocks are averaged before robust spectral scoring and full-recording Viterbi
decoding.

Training combines blockwise forward-backward expected heart-rate risk and
interpolated heart-rate-bin negative log-likelihood with a cardiac-versus-
wrong-peak spectral contrast objective, heart-rate-region preservation and
mask regularisation. BioHarness IBI-derived heart-rate references provide
supervision and are not used as network inputs.

Expected input: one {TRAIN_PT_DIR}/sXXX_session.pt file per participant.
Each file contains:
    x: normalized full radar spectra, float array [N, 15, 1438]
    y: aligned BioHarness reference heart rate, float array [N]
The spectra must use the STAT FFT frequency grid and 36-s windows with an
18-s hop.

Edit DATA_ROOT to the location of ICR-DS.
Run: python train_STAT.py. Only TRAIN data are loaded; the final
model is saved after 100 epochs, without validation or early stopping.
"""
from __future__ import annotations

import csv
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Model: frequency attention, then temporal attention; no cell/spatial MHSA.
# ---------------------------------------------------------------------------
class AxisTransformerUnit(nn.Module):
    def __init__(self, width: int, nhead: int, feedforward: int):
        super().__init__()
        self.attn_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            width, nhead, dropout=0.0, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, feedforward),
            nn.GELU(),
            nn.Linear(feedforward, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.attn_norm(x)
        attended, _ = self.attn(q, q, q, need_weights=False)
        x = x + attended
        return x + self.ff(self.ff_norm(x))


class FrequencyTemporalBlock(nn.Module):
    def __init__(
        self,
        width: int,
        nhead: int,
        feedforward: int,
        block_windows: int = 3,
        cells: int = 15,
        bins: int = 136,
    ):
        super().__init__()
        self.block_windows = int(block_windows)
        self.cells = int(cells)
        self.bins = int(bins)
        self.frequency = AxisTransformerUnit(width, nhead, feedforward)
        self.temporal = AxisTransformerUnit(width, nhead, feedforward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k, c, b, d = x.shape
        if (k, c, b) != (self.block_windows, self.cells, self.bins):
            raise ValueError(f"Unexpected feature shape: {tuple(x.shape)}")

        # One frequency sequence per (window, radar cell).
        x = self.frequency(x.reshape(k * c, b, d)).reshape(k, c, b, d)

        # One temporal sequence per (radar cell, frequency bin).
        x = x.permute(1, 2, 0, 3).contiguous().reshape(c * b, k, d)
        x = self.temporal(x)
        return x.reshape(c, b, k, d).permute(2, 0, 1, 3).contiguous()


class FrequencyTemporalPeakMask(nn.Module):
    def __init__(
        self,
        input_scale: float = 8.0,
        max_attenuation: float = 1.25,
        width: int = 32,
        block_windows: int = 3,
        initial_mask_logit: float = -3.0,
        nhead: int = 4,
        num_layers: int = 2,
        feedforward: int = 96,
    ):
        super().__init__()
        if (
            block_windows < 3
            or block_windows % 2 == 0
            or width % nhead != 0
            or width < 8
        ):
            raise ValueError(
                "Require an odd context length >= 3 and width divisible by nhead"
            )
        if input_scale <= 0 or max_attenuation <= 0 or num_layers < 1:
            raise ValueError("Invalid model configuration")

        self.block_windows = int(block_windows)
        self.input_scale = float(input_scale)
        self.max_attenuation = float(max_attenuation)
        self.scalar_embed = nn.Sequential(
            nn.Linear(1, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.time_position = nn.Parameter(torch.empty(self.block_windows, width))
        self.cell_position = nn.Parameter(torch.empty(15, width))
        self.frequency_position = nn.Parameter(torch.empty(136, width))
        nn.init.normal_(self.time_position, mean=0.0, std=0.02)
        nn.init.normal_(self.cell_position, mean=0.0, std=0.02)
        nn.init.normal_(self.frequency_position, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList([
            FrequencyTemporalBlock(
                width, nhead, feedforward,
                block_windows=self.block_windows, cells=15, bins=136,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 1)
        # A constant initial mask preserves the robust-score evidence/path at epoch 0.
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, initial_mask_logit)

    def forward(self, y: torch.Tensor) -> dict[str, torch.Tensor]:
        if y.ndim != 3 or tuple(y.shape[1:]) != (15, 136):
            raise ValueError(f"Expected [T,15,136], got {tuple(y.shape)}")
        t = len(y)
        if not 1 <= t <= self.block_windows:
            raise ValueError(f"Expected between 1 and {self.block_windows} windows")
        if not bool(torch.isfinite(y).all()):
            raise ValueError("Nonfinite radar input")

        # Short blocks are padded only for the Transformer; padded outputs are discarded.
        if t < self.block_windows:
            y_padded = torch.cat(
                [y, y[-1:].expand(self.block_windows - t, -1, -1)], dim=0
            )
        else:
            y_padded = y

        h = self.scalar_embed((y_padded / self.input_scale).unsqueeze(-1))
        h = (
            h
            + self.time_position[:, None, None, :]
            + self.cell_position[None, :, None, :]
            + self.frequency_position[None, None, :, :]
        )
        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)
        logits = self.output(h).squeeze(-1)[:t]
        attenuation = self.max_attenuation * torch.sigmoid(logits)
        enhanced = y - attenuation
        return {
            "enhanced": enhanced,
            "attenuation": attenuation,
            "fused": enhanced.mean(dim=1),
            "input_fused": y.mean(dim=1),
        }


# ---------------------------------------------------------------------------
# Data paths and K=3 STAT configuration.
# ---------------------------------------------------------------------------
DATA_ROOT = Path("path/to/ICR-DS")  # EDIT THIS
TRAIN_PT_DIR = DATA_ROOT / "Train" / "STAT"
OUTPUT_DIR = DATA_ROOT / "results" / "STAT"

# Main-results windowing: 36-s windows, 18-s hop.
WINDOW_SECONDS = 36.0
WINDOW_HOP_SECONDS = 18.0

EPOCHS = 100
SEED = 2031
CPU_THREADS = 4

RADAR_BACKGROUND_WINDOWS = 3
RADAR_BACKGROUND_QUANTILE = 0.75
RADAR_BACKGROUND_SMOOTH_HZ = 0.18
RADAR_BACKGROUND_STRENGTH = 0.90

FB_BLOCK_WINDOWS = 3
FB_BLOCK_STRIDE = 2
HEAD_WIDTH = 32
ATTENTION_HEADS = 4
ATTENTION_LAYERS = 2
ATTENTION_FF = 96
INPUT_SCALE = 8.0
MAX_ATTENUATION = 1.25
INITIAL_MASK_LOGIT = -3.0
HEAD_LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

TRACKER_LAMBDA = 8.0
TRACKER_MAX_CHANGE_BPM = 30.0
BPM_RISK_WEIGHT = 0.10
FB_NLL_WEIGHT = 0.50
CONTRAST_WEIGHT = 0.20
CONTRAST_MARGIN = 0.50
CONTRAST_IMPROVEMENT = 0.25
CONTRAST_SOFTPLUS_TEMP = 0.50
WRONG_PEAK_SOFTMAX_TEMP = 0.50
HR_REGION_SIGMA_BPM = 1.0
HR_REGION_HALF_WIDTH_BPM = 2.5
WRONG_EXCLUSION_BPM = 3.0
HR_REGION_ATTENUATION_WEIGHT = 0.10
ATTENUATION_SIZE_WEIGHT = 0.05
ATTENUATION_TIME_WEIGHT = 0.02
ATTENUATION_FREQ_WEIGHT = 0.02
PRINT_PER_PERSON_EVERY = 5

# The 21 participants in the original STAT training cohort.
# Set to None only when deliberately adapting the guide to another cohort.
EXPECTED_TRAIN_PARTICIPANTS = frozenset({
    "s1", "s2", "s4", "s6", "s7", "s12", "s13", "s14", "s15",
    "s16", "s17", "s19", "s20", "s21", "s22", "s23", "s24",
    "s25", "s26", "s27", "s28",
})

BPM = np.arange(45, 181, dtype=np.float32)
F_ALL = np.fft.rfftfreq(4096, d=0.09)
F_FULL = F_ALL[(F_ALL >= 0.1) & (F_ALL <= 4.0)]
HR_HZ = BPM.astype(np.float64) / 60.0
INTERP_RIGHT = np.clip(np.searchsorted(F_FULL, HR_HZ), 1, len(F_FULL) - 1)
INTERP_LEFT = INTERP_RIGHT - 1
INTERP_WEIGHT = (
    (HR_HZ - F_FULL[INTERP_LEFT])
    / (F_FULL[INTERP_RIGHT] - F_FULL[INTERP_LEFT])
)


@dataclass
class Recording:
    name: str
    x: torch.Tensor
    correction: torch.Tensor
    corrected: torch.Tensor
    hr: np.ndarray
    low: torch.Tensor
    high: torch.Tensor
    upper: torch.Tensor
    valid: torch.Tensor
    corrected_mae: float
    raw_mae: float

    @property
    def n(self) -> int:
        return len(self.x)


def interpolate_to_bpm(full_spectra: np.ndarray) -> np.ndarray:
    return (
        full_spectra[..., INTERP_LEFT] * (1.0 - INTERP_WEIGHT)
        + full_spectra[..., INTERP_RIGHT] * INTERP_WEIGHT
    ).astype(np.float32)


def windowed_radar_correction(full_spectra: np.ndarray) -> np.ndarray:
    if (
        full_spectra.ndim != 3
        or full_spectra.shape[1] != 15
        or full_spectra.shape[2] != len(F_FULL)
        or len(full_spectra) == 0
    ):
        raise ValueError(
            f"Expected [N,15,{len(F_FULL)}] full spectra, got {full_spectra.shape}"
        )
    if not np.isfinite(full_spectra).all():
        raise ValueError("Nonfinite spectrum in input file")

    n = len(full_spectra)
    context = min(RADAR_BACKGROUND_WINDOWS, n)
    starts = np.clip(np.arange(n) - context // 2, 0, n - context)
    correction = np.empty((n, 15, len(BPM)), dtype=np.float32)
    sigma = RADAR_BACKGROUND_SMOOTH_HZ / np.median(np.diff(F_FULL))

    for start in np.unique(starts):
        baseline = np.quantile(
            full_spectra[start:start + context],
            RADAR_BACKGROUND_QUANTILE,
            axis=0,
        )
        baseline = gaussian_filter1d(baseline, sigma=sigma, axis=-1)
        at_hr = interpolate_to_bpm(baseline[None])[0]
        correction[starts == start] = (
            RADAR_BACKGROUND_STRENGTH * at_hr
        ).astype(np.float32)

    return correction


def robust_score_np(fused: np.ndarray) -> np.ndarray:
    centered = fused - np.median(fused, axis=-1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(centered), axis=-1, keepdims=True)
    return np.clip(centered / np.maximum(mad, 0.5), -6.0, 6.0).astype(np.float32)


def true_torch_median(x: torch.Tensor) -> torch.Tensor:
    ordered = x.sort(dim=-1).values
    k = x.shape[-1]
    if k % 2:
        return ordered[..., k // 2:k // 2 + 1]
    return 0.5 * (
        ordered[..., k // 2 - 1:k // 2]
        + ordered[..., k // 2:k // 2 + 1]
    )


def robust_score_torch(fused: torch.Tensor) -> torch.Tensor:
    centered = fused - true_torch_median(fused)
    mad = 1.4826 * true_torch_median(centered.abs())
    return (centered / mad.clamp(min=0.5)).clamp(-6.0, 6.0)


def penalty_np() -> np.ndarray:
    delta = np.abs(BPM[:, None] - BPM[None, :])
    penalties = TRACKER_LAMBDA * (delta / 5.0) ** 2
    penalties[delta > TRACKER_MAX_CHANGE_BPM] = np.inf
    return penalties.astype(np.float32)


def viterbi_np(evidence: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    n, nb = evidence.shape
    dp = np.empty((n, nb), dtype=np.float32)
    back = np.empty((n, nb), dtype=np.int16)
    dp[0] = evidence[0]
    choices = np.arange(nb)
    for t in range(1, n):
        scores = dp[t - 1][None, :] - penalty
        back[t] = scores.argmax(axis=1)
        dp[t] = evidence[t] + scores[choices, back[t]]
    path = np.empty(n, dtype=np.int16)
    path[-1] = dp[-1].argmax()
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return BPM[path]


def transition_torch(device: torch.device) -> torch.Tensor:
    bpm = torch.arange(45.0, 181.0, device=device)
    delta = (bpm[:, None] - bpm[None, :]).abs()
    penalty = TRACKER_LAMBDA * (delta / 5.0) ** 2
    return -penalty.masked_fill(delta > TRACKER_MAX_CHANGE_BPM, float("inf"))


def fb_logposterior(
    evidence: torch.Tensor,
    transition: torch.Tensor,
) -> torch.Tensor:
    alpha = [evidence[0]]
    for t in range(1, len(evidence)):
        alpha.append(evidence[t] + torch.logsumexp(
            alpha[-1][:, None] + transition, dim=0
        ))
    beta = [torch.zeros_like(evidence[-1])]
    for t in range(len(evidence) - 2, -1, -1):
        beta.append(torch.logsumexp(
            transition + evidence[t + 1][None, :] + beta[-1][None, :],
            dim=1,
        ))
    beta.reverse()
    return F.log_softmax(torch.stack(
        [a + b for a, b in zip(alpha, beta)], dim=0
    ), dim=-1)


def hr_targets(hr: np.ndarray):
    valid = np.isfinite(hr) & (hr >= 45.0) & (hr <= 180.0)
    safe = np.where(valid, hr, 45.0).astype(np.float32)
    low = np.floor(safe).astype(np.int64) - 45
    high = np.minimum(low + 1, 135)
    upper = np.where(high > low, safe - np.floor(safe), 0.0).astype(np.float32)
    return (
        torch.from_numpy(low),
        torch.from_numpy(high),
        torch.from_numpy(upper),
        torch.from_numpy(valid),
    )


def load_train_cohort() -> list[Recording]:
    if not TRAIN_PT_DIR.is_dir():
        raise FileNotFoundError(f"Training .pt directory not found: {TRAIN_PT_DIR}")

    paths = sorted(TRAIN_PT_DIR.glob("s*_session.pt"))
    if not paths:
        raise FileNotFoundError(f"No s*_session.pt files found in {TRAIN_PT_DIR}")

    people: list[Recording] = []
    seen: set[str] = set()
    penalty = penalty_np()

    for path in paths:
        match = re.fullmatch(r"s(\d+)_session\.pt", path.name, flags=re.IGNORECASE)
        if not match:
            continue
        participant = f"s{int(match.group(1))}"
        if participant in seen:
            raise ValueError(f"Duplicate participant file: {participant}")
        seen.add(participant)

        try:
            sample = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            sample = torch.load(path, map_location="cpu")

        if not isinstance(sample, dict) or "x" not in sample or "y" not in sample:
            raise ValueError(f"Expected a dictionary containing x and y in {path}")

        spectra = sample["x"]
        labels = sample["y"]
        if isinstance(spectra, torch.Tensor):
            spectra = spectra.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        spectra = np.asarray(spectra, dtype=np.float32)
        hr = np.asarray(labels, dtype=np.float32).reshape(-1)

        if (
            spectra.ndim != 3
            or spectra.shape[1] != 15
            or spectra.shape[2] != len(F_FULL)
            or len(spectra) != len(hr)
            or len(spectra) < 4
        ):
            raise ValueError(
                f"Unexpected data in {path}: x={spectra.shape}, y={hr.shape}; "
                f"expected x=[N,15,{len(F_FULL)}], y=[N]"
            )
        if not np.isfinite(spectra).all():
            raise ValueError(f"Nonfinite radar spectra in {path}")

        x = interpolate_to_bpm(spectra)
        correction = windowed_radar_correction(spectra)
        corrected = x - correction
        low, high, upper, valid = hr_targets(hr)
        if not bool(valid.any()):
            raise ValueError(f"No valid heart-rate labels in {path}")

        raw_path = viterbi_np(robust_score_np(x.mean(axis=1)), penalty)
        corrected_path = viterbi_np(
            robust_score_np(corrected.mean(axis=1)), penalty
        )
        valid_np = valid.numpy()
        raw_mae = float(np.abs(raw_path[valid_np] - hr[valid_np]).mean())
        corrected_mae = float(
            np.abs(corrected_path[valid_np] - hr[valid_np]).mean()
        )

        people.append(Recording(
            name=participant,
            x=torch.from_numpy(x),
            correction=torch.from_numpy(correction),
            corrected=torch.from_numpy(corrected),
            hr=hr,
            low=low,
            high=high,
            upper=upper,
            valid=valid,
            corrected_mae=corrected_mae,
            raw_mae=raw_mae,
        ))

    if not people:
        raise RuntimeError(f"No valid participant .pt files found in {TRAIN_PT_DIR}")

    if EXPECTED_TRAIN_PARTICIPANTS is not None:
        actual = {rec.name for rec in people}
        if actual != set(EXPECTED_TRAIN_PARTICIPANTS):
            raise RuntimeError(
                "Training participant set does not match the original 21-person "
                "STAT training cohort.\n"
                f"Missing: {sorted(set(EXPECTED_TRAIN_PARTICIPANTS) - actual)}\n"
                f"Unexpected: {sorted(actual - set(EXPECTED_TRAIN_PARTICIPANTS))}\n"
                "Edit EXPECTED_TRAIN_PARTICIPANTS only if you intentionally use "
                "a different cohort."
            )

    return people


def cardiac_contrast(
    evidence: torch.Tensor,
    hr: torch.Tensor,
    valid: torch.Tensor,
):
    selected = evidence[valid]
    truth = hr[valid]
    grid = torch.arange(
        45.0, 181.0, dtype=selected.dtype, device=selected.device
    )
    dist = (grid[None, :] - truth[:, None]).abs()
    good_band = dist <= HR_REGION_HALF_WIDTH_BPM
    gauss = torch.exp(-0.5 * (dist / HR_REGION_SIGMA_BPM) ** 2)
    good_weights = gauss * good_band.to(dtype=selected.dtype)
    good_weights = good_weights / good_weights.sum(
        dim=-1, keepdim=True
    ).clamp(min=1e-12)
    good_score = (selected * good_weights).sum(dim=-1)
    wrong_band = dist > WRONG_EXCLUSION_BPM
    wrong_logits = selected.masked_fill(~wrong_band, -1e4)
    wrong_weights = F.softmax(
        wrong_logits / WRONG_PEAK_SOFTMAX_TEMP, dim=-1
    )
    wrong_score = (wrong_weights * selected).sum(dim=-1)
    return good_score - wrong_score, good_weights, wrong_score


def training_blocks(n: int) -> list[tuple[int, int]]:
    if n < 4:
        raise ValueError(f"Need at least four radar windows, got {n}")
    starts = list(range(0, n - FB_BLOCK_WINDOWS + 1, FB_BLOCK_STRIDE))
    if starts[-1] != n - FB_BLOCK_WINDOWS:
        starts.append(n - FB_BLOCK_WINDOWS)
    return [(start, start + FB_BLOCK_WINDOWS) for start in starts]


def loss_for_block(
    model,
    rec: Recording,
    transition: torch.Tensor,
    device: torch.device,
    start: int,
    stop: int,
):
    if not (0 <= start < stop <= rec.n and stop - start <= FB_BLOCK_WINDOWS):
        raise ValueError(f"Invalid block {start}:{stop} for {rec.name}")

    out = model(rec.corrected[start:stop].to(device))
    attenuation = out["attenuation"]
    evidence = robust_score_torch(out["fused"])
    valid = rec.valid[start:stop].to(device)
    if not bool(valid.any()):
        return None

    logp = fb_logposterior(evidence, transition)
    posterior = logp.exp()
    indices = torch.arange(stop - start, device=device)
    truth = torch.as_tensor(
        rec.hr[start:stop], device=device, dtype=evidence.dtype
    )
    safe_truth = torch.where(valid, truth, torch.full_like(truth, 45.0))
    bpm_grid = torch.arange(45.0, 181.0, device=device, dtype=evidence.dtype)
    risk = (
        posterior * (bpm_grid[None, :] - safe_truth[:, None]).abs()
    ).sum(dim=-1)[valid].mean()

    low = rec.low[start:stop].to(device)
    high = rec.high[start:stop].to(device)
    upper = rec.upper[start:stop].to(device)
    nll = -(
        (1.0 - upper) * logp[indices, low]
        + upper * logp[indices, high]
    )[valid].mean()

    new_margin, good_weights, _ = cardiac_contrast(
        evidence, safe_truth, valid
    )
    with torch.no_grad():
        baseline_evidence = robust_score_torch(
            rec.corrected[start:stop].to(device).mean(dim=1)
        )
        old_margin, _, _ = cardiac_contrast(
            baseline_evidence, safe_truth, valid
        )
        target_margin = torch.maximum(
            old_margin + CONTRAST_IMPROVEMENT,
            torch.full_like(old_margin, CONTRAST_MARGIN),
        )

    contrast = (
        F.softplus((target_margin - new_margin) / CONTRAST_SOFTPLUS_TEMP)
        * CONTRAST_SOFTPLUS_TEMP
    ).mean()
    preserve_hr = (
        attenuation.mean(dim=1)[valid] * good_weights
    ).sum(dim=-1).mean()
    size = attenuation.square().mean()
    time_reg = (attenuation[1:] - attenuation[:-1]).square().mean()
    freq_reg = (attenuation[:, :, 1:] - attenuation[:, :, :-1]).square().mean()

    total = (
        BPM_RISK_WEIGHT * risk
        + FB_NLL_WEIGHT * nll
        + CONTRAST_WEIGHT * contrast
        + HR_REGION_ATTENUATION_WEIGHT * preserve_hr
        + ATTENUATION_SIZE_WEIGHT * size
        + ATTENUATION_TIME_WEIGHT * time_reg
        + ATTENUATION_FREQ_WEIGHT * freq_reg
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError(
            f"Nonfinite loss for {rec.name}, block {start}:{stop}"
        )

    return total, {
        "risk": float(risk.detach()),
        "nll": float(nll.detach()),
        "contrast": float(contrast.detach()),
        "margin": float(new_margin.mean().detach()),
        "baseline_margin": float(old_margin.mean().detach()),
        "preserve_hr": float(preserve_hr.detach()),
        "mean_attenuation": float(attenuation.mean().detach()),
        "total": float(total.detach()),
    }


def infer_recording(
    model: FrequencyTemporalPeakMask,
    rec: Recording,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Mean-merge overlapping K=3 mask predictions, then decode one full path."""
    x = rec.corrected.to(device)
    mask_sum = torch.zeros_like(x)
    counts = torch.zeros((rec.n, 1, 1), device=device, dtype=x.dtype)

    for start, stop in training_blocks(rec.n):
        attenuation = model(x[start:stop])["attenuation"]
        mask_sum[start:stop] += attenuation
        counts[start:stop] += 1.0

    if not bool((counts > 0).all()):
        raise RuntimeError(f"Uncovered windows for {rec.name}")

    attenuation = mask_sum / counts
    enhanced = x - attenuation
    return {
        "enhanced": enhanced,
        "attenuation": attenuation,
        "fused": enhanced.mean(dim=1),
        "input_fused": x.mean(dim=1),
    }


def evaluate_training(
    model,
    people: list[Recording],
    device: torch.device,
    penalty: np.ndarray,
) -> dict:
    model.eval()
    rows = []

    with torch.inference_mode():
        for rec in people:
            out = infer_recording(model, rec, device)
            enhanced = out["enhanced"].cpu().numpy()
            attenuation = out["attenuation"].cpu().numpy()
            fused = enhanced.mean(axis=1)
            before = rec.corrected.numpy().mean(axis=1)
            prediction = viterbi_np(robust_score_np(fused), penalty)
            baseline = viterbi_np(robust_score_np(before), penalty)
            valid = rec.valid.numpy()
            enhanced_mae = float(
                np.abs(prediction[valid] - rec.hr[valid]).mean()
            )
            baseline_mae = float(
                np.abs(baseline[valid] - rec.hr[valid]).mean()
            )

            truth = torch.as_tensor(rec.hr, device=device)
            valid_t = rec.valid.to(device)
            safe_truth = torch.where(
                valid_t, truth, torch.full_like(truth, 45.0)
            )
            new_margin, _, _ = cardiac_contrast(
                robust_score_torch(out["fused"]), safe_truth, valid_t
            )
            old_margin, _, _ = cardiac_contrast(
                robust_score_torch(rec.corrected.to(device).mean(dim=1)),
                safe_truth,
                valid_t,
            )

            rows.append({
                "participant": rec.name,
                "enhanced_mae": enhanced_mae,
                "baseline_mae": baseline_mae,
                "raw_mae": rec.raw_mae,
                "attenuation_mean": float(attenuation.mean()),
                "attenuation_p95": float(np.percentile(attenuation, 95)),
                "margin_gain": float((new_margin - old_margin).mean()),
                "contrast_improved": float(
                    (new_margin > old_margin).float().mean()
                ),
            })

    return {
        "enhanced_mae": float(np.mean([r["enhanced_mae"] for r in rows])),
        "baseline_mae": float(np.mean([r["baseline_mae"] for r in rows])),
        "raw_mae": float(np.mean([r["raw_mae"] for r in rows])),
        "attenuation_mean": float(
            np.mean([r["attenuation_mean"] for r in rows])
        ),
        "attenuation_p95": float(
            np.mean([r["attenuation_p95"] for r in rows])
        ),
        "margin_gain": float(np.mean([r["margin_gain"] for r in rows])),
        "contrast_improved": float(
            np.mean([r["contrast_improved"] for r in rows])
        ),
        "participants": rows,
    }


def save_atomic(obj, path: Path):
    temporary = path.with_name(path.name + ".incomplete")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def main():
    if CPU_THREADS > 0:
        torch.set_num_threads(CPU_THREADS)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    penalty = penalty_np()
    train = load_train_cohort()

    model = FrequencyTemporalPeakMask(
        input_scale=INPUT_SCALE,
        max_attenuation=MAX_ATTENUATION,
        width=HEAD_WIDTH,
        block_windows=FB_BLOCK_WINDOWS,
        initial_mask_logit=INITIAL_MASK_LOGIT,
        nhead=ATTENTION_HEADS,
        num_layers=ATTENTION_LAYERS,
        feedforward=ATTENTION_FF,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY
    )
    transition = transition_torch(device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "protocol": "training_only_fixed_epochs_no_dev_no_early_stopping",
        "epochs": EPOCHS,
        "seed": SEED,
        "window_seconds": WINDOW_SECONDS,
        "window_hop_seconds": WINDOW_HOP_SECONDS,
        "radar_background_windows": RADAR_BACKGROUND_WINDOWS,
        "radar_background_quantile": RADAR_BACKGROUND_QUANTILE,
        "radar_background_smooth_hz": RADAR_BACKGROUND_SMOOTH_HZ,
        "radar_background_strength": RADAR_BACKGROUND_STRENGTH,
        "block_windows": FB_BLOCK_WINDOWS,
        "block_stride": FB_BLOCK_STRIDE,
        "width": HEAD_WIDTH,
        "attention_heads": ATTENTION_HEADS,
        "attention_layers": ATTENTION_LAYERS,
        "attention_ff": ATTENTION_FF,
        "attention_axes": "frequency_then_temporal",
        "cell_attention": False,
        "dropout": 0.0,
        "input_scale": INPUT_SCALE,
        "max_attenuation": MAX_ATTENUATION,
        "initial_mask_logit": INITIAL_MASK_LOGIT,
        "learning_rate": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRAD_CLIP,
        "optimizer": "AdamW",
        "gradient_step": "mean_block_gradient_once_per_participant",
        "merge_inference": "mean_overlapping_window_attenuations",
        "tracker_lambda": TRACKER_LAMBDA,
        "tracker_max_change_bpm": TRACKER_MAX_CHANGE_BPM,
        "bpm_risk_weight": BPM_RISK_WEIGHT,
        "fb_nll_weight": FB_NLL_WEIGHT,
        "contrast_weight": CONTRAST_WEIGHT,
        "contrast_margin": CONTRAST_MARGIN,
        "contrast_improvement": CONTRAST_IMPROVEMENT,
        "contrast_softplus_temp": CONTRAST_SOFTPLUS_TEMP,
        "wrong_peak_temp": WRONG_PEAK_SOFTMAX_TEMP,
        "hr_region_sigma_bpm": HR_REGION_SIGMA_BPM,
        "hr_region_half_width_bpm": HR_REGION_HALF_WIDTH_BPM,
        "wrong_exclusion_bpm": WRONG_EXCLUSION_BPM,
        "hr_protection_weight": HR_REGION_ATTENUATION_WEIGHT,
        "attenuation_size_weight": ATTENUATION_SIZE_WEIGHT,
        "attenuation_time_weight": ATTENUATION_TIME_WEIGHT,
        "attenuation_frequency_weight": ATTENUATION_FREQ_WEIGHT,
        "dev_used": False,
        "early_stopping": False,
        "checkpoint_selection": "final_epoch_only",
    }

    csv_path = OUTPUT_DIR / "training_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        csv.writer(handle).writerow([
            "epoch", "train_enhanced_mae", "train_background_mae",
            "train_raw_mae", "train_expected_bpm_risk", "train_fb_nll",
            "train_contrast_loss", "train_margin", "train_baseline_margin",
            "train_hr_region_attenuation", "train_mean_attenuation",
            "train_contrast_gain", "train_pct_windows_contrast_improved",
        ])

    context_seconds = WINDOW_SECONDS + (FB_BLOCK_WINDOWS - 1) * WINDOW_HOP_SECONDS
    print("STAT K=3 FREQUENCY-TEMPORAL PEAK-MASK TRAINING", flush=True)
    print(f"Training participants: {len(train)} | device: {device}", flush=True)
    print(
        f"Input windows: {WINDOW_SECONDS:.0f}s, hop: "
        f"{WINDOW_HOP_SECONDS:.0f}s",
        flush=True,
    )
    print(
        f"K=3 context: {FB_BLOCK_WINDOWS} windows = {context_seconds:.0f}s | "
        f"block stride: {FB_BLOCK_STRIDE} windows = "
        f"{FB_BLOCK_STRIDE * WINDOW_HOP_SECONDS:.0f}s",
        flush=True,
    )
    print(
        f"Radar background: q{RADAR_BACKGROUND_QUANTILE:.2f}, "
        f"smooth={RADAR_BACKGROUND_SMOOTH_HZ:.2f}Hz, "
        f"strength={RADAR_BACKGROUND_STRENGTH:.2f}",
        flush=True,
    )
    print(
        f"Epochs: {EPOCHS} (fixed; no DEV, no early stopping) | "
        f"output: {OUTPUT_DIR}",
        flush=True,
    )
    print(
        f"Trainable parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        flush=True,
    )

    initial = evaluate_training(model, train, device, penalty)
    if abs(initial["enhanced_mae"] - initial["baseline_mae"]) > 1e-5:
        raise RuntimeError(
            "Initial mask changed the background-corrected path; "
            "check the initial mask initialisation."
        )
    print(
        f"EPOCH 000 | enhanced {initial['enhanced_mae']:.3f} bpm | "
        f"background {initial['baseline_mae']:.3f} bpm | "
        f"raw {initial['raw_mae']:.3f} bpm",
        flush=True,
    )
    save_atomic({
        "epoch": 0,
        "config": config,
        "model_state": model.state_dict(),
        "training_metrics": initial,
    }, OUTPUT_DIR / "epoch_000.pt")

    total_blocks = sum(len(training_blocks(rec.n)) for rec in train)
    print(f"Training blocks per epoch: {total_blocks}", flush=True)

    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        model.train()
        sums: dict[str, float] = {}
        used_blocks = 0

        for index in np.random.permutation(len(train)):
            rec = train[int(index)]
            blocks = [
                (start, stop)
                for start, stop in training_blocks(rec.n)
                if bool(rec.valid[start:stop].any())
            ]
            if not blocks:
                raise ValueError(f"No labelled blocks for {rec.name}")

            optimizer.zero_grad(set_to_none=True)
            participant_sums: dict[str, float] = {}

            for start, stop in blocks:
                result = loss_for_block(
                    model, rec, transition, device, start, stop
                )
                if result is None:
                    continue
                total, stats = result
                (total / len(blocks)).backward()
                for key, value in stats.items():
                    participant_sums[key] = (
                        participant_sums.get(key, 0.0) + value
                    )
                used_blocks += 1

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP, error_if_nonfinite=True
            )
            optimizer.step()

            for key, value in participant_sums.items():
                sums[key] = sums.get(key, 0.0) + value / len(blocks)

        sums = {key: value / len(train) for key, value in sums.items()}
        metrics = evaluate_training(model, train, device, penalty)

        print(
            f"\nEPOCH {epoch:03d}/{EPOCHS} | "
            f"{time.perf_counter() - started:.1f}s | {used_blocks} blocks | "
            f"risk={sums['risk']:.3f} | FB NLL={sums['nll']:.3f} | "
            f"contrast={sums['contrast']:.3f}",
            flush=True,
        )
        print(
            f"  TRAIN enhanced {metrics['enhanced_mae']:.3f} | "
            f"background {metrics['baseline_mae']:.3f} | "
            f"raw {metrics['raw_mae']:.3f} bpm",
            flush=True,
        )
        print(
            f"  contrast gain {metrics['margin_gain']:+.3f} | "
            f"mean attenuation {metrics['attenuation_mean']:.4f} | "
            f"p95 {metrics['attenuation_p95']:.4f}",
            flush=True,
        )

        if epoch == 1 or epoch == EPOCHS or epoch % PRINT_PER_PERSON_EVERY == 0:
            print("  Participant enhanced / background / raw MAE:", flush=True)
            for person in metrics["participants"]:
                print(
                    f"    {person['participant']:<12} "
                    f"{person['enhanced_mae']:6.2f} / "
                    f"{person['baseline_mae']:6.2f} / "
                    f"{person['raw_mae']:6.2f} bpm",
                    flush=True,
                )

        with csv_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([
                epoch,
                metrics["enhanced_mae"],
                metrics["baseline_mae"],
                metrics["raw_mae"],
                sums["risk"],
                sums["nll"],
                sums["contrast"],
                sums["margin"],
                sums["baseline_margin"],
                sums["preserve_hr"],
                metrics["attenuation_mean"],
                metrics["margin_gain"],
                metrics["contrast_improved"],
            ])

        checkpoint = {
            "epoch": epoch,
            "config": config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "training_loss": sums,
            "training_metrics": metrics,
        }
        epoch_path = OUTPUT_DIR / f"epoch_{epoch:03d}.pt"
        save_atomic(checkpoint, epoch_path)
        save_atomic(checkpoint, OUTPUT_DIR / "latest.pt")
        print(f"  SAVED {epoch_path.name} and latest.pt", flush=True)

    print(
        f"Training completed: all {EPOCHS} fixed epochs were run. "
        f"The final-epoch checkpoint is epoch_{EPOCHS:03d}.pt; no best-epoch "
        "or DEV checkpoint selection was performed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
