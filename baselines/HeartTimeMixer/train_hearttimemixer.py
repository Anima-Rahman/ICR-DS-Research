"""HeartTimeMixer-inspired heart-rate estimation on ICR-DS.

Based on Sun, Zhao and Tong, Biomedical Signal Processing and Control (2026),
DOI: 10.1016/j.bspc.2026.110034. This is not official author code.
The adaptation uses 36-s/11.11-Hz complex I/Q and EDACM, causal MTI, seven
VMD modes and temporal/frequency mixing. It reads the complete
radar/rx_iq_36s archive from ICR-DS mmFormer-style export files, NOT the
four-bin mmFormer network input. No separate HeartTimeMixer HDF5 is needed.

Edit DATA_ROOT (and SESSION_ROOT if physiology links are broken).
Run: python train_hearttimemixer.py. Only TRAIN data are loaded; the final
model is saved after 100 epochs, without validation or early stopping.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# EDIT: directory containing the ICR-DS Train/ mmFormer-style exports.
DATA_ROOT = Path('path/to/ICR-DS')
TRAIN_ROOT = DATA_ROOT / 'Train' / 'mmFormer'
OUTPUT_ROOT = Path('results/hearttimemixer')
CACHE_DIR = OUTPUT_ROOT / 'feature_cache'
SESSION_ROOT = None  # or Path('path/to/aligned/session_hdf5')
BLOCK_SECONDS = 36.0
STRIDE_SECONDS = 18.0
STRIDE_SAMPLES = 200
FRAME_PERIOD_S = 0.09
TIME_SAMPLES = 400
NUM_MODES = 7
EXPECTED_RX = 4
TARGET_MIN_BPM, TARGET_MAX_BPM = 36.0, 210.0
CACHE_SCHEMA = 3


@dataclass(frozen=True)
class FeatureConfig:
    receiver: int = 0
    range_bin: int | None = None  # None: max causal-MTI power from first 18s, radar only
    mti_frames: int = 33         # paper does NOT specify W; 33 ~ 3s at 11.11 Hz
    alpha: float = 2000.0       # paper value
    modes: int = 7              # paper value
    vmd_max_iter: int = 300     # implementation choice, NOT paper-specified
    vmd_tol: float = 1e-6       # implementation choice, NOT paper-specified
    freq_bins: int = 100       # original paper frequency-branch size
    fft_length: int = 512      # implementation choice; interpolates magnitude to 100 bins


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_timestamps(h5: h5py.File, *, source_path: Path, session_root: Path | None) -> np.ndarray:
    try:
        ts = np.asarray(h5['source_physiology/ibi_ts'], dtype=np.float64)
        # Verify the aligned radar and physiology start times.
        link = h5.get('source_physiology', getlink=True)
        if isinstance(link, h5py.ExternalLink):
            linked_file = (source_path.parent / link.filename).resolve()
            if linked_file.is_file():
                with h5py.File(linked_file, 'r') as original:
                    if 'start_time_iso' in original.attrs and 'aligned_start_time_iso' in h5.attrs:
                        a = _utc_datetime(original.attrs['start_time_iso'])
                        b = _utc_datetime(h5.attrs['aligned_start_time_iso'])
                        if abs((a-b).total_seconds()) > 0.1:
                            raise RuntimeError(f'BioHarness/radar start timestamps differ for {source_path}')
    except (KeyError, OSError, ValueError) as exc:
        if session_root is None:
            raise RuntimeError(
                f'{source_path}: cannot resolve source_physiology/ibi_ts. '
                'The source export has an external HDF5 link to the '
                'original session. Ensure that file remains at its linked '
                'location, or set SESSION_ROOT to the directory containing '
                'the matching sNNN_session.h5. The training script will '
                'not substitute hr_1hz for IBI timestamps.'
            ) from exc
        match = re.search(r's(\d+)_mmformer\.h5$', source_path.name, re.I)
        if not match:
            raise RuntimeError(f'Cannot derive session identifier from {source_path.name}') from exc
        sought = f's{int(match.group(1)):03d}_session.h5'
        hits = list(session_root.rglob(sought))
        if len(hits) != 1:
            raise RuntimeError(f'Expected exactly one {sought} under {session_root}, found {len(hits)}') from exc
        with h5py.File(hits[0], 'r') as original:
            if 'raw/physiology/ibi_ts' not in original:
                raise KeyError(f'{hits[0]} missing raw/physiology/ibi_ts')
            # Guard against attaching beat labels from a different time period.
            if 'start_time_iso' in original.attrs and 'aligned_start_time_iso' in h5.attrs:
                a = _utc_datetime(original.attrs['start_time_iso'])
                b = _utc_datetime(h5.attrs['aligned_start_time_iso'])
                if abs((a - b).total_seconds()) > 0.1:
                    raise RuntimeError(f'Session-start timestamp mismatch for {hits[0]}')
            ts = np.asarray(original['raw/physiology/ibi_ts'], dtype=np.float64)
    if np.any(~np.isfinite(ts)):
        ts = ts[np.isfinite(ts)]
    if np.any(ts < 0):
        raise ValueError(f'{source_path}: negative ibi_ts values; expected seconds after aligned radar start')
    return np.sort(ts)


def _utc_datetime(value) -> datetime:
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError(f'Naive timestamp: {value!r}')
    return dt.astimezone(timezone.utc)


def labels_from_beat_times(times: np.ndarray, n_blocks: int):
    starts = np.arange(n_blocks, dtype=np.float64) * STRIDE_SECONDS
    ends = starts + BLOCK_SECONDS
    # left-inclusive / right-exclusive; overlapping windows intentionally reuse beats.
    counts = (np.searchsorted(times, ends, side='left') -
              np.searchsorted(times, starts, side='left')).astype(np.int32)
    hr = counts.astype(np.float64) * (60.0 / BLOCK_SECONDS)
    valid = (counts > 0) & (hr >= TARGET_MIN_BPM) & (hr <= TARGET_MAX_BPM)
    hr[~valid] = np.nan
    return hr.astype(np.float32), counts, valid


def causal_mti(x: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError('MTI history must be >=1 frame')
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 2:
        raise ValueError('MTI expects [time, ranges]')
    cs = np.concatenate((np.zeros((1, x.shape[1]), np.complex128), np.cumsum(x, axis=0)), axis=0)
    n = np.arange(len(x), dtype=np.int64)
    start = np.maximum(0, n - window)
    count = n - start
    background = np.zeros_like(x)
    nonzero = count > 0
    background[nonzero] = (cs[n[nonzero]] - cs[start[nonzero]]) / count[nonzero, None]
    return x - background


def edacm_phase_difference(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    I, Q = x.real, x.imag
    energy = I*I + Q*Q
    eps = max(1e-8, 1e-6 * float(np.mean(energy)))
    out = np.zeros(len(x), dtype=np.float64)
    if len(x) > 1:
        dI, dQ = np.diff(I), np.diff(Q)
        out[1:] = (I[1:]*dQ - Q[1:]*dI) / (energy[1:] + eps)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError('EDACM generated nonfinite values')
    return out


def vmd_from_scratch(signal: np.ndarray, *, k: int, alpha: float,
                     tol: float, max_iter: int) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(signal, dtype=np.float64).reshape(-1)
    if k < 1 or len(f) < 16 or not np.all(np.isfinite(f)):
        raise ValueError('Invalid VMD mode count or input')
    n = len(f)
    # Mirror 1/2 the signal at both ends (T=2N, for N even).
    half = n // 2
    padded = np.concatenate((f[:half][::-1], f, f[-half:][::-1]))
    t = padded.size
    freq = np.arange(t, dtype=np.float64) / t - 0.5
    f_hat = np.fft.fftshift(np.fft.fft(padded))
    f_plus = f_hat.copy()
    f_plus[:t//2] = 0.0
    u = np.zeros((k, t), dtype=np.complex128)
    dual = np.zeros(t, dtype=np.complex128)
    centers = np.linspace(0.0, 0.45, k, dtype=np.float64)
    # Relaxed VMD: tau=0, as an explicitly chosen ADMM setting.
    for _ in range(max_iter):
        old = u.copy()
        sum_u = np.sum(u, axis=0)
        for mode in range(k):
            sum_u -= u[mode]
            denom = 1.0 + 2.0 * alpha * (freq - centers[mode]) ** 2
            u[mode] = (f_plus - sum_u - dual / 2.0) / denom
            sum_u += u[mode]
            positive = np.abs(u[mode, t//2:])**2
            power = float(positive.sum())
            if power > 1e-18:
                centers[mode] = float(np.dot(freq[t//2:], positive) / power)
        error = float(np.linalg.norm(u - old)**2 / (np.linalg.norm(old)**2 + 1e-12))
        if error < tol:
            break
    # Recreate Hermitian two-sided spectrum, then crop mirrored tails.
    full = np.zeros_like(u)
    full[:, t//2:] = u[:, t//2:]
    full[:, 1:t//2] = np.conj(u[:, :t//2:-1])  # mirrored positive frequencies
    full[:, 0] = np.conj(full[:, -1])
    modes_full = np.fft.ifft(np.fft.ifftshift(full, axes=1), axis=1).real
    modes = modes_full[:, half:half+n]
    order = np.argsort(centers)
    modes = modes[order]
    centers = centers[order]
    if not np.all(np.isfinite(modes)):
        raise FloatingPointError('VMD returned nonfinite modes')
    return modes.astype(np.float32), centers.astype(np.float32)


def mode_magnitude_spectra(modes: np.ndarray, *, fs: float,
                           count: int, nfft: int) -> np.ndarray:
    n = modes.shape[1]
    fft = np.fft.rfft(modes * np.hanning(n)[None, :], n=nfft, axis=-1)
    src_hz = np.fft.rfftfreq(nfft, d=1.0/fs)
    dst_hz = np.linspace(0, fs/2, count)
    result = np.stack([np.interp(dst_hz, src_hz, np.abs(z)) for z in fft], axis=-1)
    return result.astype(np.float32)


def infer_subject(path: Path) -> str:
    for component in path.parts:
        match = re.fullmatch(r'sub[-_](\d+)', component, re.IGNORECASE)
        if match:
            return f'sub-{int(match.group(1)):03d}'
    match = re.search(r's(\d+)_mmformer\.h5$', path.name, re.I)
    if not match:
        raise ValueError(f'Cannot identify subject from {path}; use sub-XX or sXXX_mmformer.h5')
    return f'sub-{int(match.group(1)):03d}'


def enumerate_iq_exports(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f'Missing folder {root}')
    paths = sorted(p for p in root.rglob('*.h5') if re.fullmatch(r's\d+_mmformer\.h5', p.name, re.I))
    if not paths:
        raise FileNotFoundError(f'No ICR-DS full-I/Q export sXXX_mmformer.h5 files found below {root}')
    return paths


def cache_name(path: Path, cfg: FeatureConfig, session_root: Path | None) -> str:
    st = path.stat()
    cfg_dict = asdict(cfg)
    # Include linked physiology file metadata: a changed beat timestamp series
    # must invalidate cached supervision, even if the mmFormer file is unchanged.
    physiology_file = None
    try:
        with h5py.File(path, 'r') as handle:
            link = handle.get('source_physiology', getlink=True)
            if isinstance(link, h5py.ExternalLink):
                candidate = (path.parent / link.filename).resolve()
                if candidate.is_file():
                    physiology_file = candidate
    except (OSError, KeyError):
        pass
    if physiology_file is None and session_root is not None:
        match = re.search(r's(\d+)_mmformer\.h5$', path.name, re.I)
        if match:
            name = f's{int(match.group(1)):03d}_session.h5'
            matches = list(session_root.rglob(name))
            if len(matches) == 1:
                physiology_file = matches[0]
    physiology_meta = None
    if physiology_file is not None:
        phys_stat = physiology_file.stat()
        physiology_meta = {'path': str(physiology_file), 'size': phys_stat.st_size,
                           'mtime_ns': phys_stat.st_mtime_ns}
    key = {'path': str(path.resolve()), 'file_size': st.st_size,
           'mtime_ns': st.st_mtime_ns, 'config': cfg_dict, 'schema': CACHE_SCHEMA,
           'session_root': str(session_root.resolve()) if session_root else None,
           'physiology_file': physiology_meta}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return f'{path.stem}_{digest}.npz'


def read_recording(path: Path, cfg: FeatureConfig, session_root: Path | None):
    with h5py.File(path, 'r') as h5:
        if 'radar/rx_iq_36s' not in h5:
            raise KeyError(f'{path} does not contain radar/rx_iq_36s')
        data = h5['radar/rx_iq_36s']
        if data.ndim != 5 or data.shape[1] != EXPECTED_RX or data.shape[2] != 2 or data.shape[4] != TIME_SAMPLES:
            raise ValueError(f'{path}: expected [N,4,2,13,400], got {data.shape}')
        n_blocks, _, _, n_ranges, _ = data.shape
        if n_ranges < 4:
            raise ValueError('Expected a full 13-range IQ archive, not only a four-bin input')
        if 'radar/range_bin_indices' not in h5:
            raise KeyError(f'{path}: missing radar/range_bin_indices')
        range_bins = np.asarray(h5['radar/range_bin_indices'], dtype=np.int64)
        if len(range_bins) != n_ranges:
            raise ValueError(f'{path}: range indexing incompatible with data shape')
        if not (0 <= cfg.receiver < EXPECTED_RX):
            raise ValueError('Receiver index out of bounds')
        period = float(h5.attrs.get('raw_frame_period_s', np.nan))
        if not np.isfinite(period) or abs(period - FRAME_PERIOD_S) > 1e-6:
            raise ValueError(f'{path}: expected explicit 0.09s frame period; got {period}')
        if 'timing/block_start_s' not in h5 or 'timing/source_block_index' not in h5:
            raise KeyError(f'{path}: timing maps missing')
        starts = np.asarray(h5['timing/block_start_s'], dtype=np.float64)
        blocks = np.asarray(h5['timing/source_block_index'], dtype=np.int64)
        if not (len(starts) == len(blocks) == n_blocks):
            raise ValueError(f'{path}: mismatched timing metadata')
        if not np.allclose(starts, np.arange(n_blocks)*STRIDE_SECONDS, atol=1e-6) or not np.array_equal(blocks, np.arange(n_blocks)):
            raise ValueError(f'{path}: blocks are nonconsecutive or have gaps')
        beats = read_timestamps(h5, source_path=path, session_root=session_root)
        hr, counts, valid = labels_from_beat_times(beats, n_blocks)
        if not valid.any():
            raise ValueError(f'{path}: no valid beat-count HR labels')
        # The 4-RX archive is [N, RX, IQ, R, time]. Only one RX is used.
        raw = np.asarray(data[:, cfg.receiver, :, :, :], dtype=np.float64)
    # Reassemble across all 36s overlapping windows before MTI and EDACM; this avoids
    # artificial resets at radar block boundaries.
    complex_series = raw[:, 0] + 1j*raw[:, 1]  # [N,R,T]
    complex_series = np.transpose(complex_series, (0, 2, 1))  # [window, time, range]
    # Reassemble the native continuous signal WITHOUT repeating the shared 18s overlap.
    complex_series = np.concatenate(
        [complex_series[0], complex_series[1:, STRIDE_SAMPLES:, :].reshape(-1, n_ranges)], axis=0)
    assert len(complex_series) == (n_blocks - 1) * STRIDE_SAMPLES + TIME_SAMPLES
    filtered = causal_mti(complex_series, cfg.mti_frames)
    if cfg.range_bin is not None:
        matches = np.flatnonzero(range_bins == cfg.range_bin)
        if len(matches) != 1:
            raise ValueError(f'Range bin {cfg.range_bin} not in {range_bins.tolist()}')
        idx = int(matches[0])
    else:
        # Radar-only: use the first 200 frames (18 s) to select one range bin
        # for the entire recording; no BioHarness reference used in selection.
        warmup = min(cfg.mti_frames, TIME_SAMPLES//2)
        power = np.mean(np.abs(filtered[warmup:TIME_SAMPLES])**2, axis=0)
        idx = int(np.argmax(power))
    signal = edacm_phase_difference(filtered[:, idx])
    time_features, frequency_features, centers_hz = [], [], []
    for b in range(n_blocks):
        part = signal[b*STRIDE_SAMPLES:b*STRIDE_SAMPLES+TIME_SAMPLES]
        modes, centres = vmd_from_scratch(part, k=cfg.modes, alpha=cfg.alpha,
                                         tol=cfg.vmd_tol, max_iter=cfg.vmd_max_iter)
        time_features.append(modes.T)
        frequency_features.append(mode_magnitude_spectra(
            modes, fs=1.0/FRAME_PERIOD_S, count=cfg.freq_bins, nfft=cfg.fft_length))
        centers_hz.append(centres / FRAME_PERIOD_S)
    return {
        'time': np.stack(time_features).astype(np.float32),
        'frequency': np.stack(frequency_features).astype(np.float32),
        'labels': hr,
        'counts': counts,
        'valid': valid,
        'selected_receiver': np.int32(cfg.receiver),
        'selected_range_bin': np.int32(range_bins[idx]),
        'vmd_centers_hz': np.stack(centers_hz),
    }


def load_or_build(path: Path, cfg: FeatureConfig, cache_dir: Path,
                  session_root: Path | None, rebuild: bool) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / cache_name(path, cfg, session_root)
    if output.exists() and not rebuild:
        with np.load(output, allow_pickle=False) as obj:
            return {k: obj[k] for k in obj.files}
    result = read_recording(path, cfg, session_root)
    # Atomic feature cache; original HDF5 is read-only.
    temp = output.with_suffix('.partial.npz')
    if temp.exists():
        raise FileExistsError(f'Stale partial feature cache: {temp}')
    try:
        np.savez_compressed(temp, **result)
        os.replace(temp, output)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return result


class HeartDataset(Dataset):
    def __init__(self, time: np.ndarray, frequency: np.ndarray, labels: np.ndarray,
                 *, normalizer: 'Normalizer', augment: bool = False):
        self.t = torch.from_numpy(normalizer.apply_time(time))
        self.f = torch.from_numpy(normalizer.apply_frequency(frequency))
        self.y = torch.from_numpy(labels.astype(np.float32))
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        spectrum = self.f[i]
        if self.augment:
            # Table 5: Gaussian noise, sigma=0.01, frequency inputs only,
            # training only (applied after train-fitted standardisation).
            spectrum = spectrum + 0.01 * torch.randn_like(spectrum)
        return self.t[i], spectrum, self.y[i]


class Normalizer:
    def __init__(self, t: np.ndarray, f: np.ndarray):
        self.t_mean = np.mean(t, axis=(0, 1), keepdims=True)
        self.t_std = np.maximum(np.std(t, axis=(0, 1), keepdims=True), 1e-6)
        self.f_mean = np.mean(f, axis=(0, 1), keepdims=True)
        self.f_std = np.maximum(np.std(f, axis=(0, 1), keepdims=True), 1e-6)

    def apply_time(self, t):
        return np.clip((t-self.t_mean)/self.t_std, -30, 30).astype(np.float32)

    def apply_frequency(self, f):
        return np.clip((f-self.f_mean)/self.f_std, -30, 30).astype(np.float32)

    def as_dict(self):
        return {key: getattr(self, key).reshape(-1).astype(float).tolist()
                for key in ('t_mean', 't_std', 'f_mean', 'f_std')}


# --------------------- Paper-inspired HeartTimeMixer -------------------------
def moving_average(x: torch.Tensor, kernel: int) -> torch.Tensor:
    k = int(kernel)
    left = (k-1)//2
    out = F.avg_pool1d(F.pad(x.transpose(1, 2), (left, k-1-left), mode='replicate'),
                       kernel_size=k, stride=1)
    return out.transpose(1, 2)


class MixingBranch(nn.Module):
    def __init__(self, sizes: list[int], width: int, dropout: float,
                 moving_kernels: tuple[int, int]):
        super().__init__()
        self.sizes = sizes
        self.kernels = moving_kernels
        self.embedding = nn.Linear(NUM_MODES, width)
        self.season_mix = nn.ModuleList([
            nn.Sequential(nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout), nn.Linear(b, b))
            for a, b in zip(sizes[:-1], sizes[1:])
        ])
        self.trend_mix = nn.ModuleList([
            nn.Sequential(nn.Linear(b, a), nn.GELU(), nn.Dropout(dropout), nn.Linear(a, a))
            for a, b in zip(sizes[:-1], sizes[1:])
        ])
        self.feedforward = nn.ModuleList([
            nn.Sequential(nn.Linear(width, width*4), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(width*4, width), nn.Dropout(dropout))
            for _ in sizes
        ])
        self.estimators = nn.ModuleList([nn.Linear(width, width) for _ in sizes])
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        multi = [input_features]
        for _ in range(len(self.sizes)-1):
            multi.append(F.avg_pool1d(multi[-1].transpose(1, 2), 2, 2).transpose(1, 2))
        h = [self.dropout(self.embedding(x)) for x in multi]
        seasonal = []
        trend = []
        for x in h:
            # Fig. 6 lists k=25,51 at 30Hz; preserve their *time spans*
            # with k=9,19 at this dataset's 11.11Hz, averaging two estimates.
            tr = sum(moving_average(x, k) for k in self.kernels)/len(self.kernels)
            seasonal.append(x-tr)
            trend.append(tr)
        # Explicit seasonal bottom-up propagation.
        for i in range(len(seasonal)-1):
            transferred = self.season_mix[i](seasonal[i].transpose(1, 2)).transpose(1, 2)
            seasonal[i+1] = seasonal[i+1] + transferred
        # Explicit trend top-down propagation.
        for i in range(len(trend)-2, -1, -1):
            transferred = self.trend_mix[i](trend[i+1].transpose(1, 2)).transpose(1, 2)
            trend[i] = trend[i] + transferred
        # Residual FFN: Linear -> GELU -> Linear, per Fig. 6.
        vectors = []
        for i in range(len(h)):
            mixed = h[i] + seasonal[i] + trend[i]
            mixed = mixed + self.feedforward[i](mixed)
            vectors.append(self.estimators[i](mixed.mean(dim=1)))
        # Multi-estimator averages (paper's Time Avg / Freq Avg).
        return torch.stack(vectors, dim=1).mean(dim=1)


class HeartTimeMixer(nn.Module):
    def __init__(self, time_length=400, frequency_length=100, width=48,
                 scales=4, dropout=0.2, kernels=(9, 19)):
        super().__init__()
        if scales < 2:
            raise ValueError('Need >=2 scales for decomposable mixing')
        def lengths(n):
            result = [n]
            for _ in range(scales-1):
                result.append(result[-1]//2)
            if min(result) < 4:
                raise ValueError('Too many scales for input length')
            return result
        self.temporal = MixingBranch(lengths(time_length), width, dropout, kernels)
        self.spectral = MixingBranch(lengths(frequency_length), width, dropout, kernels)
        self.head = nn.Linear(width*2, 1)

    def forward(self, time_modes, frequency_modes):
        if time_modes.ndim != 3 or frequency_modes.ndim != 3:
            raise ValueError('Expected [B,T,7] and [B,F,7]')
        time_vec = self.temporal(time_modes)
        freq_vec = self.spectral(frequency_modes)
        return self.head(torch.cat((time_vec, freq_vec), dim=-1)).squeeze(-1)


SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LR = 5e-4
HUBER_DELTA = 1.0


def prepare_partition(paths, cfg):
    parts = [load_or_build(path, cfg, CACHE_DIR, SESSION_ROOT, rebuild=False)
             for path in paths]
    time = np.concatenate([part['time'][part['valid']] for part in parts])
    freq = np.concatenate([part['frequency'][part['valid']] for part in parts])
    labels = np.concatenate([part['labels'][part['valid']] for part in parts])
    if len(labels) == 0:
        raise ValueError('No valid beat-count HR windows')
    return time, freq, labels


def main():
    seed_everything(SEED)
    train_files = enumerate_iq_exports(TRAIN_ROOT)
    cfg = FeatureConfig()
    train_time, train_freq, train_hr = prepare_partition(train_files, cfg)
    normalizer = Normalizer(train_time, train_freq)
    dataset = HeartDataset(train_time, train_freq, train_hr,
                           normalizer=normalizer, augment=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeartTimeMixer().to(device)
    with torch.no_grad():
        model.head.bias.fill_(float(train_hr.mean()))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n = 0.0, 0
        for time, freq, hr in loader:
            time, freq, hr = time.to(device), freq.to(device), hr.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.huber_loss(model(time, freq), hr, delta=HUBER_DELTA)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * hr.numel()
            n += hr.numel()
        print(f'Epoch {epoch:03d}/{EPOCHS}: training Huber={total_loss/n:.4f}')

    torch.save(model.state_dict(), OUTPUT_ROOT / 'hearttimemixer_final.pt')
    print(f'Saved final trained model to {OUTPUT_ROOT / "hearttimemixer_final.pt"}')


if __name__ == '__main__':
    main()
