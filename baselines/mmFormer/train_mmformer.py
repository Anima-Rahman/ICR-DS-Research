"""mmFormer-inspired HR regression on ICR-DS.

Based on the mmFormer backbone in WaveBP (Hu et al., ACM IMWUT, 2024).
Uses 36-s radar I/Q windows with four receivers and four range bins;
shares mmFormer backbone weights across receivers and averages their HR
predictions. This scalar-HR implementation is not the full BP-waveform
WaveBP system, which also uses beamforming and other components.

Edit DATA_ROOT (and SESSION_ROOT if physiology links are broken).
Run: python train_mmformer.py. Only TRAIN data are read; a final-epoch
checkpoint is saved after 100 epochs with no validation or early stopping.
"""
from __future__ import annotations

import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# EDIT: directory containing the ICR-DS Train/ exports.
DATA_ROOT = Path('path/to/ICR-DS')
TRAIN_ROOT = DATA_ROOT / 'Train' / 'mmFormer'
OUTPUT_ROOT = Path('results/mmformer')
SESSION_ROOT = None  # or Path('path/to/aligned/session_hdf5')
DATASET_KEY = 'mmformer/input'
EXPECTED_SAMPLE_SHAPE = (4, 2, 4, 400)
BLOCK_SECONDS = 36.0
STRIDE_SECONDS = 18.0
FRAME_PERIOD_S = 0.09
HR_MIN, HR_MAX = 36.0, 210.0
PAPER_URL = 'https://www.chenhuangxun.com/files/ubicomp24-wavebp.pdf'


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_utc(value) -> datetime:
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError(f'Expected timezone-aware ISO timestamp; got {value!r}')
    return dt.astimezone(timezone.utc)


def file_list(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f'Dataset folder does not exist: {root}')
    files = sorted(p for p in root.rglob('*.h5')
                   if re.fullmatch(r's\d+_mmformer\.h5', p.name, re.I))
    if not files:
        raise FileNotFoundError(f'No sXXX_mmformer.h5 found under {root}')
    return files


def subject_id(path: Path) -> str:
    for part in path.parts:
        m = re.fullmatch(r'sub[-_](\d+)', part, re.I)
        if m:
            return f'sub-{int(m.group(1)):03d}'
    m = re.fullmatch(r's(\d+)_mmformer\.h5', path.name, re.I)
    if not m:
        raise ValueError(f'Cannot infer subject from {path}')
    return f'sub-{int(m.group(1)):03d}'


def compare_start_time(model_h5: h5py.File, original: h5py.File, source: Path):
    if ('aligned_start_time_iso' in model_h5.attrs and
            'start_time_iso' in original.attrs):
        diff = abs((to_utc(model_h5.attrs['aligned_start_time_iso']) -
                    to_utc(original.attrs['start_time_iso'])).total_seconds())
        if diff > 0.1:
            raise RuntimeError(f'{source}: BioHarness and radar starts differ by {diff:.3f}s')


def read_beat_times(h5: h5py.File, source: Path,
                    session_root: Path | None) -> np.ndarray:
    ts = None
    link = h5.get('source_physiology', getlink=True)
    if isinstance(link, h5py.ExternalLink):
        linked = (source.parent / link.filename).resolve()
        if linked.is_file():
            with h5py.File(linked, 'r') as original:
                compare_start_time(h5, original, source)
                if 'raw/physiology/ibi_ts' not in original:
                    raise KeyError(f'{linked}: raw/physiology/ibi_ts is missing')
                ts = np.asarray(original['raw/physiology/ibi_ts'], np.float64)
    elif 'source_physiology/ibi_ts' in h5:
        ts = np.asarray(h5['source_physiology/ibi_ts'], np.float64)

    if ts is None:
        m = re.fullmatch(r's(\d+)_mmformer\.h5', source.name, re.I)
        assert m
        needed = f's{int(m.group(1)):03d}_session.h5'
        if session_root is None:
            raise FileNotFoundError(
                f"{source}: missing physiology link; set SESSION_ROOT to the aligned session HDF5 directory")
        candidates = list(session_root.rglob(needed))
        if len(candidates) != 1:
            raise RuntimeError(f'{source}: expected exactly one {needed} under '
                               f'{session_root}, found {len(candidates)}')
        with h5py.File(candidates[0], 'r') as original:
            compare_start_time(h5, original, source)
            if 'raw/physiology/ibi_ts' not in original:
                raise KeyError(f'{candidates[0]}: missing raw/physiology/ibi_ts')
            ts = np.asarray(original['raw/physiology/ibi_ts'], np.float64)
    ts = ts[np.isfinite(ts)]
    if np.any(ts < 0):
        raise ValueError(f'{source}: negative beat timestamps; alignment suspect')
    return np.sort(ts)


def beat_count_labels(beat_times: np.ndarray, starts: np.ndarray, ends: np.ndarray):
    starts = np.asarray(starts, np.float64)
    ends = np.asarray(ends, np.float64)
    if starts.shape != ends.shape or starts.ndim != 1:
        raise ValueError('Block timing arrays must be matching one-dimensional arrays')
    if np.any(~np.isfinite(starts)) or np.any(~np.isfinite(ends)):
        raise ValueError('Nonfinite frame time boundaries')
    if np.any(np.abs((ends - starts) - BLOCK_SECONDS) > 1e-5):
        raise ValueError('Radar input blocks must represent exactly 36 s')
    beats = np.asarray(beat_times, np.float64)
    counts = (np.searchsorted(beats, ends, side='left') -
              np.searchsorted(beats, starts, side='left')).astype(np.int32)
    hr = counts.astype(np.float64) * 60.0 / BLOCK_SECONDS
    valid = (counts > 0) & (hr >= HR_MIN) & (hr <= HR_MAX)
    return hr.astype(np.float32), counts, valid


def scan_recording(path: Path, session_root: Path | None):
    with h5py.File(path, 'r') as h:
        if DATASET_KEY not in h:
            raise KeyError(f'{path}: missing {DATASET_KEY}')
        arr = h[DATASET_KEY]
        if arr.ndim != 5 or tuple(arr.shape[1:]) != EXPECTED_SAMPLE_SHAPE:
            raise ValueError(f'{path}: expected (N,4,2,4,400) at {DATASET_KEY}; '
                             f'found {arr.shape}')
        n = arr.shape[0]
        if ('timing/block_start_s' not in h or
                'timing/block_end_s' not in h or
                'timing/source_block_index' not in h):
            raise KeyError(f'{path}: missing exporter timing arrays; cannot align IBI')
        starts = np.asarray(h['timing/block_start_s'], np.float64)
        ends = np.asarray(h['timing/block_end_s'], np.float64)
        indices = np.asarray(h['timing/source_block_index'], np.int64)
        if len(starts) != n or len(ends) != n or len(indices) != n:
            raise ValueError(f'{path}: block timing or source indices length differs from data')
        if n and (np.any(starts < 0) or np.any(np.diff(starts) < 0)):
            raise ValueError(f'{path}: negative or unsorted block starts')
        if (n and (not np.array_equal(indices, np.arange(n)) or
                   not np.allclose(starts, np.arange(n)*STRIDE_SECONDS, atol=1e-5))):
            raise ValueError(f'{path}: stored 36-second blocks are not consecutive')
        if 'block_duration_s' in h.attrs and abs(float(h.attrs['block_duration_s'])-BLOCK_SECONDS) > 1e-5:
            raise ValueError(f'{path}: unexpected HDF5 block duration')
        if 'raw_frame_period_s' in h.attrs and abs(float(h.attrs['raw_frame_period_s'])-FRAME_PERIOD_S) > 1e-6:
            raise ValueError(f'{path}: unexpected native radar sampling interval')
        if 'mmformer/range_bin_indices' in h:
            ranges = np.asarray(h['mmformer/range_bin_indices'], np.int64)
            if ranges.shape != (4,):
                raise ValueError(f'{path}: mmformer/range_bin_indices must contain 4 bins')
        beat_times = read_beat_times(h, path, session_root)
        hr, beats, valid = beat_count_labels(beat_times, starts, ends)
        if np.any(valid & ~np.isfinite(hr)):
            raise RuntimeError('Nonfinite HR target')
        metadata = [(str(path), int(i), float(starts[i]), float(ends[i]),
                     int(beats[i]), float(hr[i])) for i in np.flatnonzero(valid)]
        return metadata, int(n - len(metadata)), int(n)


class MmFormerDataset(Dataset):
    def __init__(self, samples: list[tuple], mean: np.ndarray,
                 std: np.ndarray):
        self.samples = samples
        self.mean = np.asarray(mean, np.float32).reshape(1,2,1,1)
        self.std = np.asarray(std, np.float32).reshape(1,2,1,1)
        self._open: h5py.File | None = None
        self._path: str | None = None

    def close(self):
        if self._open is not None:
            self._open.close()
            self._open = None
            self._path = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, j):
        path, block, start, end, count, hr = self.samples[j]
        if self._path != path:
            self.close()
            self._open = h5py.File(path, 'r')
            self._path = path
        x = np.asarray(self._open[DATASET_KEY][block], np.float32)
        if x.shape != EXPECTED_SAMPLE_SHAPE:
            raise ValueError(f'{path}: wrong per-block input shape {x.shape}')
        if not np.isfinite(x).all():
            raise ValueError(f'{path}: radar values contain NaN/Inf in block {block}')
        x = np.ascontiguousarray((x - self.mean) / self.std)
        return torch.from_numpy(x), torch.tensor(hr, dtype=torch.float32)

    def __del__(self):
        if hasattr(self, "_open"):
            self.close()


def fit_train_iq_stats(samples: list[tuple]):
    """Training-only I/Q mean and standard deviation."""
    take = samples
    if not take:
        raise ValueError('No training blocks to compute IQ normalization')
    n = 0
    summation = np.zeros(2, np.float64)
    sum_squares = np.zeros(2, np.float64)
    open_h5, current = None, None
    try:
        for path, block, *_ in take:
            if path != current:
                if open_h5 is not None:
                    open_h5.close()
                open_h5 = h5py.File(path, 'r')
                current = path
            x = np.asarray(open_h5[DATASET_KEY][block], np.float64)
            if not np.isfinite(x).all():
                raise ValueError(f'Nonfinite training radar I/Q: {path}, block {block}')
            summation += x.sum(axis=(0,2,3))
            sum_squares += (x*x).sum(axis=(0,2,3))
            n += x.shape[0] * x.shape[2] * x.shape[3]
    finally:
        if open_h5 is not None:
            open_h5.close()
    mean = summation / n
    std = np.sqrt(np.maximum(sum_squares/n - mean*mean, 0.0))
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


class ConvNormAct(nn.Module):
    def __init__(self, cin, cout, *, kernel=11, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, (1,kernel), stride=(1,stride),
                      padding=(0,kernel//2)),
            nn.InstanceNorm2d(cout, affine=True),
            nn.LeakyReLU(negative_slope=0.3, inplace=True))

    def forward(self, x):
        return self.net(x)


class SpatialShortcut(nn.Module):
    def __init__(self, channels: int, nbins: int = 4):
        super().__init__()
        self.channels = channels
        self.nbins = nbins
        self.q = nn.Sequential(nn.Conv2d(channels, channels, 1),
                               nn.LeakyReLU(0.3))
        self.k = nn.Sequential(nn.Conv2d(channels, channels, 1),
                               nn.LeakyReLU(0.3))
        self.v = nn.Sequential(nn.Conv2d(channels, channels, 1),
                               nn.LeakyReLU(0.3))
        self.range_fc = nn.Linear(nbins, 1)

    def forward(self, x):
        b, d, r, t = x.shape
        if r != self.nbins:
            raise ValueError(f'SpatialShortcut expected {self.nbins} range bins; got {r}')
        q, k, v = self.q(x), self.k(x), self.v(x)
        # Eq.(5): C has dimension B x D x Nb x Nb.
        corr = torch.einsum('bdrt,bdst->bdrs', q, k) / math.sqrt(d)
        corr = corr.softmax(dim=-1)
        weighted = torch.einsum('bdrs,bdst->bdrt', corr, v)
        # project Nb -> 1, as written in Eq.(5).
        return self.range_fc(weighted.permute(0,1,3,2)).permute(0,1,3,2)


class TransformerBottleneck(nn.Module):
    def __init__(self, dim=256, heads=8, depth=12, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([TransformBlock(dim, heads, dropout)
                                     for _ in range(depth)])

    def forward(self, x):
        b,c,r,t = x.shape
        sequence = x.permute(0,2,3,1).reshape(b,r*t,c)
        for layer in self.layers:
            sequence = layer(sequence)
        return sequence.reshape(b,r,t,c).permute(0,3,1,2).contiguous()


class TransformBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4*dim), nn.LeakyReLU(0.3),
                                nn.Dropout(dropout), nn.Linear(4*dim, dim))

    def forward(self, x):
        z = self.norm1(x)
        attended, _ = self.attn(z,z,z,need_weights=False)
        x = x + attended
        return x + self.ff(self.norm2(x))


class DecoderUp(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, kernel_size=(1,2), stride=(1,2))
        self.up_norm = nn.InstanceNorm2d(cout, affine=True)
        self.act = nn.LeakyReLU(0.3)
        self.conv = ConvNormAct(cout*2, cout)

    def forward(self, lower, shortcut, desired_time: int):
        x = self.act(self.up_norm(self.up(lower)))
        # 400 is divisible by 16: match the skip precisely by crop/pad.
        if x.shape[-1] != desired_time:
            x = F.interpolate(x, size=(x.shape[-2],desired_time),
                              mode='nearest')
        if shortcut.shape[-2] == 1:
            shortcut = shortcut.expand(-1,-1,x.shape[-2],-1)
        if shortcut.shape[-1] != desired_time:
            raise RuntimeError('Attention and decoder temporal axes do not match')
        return self.conv(torch.cat([x,shortcut], dim=1))


class MmFormerBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.iq_fuse = ConvNormAct(2, 16, kernel=1)
        self.enc0 = ConvNormAct(16,16)
        self.down1 = ConvNormAct(16,32,stride=2)
        self.enc1 = ConvNormAct(32,32)
        self.down2 = ConvNormAct(32,64,stride=2)
        self.enc2 = ConvNormAct(64,64)
        self.down3 = ConvNormAct(64,128,stride=2)
        self.enc3 = ConvNormAct(128,128)
        self.down4 = ConvNormAct(128,256,stride=2)
        self.mid_conv = ConvNormAct(256,256)
        self.transformer = TransformerBottleneck(256,8,12)
        self.skip0 = SpatialShortcut(16)
        self.skip1 = SpatialShortcut(32)
        self.skip2 = SpatialShortcut(64)
        self.skip3 = SpatialShortcut(128)
        self.up3 = DecoderUp(256,128)
        self.up2 = DecoderUp(128,64)
        self.up1 = DecoderUp(64,32)
        self.up0 = DecoderUp(32,16)
        self.features = nn.Conv2d(16,16,(1,11),padding=(0,5))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:3] != (2,4) or x.shape[-1] != 400:
            raise ValueError(f'Backbone expected [B,2,4,400]; got {tuple(x.shape)}')
        f0 = self.enc0(self.iq_fuse(x))
        f1 = self.enc1(self.down1(f0))
        f2 = self.enc2(self.down2(f1))
        f3 = self.enc3(self.down3(f2))
        mid = self.transformer(self.mid_conv(self.down4(f3)))
        z = self.up3(mid, self.skip3(f3), f3.shape[-1])
        z = self.up2(z, self.skip2(f2), f2.shape[-1])
        z = self.up1(z, self.skip1(f1), f1.shape[-1])
        z = self.up0(z, self.skip0(f0), f0.shape[-1])
        return self.features(z)


class MmFormerHR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MmFormerBackbone()
        self.regressor = nn.Sequential(
            ConvNormAct(16,64), ConvNormAct(64,32), ConvNormAct(32,16),
            nn.Conv2d(16,1,(1,11),padding=(0,5)))

    def set_initial_hr(self, bpm: float):
        # Avoid a cold start predicting ~0 bpm with Huber's constant gradient.
        with torch.no_grad():
            self.regressor[-1].bias.fill_(bpm)

    def forward(self, x):
        if x.ndim != 5 or tuple(x.shape[1:]) != EXPECTED_SAMPLE_SHAPE:
            raise ValueError(f'Expected [B,4,2,4,400], got {tuple(x.shape)}')
        b = x.shape[0]
        receivers = x.reshape(b*4,2,4,400)
        feats = self.backbone(receivers)
        values = self.regressor(feats).mean(dim=(-2,-1)).reshape(b,4)
        # No beamforming. Explicit ensemble of four antenna-specific HR estimates.
        return values.mean(dim=1)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
HUBER_DELTA = 1.0


def main():
    seed_all(SEED)
    paths = file_list(TRAIN_ROOT)
    records = [record for path in paths
               for record in scan_recording(path, SESSION_ROOT)[0]]
    if not records:
        raise ValueError('No valid ICR-DS training examples')

    mean, std = fit_train_iq_stats(records)
    dataset = MmFormerDataset(records, mean, std)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MmFormerHR().to(device)
    model.set_initial_hr(float(np.mean([r[-1] for r in records])))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss, n = 0.0, 0
            for radar, hr in loader:
                radar, hr = radar.to(device), hr.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.huber_loss(model(radar), hr, delta=HUBER_DELTA)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * hr.numel()
                n += hr.numel()
            print(f'Epoch {epoch:03d}/{EPOCHS}: training Huber={total_loss/n:.4f}')
        torch.save(model.state_dict(), OUTPUT_ROOT / 'mmformer_final.pt')
        print(f'Saved final trained model to {OUTPUT_ROOT / "mmformer_final.pt"}')
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
