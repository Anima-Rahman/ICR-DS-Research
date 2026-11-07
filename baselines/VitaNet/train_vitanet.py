"""VitaNet-inspired heart-rate regression on ICR-DS.

Based on Khan, Rigazio and Shahzad, ACM IMWUT (2022), DOI: 10.1145/3550330.
This adaptation retains reflection attention and a temporal encoder-decoder,
but predicts scalar HR rather than reconstructing PPG waveforms.

Input: aligned 36-s windows (400 frames, 18-s hop), 12 antennas,
13 range bins, and respiratory/cardiac phase bands.

Edit DATA_ROOT (and SESSION_ROOT if the HDF5 physiology links are broken).
Run: python train_vitanet.py. Only TRAIN data are opened. The final epoch
is saved after a fixed 100-epoch training schedule, with no
checkpoint selection or early stopping.

Network widths, scalar HR loss and learning-rate schedule are adaptations;
this is not an exact reproduction of the original PPG experiment.
"""
from __future__ import annotations

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

# EDIT: directory containing the ICR-DS Train/ folder.
DATA_ROOT = Path('path/to/ICR-DS')
TRAIN_ROOT = DATA_ROOT / 'Train' / 'VitaNet'
OUTPUT_ROOT = Path('results/vitanet')
# EDIT only if source_physiology external links do not resolve.
SESSION_ROOT = None  # or Path('path/to/aligned/session_hdf5')
DATASET_KEY = 'vitanet/input'
EXPECTED_SHAPE = (400, 12, 13, 2)
WINDOW_S = 36.0
STRIDE_S = 18.0
MIN_HR, MAX_HR = 36.0, 210.0


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iso_utc(value) -> datetime:
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError(f'Naive ISO timestamp, expected offset: {value!r}')
    return dt.astimezone(timezone.utc)


def enumerate_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f'Dataset directory does not exist: {root}')
    paths = sorted(p for p in root.rglob('*.h5')
                   if re.fullmatch(r's\d+_vitanet\.h5', p.name, re.I))
    if not paths:
        raise FileNotFoundError(
            f'No sXXX_vitanet.h5 recursively under {root}. '
            'Use the native 400-frame VitaNet exporter, not the earlier 128-sample version.')
    return paths


def subject_id(path: Path) -> str:
    for component in path.parts:
        found = re.fullmatch(r'sub[-_](\d+)', component, re.I)
        if found:
            return f'sub-{int(found.group(1)):03d}'
    found = re.fullmatch(r's(\d+)_vitanet\.h5', path.name, re.I)
    if not found:
        raise ValueError(f'Cannot find subject number in {path}')
    return f'sub-{int(found.group(1)):03d}'


def _compare_alignment(vita: h5py.File, original: h5py.File, source: Path):
    if 'aligned_start_time_iso' in vita.attrs and 'start_time_iso' in original.attrs:
        difference = abs((iso_utc(vita.attrs['aligned_start_time_iso']) -
                          iso_utc(original.attrs['start_time_iso'])).total_seconds())
        if difference > 0.1:
            raise RuntimeError(f'Physiology and VitaNet start times differ by '
                               f'{difference:.2f} s: {source}')


def beat_timestamps(h5: h5py.File, source: Path, session_root: Path | None) -> np.ndarray:
    link = h5.get('source_physiology', getlink=True)
    ts = None
    if isinstance(link, h5py.ExternalLink):
        linked = (source.parent / link.filename).resolve()
        if linked.is_file():
            with h5py.File(linked, 'r') as original:
                _compare_alignment(h5, original, source)
                if 'raw/physiology/ibi_ts' not in original:
                    raise KeyError(f'{linked}: missing raw/physiology/ibi_ts')
                ts = np.asarray(original['raw/physiology/ibi_ts'], dtype=np.float64)
    if ts is None:
        try:
            ts = np.asarray(h5['source_physiology/ibi_ts'], dtype=np.float64)
        except (KeyError, OSError, ValueError) as exc:
            if session_root is None:
                raise RuntimeError(
                    f'{source}: cannot open source_physiology/ibi_ts (external HDF5 link). '
                    'Either restore the original session file to its linked location, '
                    'or set SESSION_ROOT to the directory containing sXXX_session.h5 files.') from exc
            number = re.fullmatch(r's(\d+)_vitanet\.h5', source.name, re.I)
            if not number:
                raise RuntimeError(f'Cannot match session HDF5 for {source.name}') from exc
            wanted = f's{int(number.group(1)):03d}_session.h5'
            hits = list(session_root.rglob(wanted))
            if len(hits) != 1:
                raise RuntimeError(f'Expected precisely one {wanted} beneath '
                                   f'{session_root}, found {len(hits)}') from exc
            with h5py.File(hits[0], 'r') as original:
                _compare_alignment(h5, original, source)
                if 'raw/physiology/ibi_ts' not in original:
                    raise KeyError(f'{hits[0]}: missing raw/physiology/ibi_ts')
                ts = np.asarray(original['raw/physiology/ibi_ts'], dtype=np.float64)
    ts = ts[np.isfinite(ts)]
    if np.any(ts < 0):
        raise ValueError(f'{source}: negative IBI event timestamps; alignment invalid')
    return np.sort(ts)


def count_derived_hr(times: np.ndarray, starts: np.ndarray, stops: np.ndarray):
    times = np.sort(np.asarray(times, np.float64))
    starts = np.asarray(starts, np.float64)
    stops = np.asarray(stops, np.float64)
    if starts.shape != stops.shape or np.any(~np.isfinite(starts + stops)):
        raise ValueError('Nonfinite or mismatched VitaNet window times')
    if np.any(np.abs((stops - starts) - WINDOW_S) > 1e-5):
        raise ValueError('Expected precisely 36 s per VitaNet input window')
    count = (np.searchsorted(times, stops, side='left') -
             np.searchsorted(times, starts, side='left')).astype(np.int32)
    hr = count.astype(np.float64) * (60.0 / WINDOW_S)
    valid = (count > 0) & (hr >= MIN_HR) & (hr <= MAX_HR)
    return hr.astype(np.float32), count, valid


def scan_file(source: Path, session_root: Path | None):
    with h5py.File(source, 'r') as h5:
        if DATASET_KEY not in h5:
            raise KeyError(f'{source}: missing {DATASET_KEY}')
        ds = h5[DATASET_KEY]
        if ds.ndim != 5 or tuple(ds.shape[1:]) != EXPECTED_SHAPE:
            raise ValueError(f'{source}: expected [N,400,12,13,2], got {ds.shape}. '
                             'Do not accidentally use the 6.4 s exporter.')
        n = int(ds.shape[0])
        if 'vitanet/window_start_s' not in h5 or 'vitanet/window_end_s' not in h5:
            raise KeyError(f'{source}: missing vitanet/window_start_s or window_end_s; '
                           'refusing to infer potentially misaligned labels')
        starts = np.asarray(h5['vitanet/window_start_s'], dtype=np.float64)
        stops = np.asarray(h5['vitanet/window_end_s'], dtype=np.float64)
        if len(starts) != n or len(stops) != n:
            raise RuntimeError(f'{source}: radar/clock window counts differ')
        if n and (np.any(starts < 0) or np.any(np.diff(starts) < 0)):
            raise ValueError(f'{source}: negative or unsorted window times')
        if 'block_duration_s' in h5.attrs and abs(float(h5.attrs['block_duration_s']) - WINDOW_S) > 1e-5:
            raise ValueError(f'{source}: not native 36 s data')
        if 'model_input_rate_hz' in h5.attrs and abs(float(h5.attrs['model_input_rate_hz']) - 1/0.09) > 0.02:
            raise ValueError(f'{source}: expected 11.11 Hz native sampling')
        events = beat_timestamps(h5, source, session_root)
        hr, counts, valid = count_derived_hr(events, starts, stops)
    metadata = [(str(source), int(i), float(starts[i]), float(stops[i]),
                 int(counts[i]), float(hr[i])) for i in np.flatnonzero(valid)]
    return metadata, int(np.count_nonzero(~valid)), n


class ICRDS(Dataset):
    def __init__(self, records, means, stds):
        self.records = records
        self.mean = np.asarray(means, dtype=np.float32).reshape(1, 1, 1, 2)
        self.std = np.asarray(stds, dtype=np.float32).reshape(1, 1, 1, 2)
        self._handles: dict[str, h5py.File] = {}

    def __len__(self):
        return len(self.records)

    def close(self):
        for f in self._handles.values():
            f.close()
        self._handles.clear()

    def __del__(self):
        if hasattr(self, "_handles"):
            self.close()

    def __getitem__(self, k):
        source, block, *_ = self.records[k]
        if source not in self._handles:
            self._handles[source] = h5py.File(source, 'r')
        sample = np.asarray(self._handles[source][DATASET_KEY][block], dtype=np.float32)
        if not np.isfinite(sample).all():
            raise ValueError(f'{source}, block {block}: nonfinite VitaNet input')
        sample = np.clip((sample - self.mean) / self.std, -30, 30).astype(np.float32)
        return torch.from_numpy(sample), torch.tensor(self.records[k][-1], dtype=torch.float32)


def training_band_stats(records):
    """Compute per-band mean and standard deviation from training data only."""
    chosen = records
    sums = np.zeros(2, np.float64)
    sq_sums = np.zeros(2, np.float64)
    n_values = 0
    current_path = None
    h5 = None
    try:
        for source, block, *_ in chosen:
            if source != current_path:
                if h5 is not None:
                    h5.close()
                h5 = h5py.File(source, 'r')
                current_path = source
            arr = np.asarray(h5[DATASET_KEY][block], dtype=np.float64)
            if not np.isfinite(arr).all():
                raise ValueError(f'{source}, block {block}: nonfinite VitaNet input')
            sums += arr.sum(axis=(0, 1, 2))
            sq_sums += np.square(arr).sum(axis=(0, 1, 2))
            n_values += int(np.prod(arr.shape[:-1]))
    finally:
        if h5 is not None:
            h5.close()
    if n_values == 0:
        raise ValueError('No training data to estimate normalization')
    mean = sums / n_values
    std = np.sqrt(np.maximum(sq_sums / n_values - mean**2, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


class ConvResidual(nn.Module):
    """Temporal CNN with residual shortcut, BN and 0.2 dropout."""
    def __init__(self, c_in, c_out, kernel=5, dropout=0.2):
        super().__init__()
        self.c1 = nn.Conv1d(c_in, c_out, kernel, padding=kernel//2)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.c2 = nn.Conv1d(c_out, c_out, kernel, padding=kernel//2)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.shortcut = (nn.Identity() if c_in == c_out else nn.Conv1d(c_in,c_out,1))
        self.drop = nn.Dropout1d(dropout)

    def forward(self, x):
        z = self.drop(F.relu(self.bn1(self.c1(x))))
        z = self.bn2(self.c2(z))
        return F.relu(z + self.shortcut(x))


class PerAntennaRangeAttention(nn.Module):
    """Attention (F -> F_a=4) on each antenna's range reflections.

    Time-axis convolutions are shared across antennas/bins. A softmax over
    range bins yields four learned weighted reflection combinations per
    antenna, keeping both physiological bands for the encoder/decoder.
    """
    def __init__(self, antennas=12, ranges=13, heads=4, hidden=24):
        super().__init__()
        self.antennas = antennas
        self.ranges = ranges
        self.heads = heads
        self.temporal = nn.Sequential(
            nn.Conv1d(2, hidden, 7, padding=3),
            nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
        )
        self.projector = nn.Linear(hidden, heads)

    def forward(self, x):
        # x: (batch, time, antenna, range, band)
        b, t, a, r, d = x.shape
        if (a, r, d) != (self.antennas, self.ranges, 2):
            raise ValueError(f'Expected antennas={self.antennas}, range={self.ranges}, '
                             f'bands=2; got {tuple(x.shape)}')
        phase = x.permute(0, 2, 3, 4, 1).reshape(b*a*r, d, t)
        features = self.temporal(phase).mean(dim=-1)
        logits = self.projector(features).reshape(b, a, r, self.heads)
        weights = logits.softmax(dim=2)  # separate range distributions per head
        reduced = torch.einsum('btard,barh->btahd', x, weights)
        reduced = reduced.permute(0, 2, 3, 4, 1).reshape(b, a*self.heads*d, t)
        return reduced


class VitaNetHR(nn.Module):
    """VitaNet attention + residual CNN encoder-decoder with HR-only head.

    This deliberately preserves encoder/decoder topology. The decoded feature
    sequence is pooled to one HR output, because the available target is bpm,
    not a reference PPG waveform. CNN sizes are *not* verified original sizes.
    """
    def __init__(self, antennas=12, ranges=13, heads=4, dropout=0.2):
        super().__init__()
        self.attention = PerAntennaRangeAttention(antennas, ranges, heads)
        n_channel = antennas * heads * 2
        self.enc1 = ConvResidual(n_channel, 64, kernel=7, dropout=dropout)
        self.enc2 = ConvResidual(64, 128, kernel=5, dropout=dropout)
        self.enc3 = ConvResidual(128, 256, kernel=5, dropout=dropout)
        self.mid = ConvResidual(256, 256, kernel=3, dropout=dropout)
        self.dec2 = ConvResidual(256, 128, kernel=5, dropout=dropout)
        self.dec1 = ConvResidual(128, 64, kernel=5, dropout=dropout)
        self.dec0 = ConvResidual(64, 64, kernel=7, dropout=dropout)
        self.regressor = nn.Sequential(
            nn.Linear(64+256, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        z = self.attention(x)                     # [B,96,400]
        z = self.enc1(z)                          # [B,64,400]
        z = self.enc2(F.avg_pool1d(z, 2))         # [B,128,200]
        z = self.enc3(F.avg_pool1d(z, 2))         # [B,256,100]
        middle = self.mid(F.avg_pool1d(z, 2))     # [B,256,50]
        z = self.dec2(F.interpolate(middle, scale_factor=2, mode='linear', align_corners=False))
        z = self.dec1(F.interpolate(z, scale_factor=2, mode='linear', align_corners=False))
        z = self.dec0(F.interpolate(z, scale_factor=2, mode='linear', align_corners=False))
        vector = torch.cat((z.mean(dim=-1), middle.mean(dim=-1)), dim=1)
        return self.regressor(vector).squeeze(-1)


SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
WARMUP_EPOCHS = 5
WARMUP_FACTOR = 0.1
HUBER_DELTA = 1.0


def main():
    seed_all(SEED)
    train_files = enumerate_files(TRAIN_ROOT)
    records = [record for path in train_files
               for record in scan_file(path, SESSION_ROOT)[0]]
    if not records:
        raise ValueError('No valid ICR-DS training windows')

    mean, std = training_band_stats(records)
    dataset = ICRDS(records, mean, std)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VitaNetHR().to(device)
    with torch.no_grad():
        model.regressor[-1].bias.fill_(float(np.mean([r[-1] for r in records])))
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR * WARMUP_FACTOR)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        for epoch in range(1, EPOCHS + 1):
            if epoch == WARMUP_EPOCHS + 1:
                for group in optimizer.param_groups:
                    group['lr'] = LR
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
        torch.save(model.state_dict(), OUTPUT_ROOT / 'vitanet_final.pt')
        print(f'Saved final trained model to {OUTPUT_ROOT / "vitanet_final.pt"}')
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
