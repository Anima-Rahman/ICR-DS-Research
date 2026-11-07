"""HRConvNet-inspired heart-rate estimation on ICR-DS.

Based on Zhao and Tong, Measurement (2025), article 117923.
The ConvNeXt-V2-style network retains the published 239,217-parameter layout.
Inputs are radar-derived EWT-inspired time/frequency modes in [2,8,400]
from overlapping 36-s windows with an 18-s hop. The EWT feature extraction
and acquisition conditions are ICR-DS adaptations, not official author code.

Edit DATA_ROOT (and SESSION_ROOT if physiology links are broken).
Run: python train_hrconvnet.py. Only TRAIN data are read. The final model is
saved after 100 epochs without checkpoint selection.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# EDIT: directory containing the ICR-DS Train/ folder.
DATA_ROOT = Path("path/to/ICR-DS")
TRAIN_ROOT = DATA_ROOT / "Train" / "HRConvNet"
OUTPUT_ROOT = Path("results/hrconvnet")
SESSION_ROOT = None  # or Path("path/to/aligned/session_hdf5")

BLOCK_SECONDS = 36.0
STRIDE_SECONDS = 18.0
TIME_SAMPLES = 400
FRAME_PERIOD = 0.09
FEATURE_SHAPE = (2, 8, 400)
MIN_BPM, MAX_BPM = 36.0, 210.0
PARAMETER_COUNT_PAPER = 239_217
FILE_RE = re.compile(r"s(\d+)_hrconvnet\.h5$", re.IGNORECASE)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic kernels where possible; model has no stochastic augmentation by default.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def utc_timestamp(value) -> datetime:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"Timezone-free timestamp is ambiguous: {value!r}")
    return dt.astimezone(timezone.utc)


def discover_split(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Missing training input directory: {folder}")
    paths = sorted((p for p in folder.rglob("*.h5") if FILE_RE.fullmatch(p.name)),
                   key=lambda p: str(p).lower())
    if not paths:
        raise FileNotFoundError(f"No sXXX_hrconvnet.h5 files under {folder}")
    subjects = [subject_id(p) for p in paths]
    duplicates = {s for s in subjects if subjects.count(s) > 1}
    if duplicates:
        # Each subject is expected to have exactly one session, as in the exporter.
        raise ValueError(f"More than one HDF5 for subject(s) {sorted(duplicates)} in {folder}")
    return paths


def subject_id(path: Path) -> str:
    m = FILE_RE.fullmatch(path.name)
    if m is None:
        raise ValueError(f"Cannot extract subject number from {path}")
    n = int(m.group(1))
    for component in path.parts:
        q = re.fullmatch(r"sub[-_](\d+)", component, re.I)
        if q and int(q.group(1)) != n:
            raise ValueError(f"Subject folder and HDF5 filename disagree: {path}")
    return f"sub-{n:03d}"


def read_beats_from_session(session_path: Path, expected_start: datetime | None) -> np.ndarray:
    with h5py.File(session_path, "r") as source:
        if "raw/physiology/ibi_ts" not in source:
            raise KeyError(f"{session_path}: missing raw/physiology/ibi_ts")
        if expected_start is not None:
            if "start_time_iso" not in source.attrs:
                raise ValueError(f"{session_path}: no start_time_iso for synchronization check")
            s = utc_timestamp(source.attrs["start_time_iso"])
            if abs((s - expected_start).total_seconds()) > 0.1:
                raise ValueError(f"Physiology/radar start mismatch for {session_path}: "
                                 f"{s.isoformat()} vs {expected_start.isoformat()}")
        return np.asarray(source["raw/physiology/ibi_ts"], dtype=np.float64).reshape(-1)


def physiology_beat_times(h5: h5py.File, path: Path,
                           session_root: Path | None) -> tuple[np.ndarray, str]:
    expected_start = None
    if "aligned_start_time_iso" in h5.attrs:
        expected_start = utc_timestamp(h5.attrs["aligned_start_time_iso"])

    link = h5.get("source_physiology", getlink=True)
    if isinstance(link, h5py.ExternalLink):
        external = (path.parent / link.filename).resolve()
        if external.is_file():
            return read_beats_from_session(external, expected_start), str(external)
    elif "source_physiology" in h5:
        if "source_physiology/ibi_ts" in h5:
            return np.asarray(h5["source_physiology/ibi_ts"], dtype=np.float64).reshape(-1), "embedded"

    tried = []
    raw_attr = h5.attrs.get("source_session_h5")
    if raw_attr is not None:
        if isinstance(raw_attr, bytes):
            raw_attr = raw_attr.decode("utf-8")
        candidate = Path(str(raw_attr))
        tried.append(str(candidate))
        if candidate.is_file():
            return read_beats_from_session(candidate, expected_start), str(candidate)

    if session_root is not None and session_root.is_dir():
        n = int(FILE_RE.fullmatch(path.name).group(1))
        expected_file = f"s{n:03d}_session.h5"
        quick = session_root / f"sub-{n:02d}" / "processed" / expected_file
        tried.append(str(quick))
        if quick.is_file():
            return read_beats_from_session(quick, expected_start), str(quick)
        hits = list(session_root.rglob(expected_file))
        if len(hits) == 1:
            return read_beats_from_session(hits[0], expected_start), str(hits[0])
        if len(hits) > 1:
            raise RuntimeError(f"Ambiguous {expected_file} below {session_root}: {hits}")

    raise RuntimeError(
        f"{path}: cannot find BioHarness source_physiology/ibi_ts. Tried {tried}. "    )


def labels_from_beats(ibi_ts: np.ndarray, n_blocks: int):
    times = np.asarray(ibi_ts, np.float64).reshape(-1)
    times = times[np.isfinite(times)]
    if np.any(times < 0):
        raise ValueError("ibi_ts includes negative timestamps, expected seconds after aligned start")
    times.sort()
    starts = np.arange(n_blocks, dtype=np.float64) * STRIDE_SECONDS
    ends = starts + BLOCK_SECONDS
    counts = (np.searchsorted(times, ends, side="left") -
              np.searchsorted(times, starts, side="left")).astype(np.int32)
    bpm = counts.astype(np.float64) * (60.0 / BLOCK_SECONDS)
    valid = (counts > 0) & (bpm >= MIN_BPM) & (bpm <= MAX_BPM)
    bpm[~valid] = np.nan
    return bpm.astype(np.float32), counts, valid


def read_recording(path: Path, *, session_root: Path | None) -> dict:
    with h5py.File(path, "r") as h5:
        if "hrconvnet/input" not in h5 or "radar/phase_valid" not in h5:
            raise KeyError(f"{path}: expected hrconvnet/input and radar/phase_valid")
        dataset = h5["hrconvnet/input"]
        if dataset.ndim != 4 or dataset.shape[1:] != FEATURE_SHAPE:
            raise ValueError(f"{path}: expected [N,2,8,400], got {dataset.shape}")
        n = int(dataset.shape[0])
        if n < 1:
            raise ValueError(f"{path}: empty HDF5")
        valid_radar = np.asarray(h5["radar/phase_valid"][:], dtype=bool).reshape(-1)
        if valid_radar.shape != (n,):
            raise ValueError(f"{path}: phase_valid does not have N entries")
        if "hrconvnet/source_block_index" not in h5 or "hrconvnet/window_start_s" not in h5:
            raise KeyError(f"{path}: missing aligned block index or start time mapping")
        indexes = np.asarray(h5["hrconvnet/source_block_index"], dtype=np.int64)
        starts = np.asarray(h5["hrconvnet/window_start_s"], dtype=np.float64)
        if (not np.array_equal(indexes, np.arange(n)) or
                not np.allclose(starts, STRIDE_SECONDS * np.arange(n), atol=1e-5)):
            raise ValueError(f"{path}: non-consecutive or shifted 36-s windows")
        if "hrconvnet/window_end_s" in h5:
            ends = np.asarray(h5["hrconvnet/window_end_s"], dtype=np.float64)
            if not np.allclose(ends - starts, BLOCK_SECONDS, atol=1e-5):
                raise ValueError(f"{path}: window durations are not all 36 s")
        period = float(h5.attrs.get("frame_period_s_assumed", np.nan))
        if not np.isfinite(period) or abs(period - FRAME_PERIOD) > 1e-7:
            raise ValueError(f"{path}: expected original frame period 0.09s, got {period}")
        if "hrconvnet" in h5 and "sampling_rate_hz" in h5["hrconvnet"].attrs:
            fs = float(h5["hrconvnet"].attrs["sampling_rate_hz"])
            if abs(fs - 1.0 / FRAME_PERIOD) > 1e-5:
                raise ValueError(f"{path}: inconsistent stored temporal sampling rate {fs}")
        beats, beats_source = physiology_beat_times(h5, path, session_root)
        hr, counts, valid_labels = labels_from_beats(beats, n)
        good = valid_radar & valid_labels
        indices = np.flatnonzero(good)
        if not len(indices):
            return {"features": np.empty((0, *FEATURE_SHAPE), np.float32),
                    "labels": np.empty(0, np.float32), "indices": indices,
                    "count_all": n, "radar_valid": int(valid_radar.sum()),
                    "label_valid": int(valid_labels.sum()), "beats_source": beats_source}
        # Read only valid windows; do not shift their parent radar block index.
        data = np.asarray(dataset[indices], dtype=np.float32)
        if not np.all(np.isfinite(data)):
            raise ValueError(f"{path}: nonfinite model inputs in radar/label-valid windows")
        return {"features": data, "labels": hr[indices], "indices": indices,
                "count_all": n, "radar_valid": int(valid_radar.sum()),
                "label_valid": int(valid_labels.sum()), "beats_source": beats_source}


@dataclass
class ChannelModeNormalizer:
    """Training-only mean/std for each EWT mode and input channel."""
    mean: np.ndarray  # [1,2,8,1]
    std: np.ndarray   # [1,2,8,1]
    enabled: bool = True

    @classmethod
    def fit(cls, train: np.ndarray, enabled: bool = True):
        if not enabled:
            return cls(np.zeros((1,2,8,1), np.float32),
                       np.ones((1,2,8,1), np.float32), enabled=False)
        # Float64 avoids catastrophic precision loss when FFT magnitudes are large.
        mean = train.mean(axis=(0,3), keepdims=True, dtype=np.float64)
        std = train.std(axis=(0,3), keepdims=True, dtype=np.float64)
        return cls(mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32))

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -30.0, 30.0).astype(np.float32)

    def as_dict(self):
        return {"enabled": self.enabled, "shape": [1,2,8,1],
                "mean": self.mean.reshape(2,8).astype(float).tolist(),
                "std": self.std.reshape(2,8).astype(float).tolist(),
                "clipping": [-30.0, 30.0]}


class HRConvDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, normalizer: ChannelModeNormalizer):
        self.x = torch.from_numpy(normalizer.apply(x))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class GRN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.linalg.vector_norm(x, ord=2, dim=(1,2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, stride=1,
                                padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pw1 = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.grn = GRN(4 * channels)
        self.pw2 = nn.Linear(4 * channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x).permute(0,2,3,1)  # [B,H,W,C]
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw2(x)
        return residual + x.permute(0,3,1,2)


class HRConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 48, kernel_size=4, stride=4),  # H=8->2, W=400->100
        )
        self.stem_norm = nn.LayerNorm(48, eps=1e-6)
        self.stage1 = nn.Sequential(ConvNeXtV2Block(48), ConvNeXtV2Block(48))
        self.down_norm = nn.LayerNorm(48, eps=1e-6)
        self.downconv = nn.Conv2d(48, 96, kernel_size=2, stride=2)  # H=2->1, W=50->25
        self.stage2 = nn.Sequential(ConvNeXtV2Block(96), ConvNeXtV2Block(96))
        self.final_norm = nn.LayerNorm(96, eps=1e-6)
        self.fc1 = nn.Linear(96, 160)
        self.fc2 = nn.Linear(160, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != FEATURE_SHAPE:
            raise ValueError(f"HRConvNet input requires [B,2,8,400], received {tuple(x.shape)}")
        x = self.stem(x)
        x = self.stem_norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.stage1(x)
        x = self.down_norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.downconv(x)
        x = self.stage2(x)
        x = x.mean(dim=(2,3))  # Global average pooling -> [B,96]
        x = self.final_norm(x)
        x = self.fc1(x)
        return self.fc2(x).squeeze(-1)


SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
LR = 5e-4


def main():
    seed_everything(SEED)
    paths = discover_split(TRAIN_ROOT)
    parts = [read_recording(path, session_root=SESSION_ROOT) for path in paths]
    valid = [part for part in parts if len(part['labels'])]
    if not valid:
        raise ValueError('No valid ICR-DS training examples')
    train_x = np.concatenate([part['features'] for part in valid])
    train_y = np.concatenate([part['labels'] for part in valid])

    normalizer = ChannelModeNormalizer.fit(train_x)
    dataset = HRConvDataset(train_x, train_y, normalizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HRConvNet().to(device)
    with torch.no_grad():
        model.fc2.bias.fill_(float(train_y.mean()))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n = 0.0, 0
        for radar, hr in loader:
            radar, hr = radar.to(device), hr.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.l1_loss(model(radar), hr)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * hr.numel()
            n += hr.numel()
        print(f'Epoch {epoch:03d}/{EPOCHS}: training MAE={total_loss/n:.4f} bpm')

    torch.save(model.state_dict(), OUTPUT_ROOT / 'hrconvnet_final.pt')
    print(f'Saved final trained model to {OUTPUT_ROOT / "hrconvnet_final.pt"}')


if __name__ == '__main__':
    main()
