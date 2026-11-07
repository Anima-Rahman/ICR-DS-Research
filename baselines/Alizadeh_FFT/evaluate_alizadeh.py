"""Alizadeh-inspired classical FFT HR evaluation on ICR-DS.

This file evaluates precomputed estimates from the Alizadeh radar exporter:
36-second windows, 18-second hop, 400 radar frames per window, and
beat-count HR = (beats in [start, stop)) * 60 / 36.
Circle-corrected estimates, fallbacks and uncorrected estimates are evaluated
on the same valid windows; metrics are pooled MAE, RMSE and Pearson r.

EDIT TEST_ROOT below to the directory containing sXXX_alizadeh.h5 files.
If HDF5 physiology external links are broken, also edit SESSION_ROOT.
Run: python evaluate_alizadeh.py
"""
from __future__ import annotations
import csv
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import re
import h5py
import numpy as np

# EDIT these paths for your copy of ICR-DS.
TEST_ROOT = Path('path/to/ICR-DS/Test/Alizadeh')
SESSION_ROOT = None  # or Path('path/to/aligned/session_hdf5')
OUTPUT_ROOT = Path('results/alizadeh')
FILE_RE = re.compile(r's(\d+)_alizadeh\.h5$', re.IGNORECASE)
BLOCK_SECONDS = 36.0
STRIDE_SECONDS = 18.0
FRAME_PERIOD_S = 0.09
WINDOW_FRAMES = 400
MIN_HR_BPM, MAX_HR_BPM = 36.0, 210.0

def timestamp_utc(value: object) -> datetime:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"Timezone missing in timestamp {value!r}")
    return dt.astimezone(timezone.utc)

def subject_id(path: Path) -> str:
    match = FILE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Not an sXXX_alizadeh.h5 file: {path}")
    number = int(match.group(1))
    for part in path.parts:
        directory = re.fullmatch(r"sub[-_](\d+)", part, re.IGNORECASE)
        if directory and int(directory.group(1)) != number:
            raise ValueError(f"Subject folder disagrees with filename: {path}")
    return f"sub-{number:03d}"

def discover(directory: Path, *, require: bool = True) -> list[Path]:
    if not directory.is_dir():
        if require:
            raise NotADirectoryError(f"Input folder does not exist: {directory}")
        return []
    paths = sorted((p for p in directory.rglob("*.h5")
                    if FILE_RE.fullmatch(p.name)), key=lambda p: str(p).lower())
    owners = [subject_id(p) for p in paths]
    duplicates = sorted(s for s in set(owners) if owners.count(s) > 1)
    if duplicates:
        raise ValueError(f"Duplicate subject recordings under {directory}: {duplicates}")
    if require and not paths:
        raise FileNotFoundError(f"No sXXX_alizadeh.h5 files under {directory}")
    return paths

def read_beats_from_session(source: Path, *, aligned_start: datetime) -> np.ndarray:
    with h5py.File(source, "r") as f:
        if "raw/physiology/ibi_ts" not in f:
            raise KeyError(f"{source}: no raw/physiology/ibi_ts")
        if "start_time_iso" not in f.attrs:
            raise ValueError(f"{source}: session has no start_time_iso for alignment verification")
        session_start = timestamp_utc(f.attrs["start_time_iso"])
        if abs((session_start - aligned_start).total_seconds()) > 0.1:
            raise ValueError(f"Radar/BioHarness aligned start mismatch for {source}: "
                             f"{session_start.isoformat()} != {aligned_start.isoformat()}")
        return np.asarray(f["raw/physiology/ibi_ts"], dtype=np.float64).reshape(-1)

def lookup_session(path: Path, h5: h5py.File, session_root: Path | None,
                   aligned_start: datetime) -> tuple[np.ndarray, str]:
    link = h5.get("source_physiology", getlink=True)
    if isinstance(link, h5py.ExternalLink):
        target = (path.parent / link.filename).resolve()
        if target.is_file():
            return read_beats_from_session(target, aligned_start=aligned_start), str(target)
    elif link is not None:
        if "source_physiology/ibi_ts" in h5:
            beats = np.asarray(h5["source_physiology/ibi_ts"], dtype=np.float64).reshape(-1)
            return beats, "embedded:source_physiology/ibi_ts"

    attr = h5.attrs.get("source_session_h5")
    if attr is not None:
        if isinstance(attr, bytes):
            attr = attr.decode("utf-8")
        target = Path(str(attr))
        if target.is_file():
            return read_beats_from_session(target, aligned_start=aligned_start), str(target)

    if session_root is not None and session_root.is_dir():
        subject_num = int(FILE_RE.fullmatch(path.name).group(1))
        name = f"s{subject_num:03d}_session.h5"
        direct = session_root / f"sub-{subject_num:02d}" / "processed" / name
        if direct.is_file():
            return read_beats_from_session(direct, aligned_start=aligned_start), str(direct)
        results = list(session_root.rglob(name))
        if len(results) == 1:
            return read_beats_from_session(results[0], aligned_start=aligned_start), str(results[0])
        if len(results) > 1:
            raise RuntimeError(f"Ambiguous matching physiology session for {path}: {results}")
    raise FileNotFoundError(f"{path}: cannot resolve BioHarness ibi_ts. "
                            "Set SESSION_ROOT to the directory containing the aligned session HDF5 files.")

def beats_to_labels(ibi_ts: np.ndarray, block_indices: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(ibi_ts, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if np.any(values < 0):
        raise ValueError("ibi_ts has negative offsets; expected relative seconds from aligned start")
    values.sort()
    starts = block_indices.astype(np.float64) * STRIDE_SECONDS
    ends = starts + BLOCK_SECONDS
    # [start, end), no double counting beats on boundaries.
    counts = np.searchsorted(values, ends, side="left") - np.searchsorted(values, starts, side="left")
    hr = counts.astype(np.float64) * (60.0 / BLOCK_SECONDS)
    label_ok = (counts > 0) & (hr >= MIN_HR_BPM) & (hr <= MAX_HR_BPM)
    hr[~label_ok] = np.nan
    return hr, counts.astype(np.int32), label_ok

def load_recording(path: Path, session_root: Path | None) -> tuple[list[dict], dict]:
    with h5py.File(path, "r") as h5:
        required = ["alizadeh/heart_rate_corrected_bpm",
                    "alizadeh/heart_rate_uncorrected_bpm", "alizadeh/valid",
                    "alizadeh/selected_circle_fit_valid",
                    "alizadeh/source_block_index", "alizadeh/window_start_s",
                    "alizadeh/window_end_s", "radar/selected_range_bin"]
        missing = [name for name in required if name not in h5]
        if missing:
            raise KeyError(f"{path}: missing exported Alizadeh datasets {missing}")
        n = len(h5["alizadeh/valid"])
        if n < 1:
            raise ValueError(f"{path}: no radar blocks")
        arrays = {key: np.asarray(h5[key]).reshape(-1) for key in required}
        for key, arr in arrays.items():
            if arr.shape != (n,):
                raise ValueError(f"{path}: {key} must have shape ({n},), got {arr.shape}")
        block = np.asarray(arrays["alizadeh/source_block_index"], dtype=np.int64)
        if not np.array_equal(block, np.arange(n, dtype=np.int64)):
            raise ValueError(f"{path}: source_block_index must be 0..N-1, not shifted")
        start_s = np.asarray(arrays["alizadeh/window_start_s"], dtype=np.float64)
        stop_s = np.asarray(arrays["alizadeh/window_end_s"], dtype=np.float64)
        if not (np.allclose(start_s, block * STRIDE_SECONDS, atol=1e-5)
                and np.allclose(stop_s, block * STRIDE_SECONDS + BLOCK_SECONDS, atol=1e-5)):
            raise ValueError(f"{path}: window starts/ends are not aligned to a 36s window and 18s hop")
        period = float(h5.attrs.get("frame_period_s_assumed", np.nan))
        samples = int(h5.attrs.get("frames_per_window", -1))
        duration = float(h5.attrs.get("window_duration_s", np.nan))
        if (not np.isfinite(period) or abs(period - FRAME_PERIOD_S) > 1e-7
                or samples != WINDOW_FRAMES or abs(duration - BLOCK_SECONDS) > 1e-6):
            raise ValueError(f"{path}: mismatch with 400 frames, 0.09s, 36s")
        if "slow_time_sampling_rate_hz" in h5.attrs:
            rate = float(h5.attrs["slow_time_sampling_rate_hz"])
            if abs(rate - 1.0 / FRAME_PERIOD_S) > 1e-4:
                raise ValueError(f"{path}: sampling rate mismatch ({rate} Hz)")
        if "aligned_start_time_iso" not in h5.attrs:
            raise KeyError(f"{path}: no aligned_start_time_iso")
        aligned_start = timestamp_utc(h5.attrs["aligned_start_time_iso"])
        if "retained_blocks" in h5.attrs and int(h5.attrs["retained_blocks"]) != n + 1:
            raise ValueError(f"{path}: retained_blocks attribute contradicts saved estimates")
        if "alizadeh/window_start_utc" in h5:
            first_utc = timestamp_utc(h5["alizadeh/window_start_utc"][0])
            if abs((first_utc - aligned_start).total_seconds()) > 0.1:
                raise ValueError(f"{path}: first UTC window does not match aligned start")
        if "alizadeh/window_end_utc" in h5:
            last_utc = timestamp_utc(h5["alizadeh/window_end_utc"][-1])
            expected_end = aligned_start + timedelta(seconds=(n - 1) * STRIDE_SECONDS + BLOCK_SECONDS)
            if abs((last_utc - expected_end).total_seconds()) > 0.1:
                raise ValueError(f"{path}: final UTC window does not match overlapping 36-s windows")
        ibi, beat_source = lookup_session(path, h5, session_root, aligned_start)
        labels, beat_counts, label_ok = beats_to_labels(ibi, block)
        radar_ok = arrays["alizadeh/valid"].astype(bool)
        fit_ok = arrays["alizadeh/selected_circle_fit_valid"].astype(bool)
        corrected = np.asarray(arrays["alizadeh/heart_rate_corrected_bpm"], dtype=np.float64)
        uncorrected = np.asarray(arrays["alizadeh/heart_rate_uncorrected_bpm"], dtype=np.float64)
        selected_bin = arrays["radar/selected_range_bin"].astype(int)
        if "range_candidate_bins" in h5:
            range_set = set(np.asarray(h5["range_candidate_bins"]).astype(int).tolist())
            if any(b not in range_set for b in selected_bin):
                raise ValueError(f"{path}: selected range bin absent from range_candidate_bins")
        subject = subject_id(path)
        rows = []
        for i in range(n):
            common = bool(radar_ok[i] and label_ok[i]
                          and np.isfinite(corrected[i]) and np.isfinite(uncorrected[i]))
            rows.append({
                "split": "", "subject": subject, "source_h5": str(path),
                "block_index": int(block[i]),
                "start_s": float(start_s[i]), "end_s": float(stop_s[i]),
                "start_utc": (aligned_start + timedelta(seconds=float(start_s[i]))).isoformat(),
                "end_utc": (aligned_start + timedelta(seconds=float(stop_s[i]))).isoformat(),
                "selected_range_bin": int(selected_bin[i]),
                "beat_count": int(beat_counts[i]),
                "reference_hr_bpm": float(labels[i]),
                "hr_uncorrected_bpm": float(uncorrected[i]),
                "hr_corrected_or_fallback_bpm": float(corrected[i]),
                "radar_valid": bool(radar_ok[i]), "label_valid": bool(label_ok[i]),
                "circle_fit_valid": bool(fit_ok[i]), "included": common,
                "included_circle_only": bool(common and fit_ok[i]),
            })
        info = {"subject": subject, "file": str(path), "beat_source": beat_source,
                "total_windows": n, "valid_radar": int(np.count_nonzero(radar_ok)),
                "valid_labels": int(np.count_nonzero(label_ok)),
                "included_matched": sum(r["included"] for r in rows),
                "circle_fit_valid_included": sum(r["included_circle_only"] for r in rows)}
        return rows, info

def pooled_metrics(rows: list[dict], *, estimate_key: str,
                   only_circle_valid: bool = False) -> dict:
    key = "included_circle_only" if only_circle_valid else "included"
    used = [r for r in rows if r[key]]
    n = len(used)
    if not n:
        return {"n_windows": 0, "mae_bpm": None, "rmse_bpm": None,
                "pearson_r": None, "pearson_note": "No valid matched windows"}
    truth = np.asarray([r["reference_hr_bpm"] for r in used], dtype=np.float64)
    pred = np.asarray([r[estimate_key] for r in used], dtype=np.float64)
    err = pred - truth
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    if n < 2:
        corr = None
        note = "Pearson undefined: fewer than 2 windows"
    elif np.ptp(truth) < 1e-12 or np.ptp(pred) < 1e-12:
        corr = None
        note = "Pearson undefined: constant labels or predictions"
    else:
        # Pooled Pearson across all windows, not mean subject-level Pearson.
        center_t = truth - truth.mean()
        center_p = pred - pred.mean()
        denom = np.linalg.norm(center_t) * np.linalg.norm(center_p)
        corr = float(np.dot(center_t, center_p) / denom) if denom > 0 else None
        if corr is not None:
            corr = float(np.clip(corr, -1.0, 1.0))
        note = None if corr is not None else "Pearson undefined: zero variance"
    return {"n_windows": n, "mae_bpm": mae, "rmse_bpm": rmse,
            "pearson_r": corr, "pearson_note": note}

def summarize(rows: list[dict]) -> dict:
    return {
        "n_total_windows": len(rows),
        "n_radar_valid": sum(r["radar_valid"] for r in rows),
        "n_ibi_label_valid": sum(r["label_valid"] for r in rows),
        "n_matched_windows": sum(r["included"] for r in rows),
        "n_circle_fit_valid_matched": sum(r["included_circle_only"] for r in rows),
        "corrected_with_fallback": pooled_metrics(
            rows, estimate_key="hr_corrected_or_fallback_bpm"),
        "uncorrected": pooled_metrics(rows, estimate_key="hr_uncorrected_bpm"),
        "circle_fit_valid_only_corrected": pooled_metrics(
            rows, estimate_key="hr_corrected_or_fallback_bpm", only_circle_valid=True),
        "circle_fit_valid_only_uncorrected": pooled_metrics(
            rows, estimate_key="hr_uncorrected_bpm", only_circle_valid=True),
    }

def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v if not isinstance(v, float) or math.isfinite(v) else ""
                             for k, v in row.items()})


def main():
    paths = discover(TEST_ROOT)
    all_rows = []
    for path in paths:
        rows, info = load_recording(path, SESSION_ROOT)
        all_rows.extend(rows)
        print(f"{info['subject']}: {info['included_matched']}/{info['total_windows']} "
              'matched radar/reference windows')
    metrics = summarize(all_rows)
    if not metrics['n_matched_windows']:
        raise ValueError('No valid matched HR predictions in the Test directory')
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / 'metrics.json').write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    _write_csv(OUTPUT_ROOT / 'predictions.csv', all_rows)
    rows = []
    for method in ('corrected_with_fallback', 'uncorrected',
                   'circle_fit_valid_only_corrected', 'circle_fit_valid_only_uncorrected'):
        result = metrics[method]
        rows.append({'method': method, **result})
    _write_csv(OUTPUT_ROOT / 'pooled_metrics.csv', rows)
    for method in ('corrected_with_fallback', 'uncorrected'):
        score = metrics[method]
        print(f"{method}: MAE={score['mae_bpm']:.3f} bpm, "
              f"RMSE={score['rmse_bpm']:.3f} bpm, r={score['pearson_r']}")
    print(f'Wrote predictions and pooled metrics to {OUTPUT_ROOT}')


if __name__ == '__main__':
    main()
