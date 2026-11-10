"""HRConvNet-inspired radar feature exporter for ICR-DS.

Run after setting DATA_ROOT below:
    python hrconvnet_generator.py

Input for each subject:
    sub-XXX/raw/radar/adc_data_N.bin (or adc_data_Raw_N.bin)
    sub-XXX/processed/sXXX_session.h5 (completed, aligned session)

Output (new file; inputs are never modified):
    sub-XXX/processed/sXXX_hrconvnet.h5

Processing summary:
    * One fixed TX/RX chirp per 0.09-second frame.
    * Hann-windowed 96-point range FFT; candidate bins 3--15.
    * Complex slow-time mean cancellation for range selection, followed by
      phase extraction from the original (unsubtracted) complex range signal.
    * 36-second inputs (400 frames), with an 18-second hop.
    * Eight-component EWT-inspired decomposition; adaptive Meyer-style
      filters with an explicitly flagged fallback when too few peaks exist.
    * Time-domain components and full-FFT magnitudes are stacked as
      hrconvnet/input with shape [N, 2, 8, 400].

This is a documented ICR-DS adaptation of the processing stages in Zhao & Tong,
Measurement (2025), not an exact reproduction. The one-chirp selection, EWT
implementation, range-bin choices and 36-second input are implementation choices.

Dependencies: numpy, scipy, h5py.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
import warnings

import h5py
import numpy as np
from scipy.signal import find_peaks

NUM_TX = 3
NUM_RX = 4
ADC_SAMPLES = 96
LOOPS_PER_TX = 255
SOURCE_BLOCK_FRAMES = 200
WINDOW_FRAMES = 400         # model input
STRIDE_FRAMES = 200         # 18-second hop
FRAME_PERIOD_S = 0.09
N_MODES = 8

# Set this to the directory containing sub-XXX folders.
DATA_ROOT = Path("path/to/ICR-DS")

# Fixed processing configuration used for the ICR-DS experiments.
TX_INDEX = 0
RX_INDEX = 0
CHIRP_LOOP = 0
FFT_SIZE = ADC_SAMPLES
RANGE_START = 3
RANGE_STOP = 15
FREQUENCY_TRANSFORM = "magnitude"  # or "log1p"; keep the experiment setting
DEMEAN_BEFORE_EWT = True

BIN_RE = re.compile(r"adc_data_(?:Raw_)?(\d+)\.bin$")


def ordered_bin_files(folder: Path) -> list[Path]:
    pairs = []
    for p in folder.glob("*.bin"):
        m = BIN_RE.fullmatch(p.name)
        if m:
            pairs.append((int(m.group(1)), p))
    pairs.sort(key=lambda x: x[0])
    if not pairs:
        raise FileNotFoundError(f"No matching adc_data_*.bin files in {folder}")
    ids = [i for i, _ in pairs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate .bin numeric indices")
    if any(b != a + 1 for a, b in zip(ids, ids[1:])):
        warnings.warn("Gaps in .bin indices; recorded-time gaps may exist")
    return [p for _, p in pairs]


def as_utc(obj) -> datetime:
    if isinstance(obj, bytes):
        obj = obj.decode()
    dt = datetime.fromisoformat(str(obj).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Time metadata lacks timezone; refusing ambiguous alignment")
    return dt.astimezone(timezone.utc)


def frames_per_file(path: Path, loops_per_tx: int) -> int:
    integers_per_frame = NUM_TX * NUM_RX * loops_per_tx * ADC_SAMPLES * 2
    num_int16 = path.stat().st_size // 2
    n = num_int16 // integers_per_frame
    if n == 0:
        raise ValueError(f"{path.name}: no full frames")
    tail = num_int16 % integers_per_frame
    if tail or path.stat().st_size % 2:
        warnings.warn(f"{path.name}: discarding an incomplete final frame/tail, as old decoder does")
    return n


def decode_single_chirp_rx(
    packed: np.ndarray, *, loops_per_tx: int, chirp_loop: int, tx: int, rx: int
) -> np.ndarray:
    """Decode same TX/RX chirp in each frame, EXACT IQ byte ordering of radar decoder.

    packed shape [n_frames, int16_per_frame]. Return [n_frames, 96] complex.
    """
    if not 0 <= tx < NUM_TX or not 0 <= rx < NUM_RX:
        raise ValueError("TX must be in 0..2 and RX in 0..3")
    if not 0 <= chirp_loop < loops_per_tx:
        raise ValueError("chirp_loop out of range")
    chirp_int16_count = NUM_RX * ADC_SAMPLES * 2
    chirps = packed.reshape(packed.shape[0], NUM_TX * loops_per_tx, chirp_int16_count)
    selected = chirps[:, tx + NUM_TX * chirp_loop, :]
    iq = np.empty((packed.shape[0], NUM_RX * ADC_SAMPLES), dtype=np.complex64)
    # Original read_dca1000_truncate() quadrature packing:
    iq[:, 0::2] = selected[:, 0::4].astype(np.float32) + 1j * selected[:, 2::4].astype(np.float32)
    iq[:, 1::2] = selected[:, 1::4].astype(np.float32) + 1j * selected[:, 3::4].astype(np.float32)
    return iq.reshape(packed.shape[0], NUM_RX, ADC_SAMPLES)[:, rx, :]


def iter_aligned_range_fft_blocks(
    paths: list[Path], counts: list[int], *, start_frame: int, n_frames: int,
    loops_per_tx: int, chirp_loop: int, tx: int, rx: int,
    fft_size: int, candidate_bins: np.ndarray, chunk_frames: int = 32
):
    """Stream ONLY aligned, full frames, preserving across-.bin window boundaries."""
    int16_per_frame = NUM_TX * NUM_RX * loops_per_tx * ADC_SAMPLES * 2
    window = np.hanning(ADC_SAMPLES).astype(np.float32)
    stop_frame = start_frame + n_frames
    global_frame = 0
    block = np.empty((WINDOW_FRAMES, len(candidate_bins)), dtype=np.complex64)
    used = 0
    yielded = 0
    for path, n_in_file in zip(paths, counts):
        file_start, file_stop = global_frame, global_frame + n_in_file
        global_frame = file_stop
        begin = max(start_frame, file_start)
        end = min(stop_frame, file_stop)
        if begin >= end:
            continue
        memory = np.memmap(path, mode="r", dtype=np.int16, shape=(n_in_file, int16_per_frame))
        lo = begin - file_start
        hi = end - file_start
        for c0 in range(lo, hi, chunk_frames):
            c1 = min(c0 + chunk_frames, hi)
            adc = decode_single_chirp_rx(memory[c0:c1], loops_per_tx=loops_per_tx,
                                         chirp_loop=chirp_loop, tx=tx, rx=rx)
            fft = np.fft.fft(adc * window[None, :], n=fft_size, axis=-1)
            selected = fft[:, candidate_bins].astype(np.complex64)
            pos = 0
            while pos < len(selected):
                take = min(len(selected) - pos, WINDOW_FRAMES - used)
                block[used:used + take] = selected[pos:pos + take]
                used += take
                pos += take
                if used == WINDOW_FRAMES:
                    yield block.copy()
                    yielded += 1
                    block[:WINDOW_FRAMES-STRIDE_FRAMES] = block[STRIDE_FRAMES:]
                    used = WINDOW_FRAMES-STRIDE_FRAMES
        del memory
    if yielded != n_frames // STRIDE_FRAMES - 1 or used != STRIDE_FRAMES:
        raise RuntimeError("Aligned raw data did not form expected complete windows")


def beta_poly(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t**4 * (35 - 84*t + 70*t*t - 20*t**3)


def adaptive_ewt_like(phase: np.ndarray, sample_hz: float, demean: bool = True):
    """EWT-inspired 8-component Meyer filter bank; exact EWT conventions unknown.

    Select 8 strongest positive-frequency spectral maxima. Band boundaries
    are midpoints between sorted maxima, and cosine/sine Meyer transitions
    provide nonoverlapping smooth filters. If there are <8 maxima, we add
    evenly spaced anchors to retain fixed model input shape (flagged).

    Analysis uses squared masks so sum of IMFs reconstructs the demeaned
    input up to floating error; this is a specified adaptation, not proof of
    the authors' unpublished EWT implementation.
    """
    y = np.asarray(phase, np.float64)
    if y.ndim != 1 or y.size != WINDOW_FRAMES or not np.all(np.isfinite(y)):
        raise ValueError("Expected finite 400-point phase window")
    if demean:
        y = y - y.mean()
    spectrum = np.abs(np.fft.rfft(y))
    available = len(spectrum) - 2  # exclude DC and Nyquist as peaks
    if available < N_MODES:
        raise ValueError("Not enough frequency bins for EWT")
    peaks, _ = find_peaks(spectrum[1:-1])
    peaks = peaks + 1
    strongest = sorted(peaks.tolist(), key=lambda i: (-spectrum[i], i))[:N_MODES]
    used_fallback = len(strongest) < N_MODES
    if used_fallback:
        # Explicitly documented fallback: cannot silently claim adaptive EWT
        anchors = np.linspace(1, len(spectrum) - 2, N_MODES + 2)
        for a in np.rint(anchors).astype(int):
            if int(a) not in strongest:
                strongest.append(int(a))
            if len(strongest) == N_MODES:
                break
    centers = np.array(sorted(strongest[:N_MODES]), dtype=np.float64)
    # Midpoints between consecutive maxima in FFT bin coordinates.
    borders = (centers[1:] + centers[:-1]) / 2
    freq_bins = np.arange(len(spectrum), dtype=np.float64)
    all_edges = np.r_[0.0, borders, float(len(spectrum)-1)]
    widths = np.empty(N_MODES-1, dtype=float)
    for j, bound in enumerate(borders):
        widths[j] = 0.24 * min(bound - all_edges[j], all_edges[j+2] - bound)
    masks = np.empty((N_MODES, len(spectrum)), dtype=np.float64)
    for mode in range(N_MODES):
        m = np.ones_like(freq_bins)
        if mode > 0:
            b, tau = borders[mode-1], widths[mode-1]
            up = np.clip((freq_bins - (b-tau)) / (2*tau), 0.0, 1.0)
            m *= np.sin(np.pi/2 * beta_poly(up))
        if mode < N_MODES-1:
            b, tau = borders[mode], widths[mode]
            down = np.clip((freq_bins - (b-tau)) / (2*tau), 0.0, 1.0)
            m *= np.cos(np.pi/2 * beta_poly(down))
        masks[mode] = m
    # Squared synthesis to create 8 real-valued signals that sum to y.
    coeff = np.fft.rfft(y)
    imfs = np.fft.irfft(coeff[None, :] * masks**2, n=y.size, axis=-1).astype(np.float32)
    boundary_hz = borders * sample_hz / y.size
    return imfs, boundary_hz.astype(np.float32), used_fallback


def _iso(utc: datetime) -> str:
    return utc.isoformat().replace("+00:00", "Z")


def export_hrconvnet(
    existing_h5: Path, bin_dir: Path, output_h5: Path, *,
    fft_size: int = 96, range_start: int = 3, range_stop: int = 15,
    tx: int = 0, rx: int = 0, chirp_loop: int = 0,
    loops_per_tx: int = LOOPS_PER_TX, frame_period_s: float = FRAME_PERIOD_S,
    freq_transform: str = "magnitude", demean_before_ewt: bool = True,
) -> Path:
    old = Path(existing_h5).resolve()
    bin_dir = Path(bin_dir).resolve()
    out = Path(output_h5).resolve()
    if not old.is_file():
        raise FileNotFoundError(f"Aligned source HDF5 not found: {old}")
    if not bin_dir.is_dir():
        raise NotADirectoryError(bin_dir)
    if out == old or out.exists():
        raise FileExistsError("Refusing to modify original HDF5 or overwrite an existing export")
    if not 0 <= tx < NUM_TX or not 0 <= rx < NUM_RX:
        raise ValueError("TX must be 0..2 and RX must be 0..3")
    if not 0 <= chirp_loop < loops_per_tx:
        raise ValueError("chirp_loop must be 0..loops_per_tx-1")
    if fft_size < ADC_SAMPLES or not (0 <= range_start <= range_stop < fft_size//2):
        raise ValueError("Need FFT >= 96 and candidate bins in nonnegative FFT range half")
    if abs(SOURCE_BLOCK_FRAMES * frame_period_s - 18.0) > 1e-6:
        raise ValueError("Must preserve original 200 frames = 18 s windowing")
    if freq_transform not in {"magnitude", "log1p"}:
        raise ValueError("freq_transform must be 'magnitude' or 'log1p'")

    rb = np.arange(range_start, range_stop+1)
    paths = ordered_bin_files(bin_dir)
    counts = [frames_per_file(p, loops_per_tx) for p in paths]
    total_frames = sum(counts)
    n_complete_raw = total_frames // SOURCE_BLOCK_FRAMES
    dropped_end = total_frames % SOURCE_BLOCK_FRAMES

    with h5py.File(old, "r") as src:
        if "raw/radar/beamformed_cube" not in src:
            raise KeyError("Old HDF5 missing raw/radar/beamformed_cube")
        old_radar = src["raw/radar/beamformed_cube"]
        if old_radar.ndim < 2 or old_radar.shape[1] != SOURCE_BLOCK_FRAMES:
            raise ValueError("Existing HDF5 does not store 200-frame windows")
        retained = int(old_radar.shape[0])
        if retained < 1 or retained > n_complete_raw:
            raise ValueError("Raw .bin frame count cannot explain existing HDF5")
        if "raw/radar/localization_cube" in src and src["raw/radar/localization_cube"].shape[0] != retained:
            raise ValueError("Original radar phase/localization branch lengths disagree")
        start_utc = as_utc(src.attrs["start_time_iso"])
        end_utc = as_utc(src.attrs["end_time_iso"])
        # Critical: existing script stores inclusive end = start + N*18 - 1.
        metadata_duration = (end_utc - start_utc).total_seconds() + 1
        expected_duration = retained * 18.0
        if abs(metadata_duration - expected_duration) > 0.1:
            raise ValueError(f"Cannot certify existing alignment: metadata spans {metadata_duration} s; "
                             f"radar windows span {expected_duration} s")
        has_physio = "raw/physiology" in src
        old_h5_shape = tuple(old_radar.shape)

    removed_start = n_complete_raw - retained
    start_frame = removed_start * SOURCE_BLOCK_FRAMES
    kept_frames = retained * SOURCE_BLOCK_FRAMES
    n_windows = retained - 1
    if n_windows < 1:
        raise ValueError("Need at least two aligned source blocks for one 36-second input")
    print(f"Raw full frames: {total_frames}; complete 18-s blocks: {n_complete_raw}")
    print(f"Trimming START: {removed_start} whole blocks ({start_frame} raw frames)")
    print(f"Trimming END: {dropped_end} leftover complete frames; per-.bin incomplete tails also discarded")
    print(f"Aligned {retained} blocks from {_iso(start_utc)}; first block maps to OLD beamformed_cube[0]")

    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + ".partial")
    if temp.exists():
        raise FileExistsError(f"Stale temp file exists: {temp}")
    utf8 = h5py.string_dtype(encoding="utf-8")

    try:
        with h5py.File(temp, "w") as dest:
            dest.attrs.update({
                "method": "HRConvNet-style adaptation (not exact original implementation)",
                "source_session_filename": old.name,
                "raw_bin_file_count": len(paths),
                "source_radar_shape": str(old_h5_shape),
                "aligned_start_time_iso": _iso(start_utc),
                "aligned_end_time_iso_from_source_inclusive": _iso(end_utc),
                "end_timestamp_convention": "old inclusive-end metadata uses last block end minus 1s; window_end_utc is exclusive",
                "alignment_method": "raw full 200-frame blocks minus source HDF5 retained blocks; start-trim whole blocks",
                "alignment_verified_against_vbox_timestamps": False,
                "assumed_gapless_concatenation": True,
                "raw_frames_per_file": np.asarray(counts, np.int64),
                "raw_complete_blocks": n_complete_raw,
                "retained_blocks": retained,
                "overlapping_windows": n_windows,
                "window_duration_s": 36.0,
                "window_stride_s": 18.0,
                "raw_blocks_trimmed_at_start": removed_start,
                "raw_frames_trimmed_at_start": start_frame,
                "raw_complete_frames_dropped_at_end": dropped_end,
                "frame_period_s_assumed": frame_period_s,
                "chirp_loop": chirp_loop,
                "tx_index_zero_based": tx,
                "rx_index_zero_based": rx,
                "loops_per_tx": loops_per_tx,
                "adc_samples_per_chirp": ADC_SAMPLES,
                "range_fft_size": fft_size,
                "fft_input_window": "Hann",
                "range_detection": "mean complex cancellation over each overlapping 36s window, then peak residual mean energy",
                "phase_source": "original complex range FFT (without MTI subtraction)",
                "ewt_method": "adaptive spectrum peak midpoint Meyer-band approximation, fixed 8 bands; fallback grid if <8 peaks",
                "ewt_demean_input": demean_before_ewt,
                "frequency_channel_method": "absolute FFT of IMF" if freq_transform == "magnitude" else "log1p(absolute FFT of IMF)",
                "labels": "not generated: link or look up original synchronized physiological ground truth",
            })
            dest.create_dataset("range_candidate_bins", data=rb.astype(np.int16))
            if has_physio:
                dest["source_physiology"] = h5py.ExternalLink(
                    os.path.relpath(old, start=out.parent), "/raw/physiology")

            radar = dest.create_group("radar")
            def dset(parent, name, shape, dtype=np.float32):
                return parent.create_dataset(name, shape=shape, dtype=dtype,
                                             chunks=(1,)+shape[1:] if len(shape)>1 else True,
                                             compression="lzf", shuffle=True)
            phase_ds = dset(radar, "phase_36s", (n_windows, WINDOW_FRAMES))
            iq_ds = dset(radar, "chest_iq_36s", (n_windows, WINDOW_FRAMES, 2))
            target_ds = dset(radar, "selected_range_bin", (n_windows,), np.int16)
            energy_ds = dset(radar, "mti_energy", (n_windows, len(rb)))
            valid_ds = dset(radar, "phase_valid", (n_windows,), np.bool_)

            model = dest.create_group("hrconvnet")
            imf_ds = dset(model, "time_imfs", (n_windows, N_MODES, WINDOW_FRAMES))
            fft_ds = dset(model, "frequency_magnitude", (n_windows, N_MODES, WINDOW_FRAMES))
            inp_ds = dset(model, "input", (n_windows, 2, N_MODES, WINDOW_FRAMES))
            border_ds = dset(model, "ewt_boundaries_hz", (n_windows, N_MODES-1))
            fallback_ds = dset(model, "ewt_fallback_used", (n_windows,), np.bool_)
            model.attrs["axis_order"] = "window, time_vs_fft, IMF_component, time_or_fft_index"
            model.attrs["channel_order"] = "time_domain_IMFs,full_FFT_magnitude_of_IMFs"
            model.attrs["n_modes"] = N_MODES
            model.attrs["sampling_rate_hz"] = 1.0/frame_period_s
            model.attrs["block_duration_s"] = 36.0
            model.attrs["frequency_domain_axis_note"] = "FFT-bin index (NOT chronological time)"

            indexes = np.arange(n_windows, dtype=np.int32)
            model.create_dataset("source_block_index", data=indexes)
            model.create_dataset("source_raw_frame_start", data=start_frame + STRIDE_FRAMES*indexes)
            model.create_dataset("source_raw_frame_end", data=start_frame + STRIDE_FRAMES*indexes + WINDOW_FRAMES)
            model.create_dataset("window_start_s", data=18.0 * indexes)
            model.create_dataset("window_end_s", data=18.0 * indexes + 36.0)
            times0 = [_iso(start_utc + timedelta(seconds=18.0*int(i))) for i in indexes]
            times1 = [_iso(start_utc + timedelta(seconds=18.0*int(i)+36.0)) for i in indexes]
            model.create_dataset("window_start_utc", data=np.asarray(times0,dtype=object), dtype=utf8)
            model.create_dataset("window_end_utc", data=np.asarray(times1,dtype=object), dtype=utf8)

            stream = iter_aligned_range_fft_blocks(
                paths, counts, start_frame=start_frame, n_frames=kept_frames,
                loops_per_tx=loops_per_tx, chirp_loop=chirp_loop,
                tx=tx, rx=rx, fft_size=fft_size, candidate_bins=rb)
            for block_idx, cube in enumerate(stream):
                # cube: [400 frames, candidate range bins] (complex)
                mti = cube - np.mean(cube, axis=0, keepdims=True)
                power = np.mean(np.abs(mti)**2, axis=0)
                chosen_col = int(np.argmax(power))
                target_ds[block_idx] = int(rb[chosen_col])
                energy_ds[block_idx] = power.astype(np.float32)
                raw_chest_iq = cube[:, chosen_col]
                iq_ds[block_idx, :, 0] = raw_chest_iq.real
                iq_ds[block_idx, :, 1] = raw_chest_iq.imag
                phase = np.unwrap(np.angle(raw_chest_iq)).astype(np.float32)
                phase_ds[block_idx] = phase
                valid = bool(np.all(np.isfinite(phase)) and
                             np.mean(np.abs(raw_chest_iq)) > 1e-5 and
                             np.std(phase) > 1e-5)
                valid_ds[block_idx] = valid
                if valid:
                    imfs, borders_hz, fallback = adaptive_ewt_like(
                        phase, 1.0/frame_period_s, demean=demean_before_ewt)
                    freq = np.abs(np.fft.fft(imfs, axis=-1)).astype(np.float32)
                    if freq_transform == "log1p":
                        freq = np.log1p(freq)
                else:
                    warnings.warn(f"Block {block_idx}: phase quality gate failed; outputs zero, phase_valid=False")
                    imfs = np.zeros((N_MODES, WINDOW_FRAMES), np.float32)
                    freq = np.zeros_like(imfs)
                    borders_hz = np.zeros(N_MODES-1, np.float32)
                    fallback = True
                imf_ds[block_idx] = imfs
                fft_ds[block_idx] = freq
                inp_ds[block_idx] = np.stack([imfs, freq], axis=0)
                border_ds[block_idx] = borders_hz
                fallback_ds[block_idx] = fallback
                if (block_idx+1) % 20 == 0 or block_idx+1 == n_windows:
                    print(f"Processed {block_idx+1}/{n_windows} overlapping 36-s windows")
            dest.flush()
        os.replace(temp, out)
    except BaseException:
        if temp.exists():
            temp.unlink()
        raise
    print(f"Saved HRConvNet-style dataset: {out}")
    print(f"  hrconvnet/input shape = ({n_windows}, 2, 8, 400)")
    return out


SUBJECT_RE = re.compile(r"sub-(\d+)$", re.IGNORECASE)
SESSION_RE = re.compile(r"s(\d+)_session\.h5$", re.IGNORECASE)


def discover_sessions(root: Path):
    """Locate ALL sub-NN directories and their matching aligned session HDF5s.

    Yields (subject_number, subject_dir, session_h5 | None, radar_dir, output_h5).
    Missing HDF5s are reported rather than guessed from another subject.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"ICR-DS root directory does not exist: {root}")

    subjects = []
    for candidate in root.iterdir():
        match = SUBJECT_RE.fullmatch(candidate.name)
        if candidate.is_dir() and match:
            subjects.append((int(match.group(1)), candidate))
    subjects.sort(key=lambda row: (row[0], row[1].name.lower()))

    for number, folder in subjects:
        processed = folder / "processed"
        radar_dir = folder / "raw" / "radar"
        candidates = []
        if processed.is_dir():
            for h5 in processed.iterdir():
                matched = SESSION_RE.fullmatch(h5.name)
                if h5.is_file() and matched and int(matched.group(1)) == number:
                    candidates.append(h5)

        if len(candidates) > 1:
            # Do not silently select an ambiguous source.
            raise ValueError(f"More than one matching session HDF5 for {folder}: "
                             + ", ".join(map(str, candidates)))
        session = candidates[0] if candidates else None
        output = processed / (session.name.replace("_session.h5", "_hrconvnet.h5")
                              if session else f"s{number:03d}_hrconvnet.h5")
        yield number, folder, session, radar_dir, output


def batch_export_hrconvnet(root: Path) -> dict:
    """Export every subject under the ICR-DS root; never overwrite outputs."""
    root = Path(root).expanduser()
    summary = dict(done=0, existing=0, missing=0, errors=0)
    found = 0
    for subject, folder, session, radar_dir, output in discover_sessions(root):
        found += 1
        label = f"sub-{subject:03d}"
        if session is None:
            print(f"[SKIP missing session HDF5] {label}: {folder / 'processed'}")
            summary["missing"] += 1
            continue
        if not radar_dir.is_dir():
            print(f"[SKIP missing raw radar folder] {label}: {radar_dir}")
            summary["missing"] += 1
            continue
        try:
            bin_files = ordered_bin_files(radar_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[SKIP invalid/missing raw files] {label}: {exc}")
            summary["missing"] += 1
            continue
        if output.exists():
            print(f"[SKIP output already exists] {label}: {output.name}")
            summary["existing"] += 1
            continue

        print(f"\n[{label}] Exporting {output.name} ({len(bin_files)} raw files)")
        try:
            export_hrconvnet(
                session, radar_dir, output,
                fft_size=FFT_SIZE, range_start=RANGE_START,
                range_stop=RANGE_STOP, tx=TX_INDEX, rx=RX_INDEX,
                chirp_loop=CHIRP_LOOP, loops_per_tx=LOOPS_PER_TX,
                frame_period_s=FRAME_PERIOD_S,
                freq_transform=FREQUENCY_TRANSFORM,
                demean_before_ewt=DEMEAN_BEFORE_EWT,
            )
            summary["done"] += 1
        except Exception as exc:
            print(f"[ERROR] {label}: {type(exc).__name__}: {exc}")
            summary["errors"] += 1

    if found == 0:
        print(f"No sub-XXX subject folders found under {root}")
    print("\nBatch summary: " + ", ".join(
        f"{key}={value}" for key, value in summary.items()))
    return summary


def main() -> None:
    summary = batch_export_hrconvnet(DATA_ROOT)
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
