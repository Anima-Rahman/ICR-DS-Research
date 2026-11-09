"""Generate VitaNet-adapted radar inputs from the ICR-DS recordings.

Input layout (the session HDF5 must already be aligned):
    ICR-DS/sub-XXX/raw/radar/adc_data_*.bin
    ICR-DS/sub-XXX/processed/sXXX_session.h5

Output (a new file; source files are read-only):
    ICR-DS/sub-XXX/processed/sXXX_vitanet.h5

The output contains:
    radar/phase_36s       [N, 400, 12, 13]       unwrapped native-rate phase
    vitanet/input         [N, 400, 12, 13, 2]   breathing and heartbeat bands
    vitanet/source_block_index, window_start_s, window_end_s,
    vitanet/window_start_utc, window_end_utc

Each input is a 36-second window (400 frames at approximately 11.11 Hz),
with an 18-second hop. The two physiological bands are 0.2-0.6 Hz and
0.8-3.5 Hz. The output keeps a relative HDF5 external link to the source
physiology group when that group exists in the aligned session file.

This is a VitaNet-inspired adaptation, not an exact reproduction of the
published implementation. It uses one fixed chirp per TX, a 96-point Hann
range FFT, 13 selected range bins (3-15), and fourth-order zero-phase
Butterworth filters. The published VitaNet used 6.4-second, 128-sample inputs
at 20 Hz and 64 range bins. Here, the model input is 36 seconds at native
radar rate, without temporal resampling.

Before running, set DATA_ROOT below to the directory containing sub-XXX
folders. The script processes all discovered subjects and never overwrites
an existing sXXX_vitanet.h5 file.

Install: pip install numpy scipy h5py
Run:     python vitanet_generator.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
import warnings

import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt


# Radar acquisition and preprocessing configuration.
NUM_TX = 3
NUM_RX = 4
ADC_SAMPLES = 96
LOOPS_PER_TX = 255
SOURCE_BLOCK_FRAMES = 200       # 18 seconds at 0.09 seconds per frame
WINDOW_FRAMES = 400             # 36-second model input
STRIDE_FRAMES = 200             # 18-second hop
FRAME_PERIOD_S = 0.09
FFT_SIZE = 96
RANGE_BINS = tuple(range(3, 16))
CHIRP_INDEX = 0                 # one fixed chirp per TX

# Set this to the directory containing sub-XXX folders.
DATA_ROOT = Path("path/to/ICR-DS")

FILE_PATTERN = re.compile(r"adc_data_(?:Raw_)?(\d+)\.bin", re.IGNORECASE)
SUBJECT_FOLDER_PATTERN = re.compile(r"sub-(\d+)", re.IGNORECASE)


def discover_subject_sessions() -> list[tuple[int, Path, Path, Path]]:
    """Find subject folders and their raw, aligned-session and output paths."""
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(
            f"Set DATA_ROOT to the ICR-DS directory containing sub-XXX folders: {DATA_ROOT}"
        )

    sessions = []
    seen = set()
    for subject_dir in DATA_ROOT.iterdir():
        if not subject_dir.is_dir():
            continue
        match = SUBJECT_FOLDER_PATTERN.fullmatch(subject_dir.name)
        if match is None:
            continue
        subject = int(match.group(1))
        if subject in seen:
            raise ValueError(f"Duplicate subject directory for sub-{subject:03d}")
        seen.add(subject)

        raw_dir = subject_dir / "raw" / "radar"
        processed_dir = subject_dir / "processed"
        source_h5 = processed_dir / f"s{subject:03d}_session.h5"
        output_h5 = processed_dir / f"s{subject:03d}_vitanet.h5"
        sessions.append((subject, raw_dir, source_h5, output_h5))

    return sorted(sessions, key=lambda item: item[0])


def export_all_subjects() -> None:
    """Export all subjects found under DATA_ROOT; existing outputs are preserved."""
    sessions = discover_subject_sessions()
    if not sessions:
        raise FileNotFoundError(f"No sub-XXX subject folders found under {DATA_ROOT}")

    succeeded = skipped = failed = 0
    print(f"Found {len(sessions)} subject folder(s) under {DATA_ROOT}", flush=True)
    for subject, raw_dir, source_h5, output_h5 in sessions:
        print(f"\n=== Subject {subject:03d} ===", flush=True)
        if not source_h5.is_file():
            print(f"SKIP: aligned session HDF5 not found: {source_h5}", flush=True)
            skipped += 1
            continue
        if not raw_dir.is_dir():
            print(f"SKIP: raw radar directory not found: {raw_dir}", flush=True)
            skipped += 1
            continue
        if output_h5.exists():
            print(f"SKIP: output already exists: {output_h5}", flush=True)
            skipped += 1
            continue

        try:
            export_vitanet(source_h5, raw_dir, output_h5)
            succeeded += 1
        except Exception as exc:
            print(f"FAILED sub-{subject:03d}: {type(exc).__name__}: {exc}", flush=True)
            failed += 1

    print(f"\nFinished: {succeeded} generated, {skipped} skipped, "
          f"{failed} failed (of {len(sessions)} subjects).", flush=True)
    if failed:
        raise RuntimeError(f"VitaNet export failed for {failed} subject(s)")


def ordered_bin_files(bin_folder: str | Path) -> list[Path]:
    folder = Path(bin_folder)
    matches = []
    if not folder.is_dir():
        raise FileNotFoundError(f"Raw radar directory does not exist: {folder}")
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() != ".bin":
            continue
        match = FILE_PATTERN.fullmatch(path.name)
        if match:
            matches.append((int(match.group(1)), path))
    if not matches:
        raise FileNotFoundError(
            f"No adc_data_<number>.bin or adc_data_Raw_<number>.bin files in {folder}"
        )
    matches.sort(key=lambda pair: pair[0])
    indexes = [i for i, _ in matches]
    if len(set(indexes)) != len(indexes):
        raise ValueError("Duplicate numeric .bin file indices: ordering is ambiguous")
    if any(b != a + 1 for a, b in zip(indexes, indexes[1:])):
        warnings.warn("Gaps between numbered .bin files: the original generator "
                      "also concatenates them; true recording time might contain gaps")
    return [path for _, path in matches]


def _utc_datetime(value: object) -> datetime:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"Timestamp {value!r} has no timezone")
    return dt.astimezone(timezone.utc)


def _frames_per_file(path: Path) -> int:
    int16_per_frame = NUM_TX * NUM_RX * LOOPS_PER_TX * ADC_SAMPLES * 2
    if path.stat().st_size % 2:
        warnings.warn(f"Odd-byte ADC tail in {path.name} will be ignored")
    n_int16 = path.stat().st_size // 2
    count = n_int16 // int16_per_frame
    if count == 0:
        raise ValueError(f"{path.name} has zero complete frames (same failure as original)")
    tail = n_int16 % int16_per_frame
    if tail:
        warnings.warn(f"{path.name}: dropping {tail} final int16 samples, "
                      "as incomplete trailing frame data are discarded")
    return count


def _decode_one_chirp_per_tx(raw_frames: np.ndarray) -> np.ndarray:
    """Decode one TX-interleaved chirp per transmitter from DCA1000 int16 data.

    raw_frames: (frames, flattened int16_per_frame).
    Return: (frames, 12 virtual TX-RX channels, 96 complex ADC samples).
    Within a frame, chirps are TX-interleaved: tx + NUM_TX*loop.
    """
    n_frames = raw_frames.shape[0]
    chirp_size_int16 = NUM_RX * ADC_SAMPLES * 2
    raw = raw_frames.reshape(n_frames, NUM_TX * LOOPS_PER_TX, chirp_size_int16)
    chirp_numbers = [tx + NUM_TX * CHIRP_INDEX for tx in range(NUM_TX)]
    chosen = raw[:, chirp_numbers, :]  # (frames, TX=3, packed samples)
    iq = np.empty((n_frames, NUM_TX, NUM_RX * ADC_SAMPLES), dtype=np.complex64)
    # DCA1000 packed I/Q order:
    # c[0::2] = d[0::4] + j*d[2::4], c[1::2] = d[1::4] + j*d[3::4]
    iq[..., 0::2] = (chosen[..., 0::4].astype(np.float32)
                      + 1j * chosen[..., 2::4].astype(np.float32))
    iq[..., 1::2] = (chosen[..., 1::4].astype(np.float32)
                      + 1j * chosen[..., 3::4].astype(np.float32))
    return iq.reshape(n_frames, NUM_TX * NUM_RX, ADC_SAMPLES)


def _selected_range_profiles(
    paths: list[Path], frames_per_file: list[int], start_frame: int,
    n_frames: int, chunk_frames: int = 64,
) -> np.ndarray:
    """Read only the aligned full frames, streaming over file boundaries."""
    out = np.empty((n_frames, NUM_TX * NUM_RX, len(RANGE_BINS)), np.complex64)
    hann = np.hanning(ADC_SAMPLES).astype(np.float32)
    int16_per_frame = NUM_TX * NUM_RX * LOOPS_PER_TX * ADC_SAMPLES * 2
    end_frame = start_frame + n_frames
    frame_cursor = 0
    write_cursor = 0

    for path, file_count in zip(paths, frames_per_file):
        file_begin, file_end = frame_cursor, frame_cursor + file_count
        frame_cursor = file_end
        take_from = max(start_frame, file_begin)
        take_until = min(end_frame, file_end)
        if take_from >= take_until:
            continue
        # Shape deliberately stops before any incomplete per-file frame.
        packed_frames = np.memmap(
            path, dtype=np.int16, mode="r", shape=(file_count, int16_per_frame)
        )
        local_begin = take_from - file_begin
        local_end = take_until - file_begin
        for lo in range(local_begin, local_end, chunk_frames):
            hi = min(lo + chunk_frames, local_end)
            iq = _decode_one_chirp_per_tx(packed_frames[lo:hi])
            spectrum = np.fft.fft(iq * hann[None, None, :], n=FFT_SIZE, axis=-1)
            selected = np.take(spectrum, RANGE_BINS, axis=-1).astype(np.complex64)
            count = hi - lo
            out[write_cursor:write_cursor + count] = selected
            write_cursor += count
        del packed_frames

    if write_cursor != n_frames:
        raise RuntimeError(f"Extracted {write_cursor} frames; expected {n_frames}")
    return out


def _safe_compressed_dataset(group: h5py.Group, key: str, shape: tuple,
                             dtype=np.float32, chunks=None) -> h5py.Dataset:
    return group.create_dataset(
        key, shape=shape, dtype=dtype, chunks=chunks,
        compression="lzf", shuffle=True
    )


def export_vitanet(existing_h5: str | Path, bin_folder: str | Path,
                   output_h5: str | Path) -> Path:
    """Create 36-second VitaNet inputs with an 18-second hop."""
    original = Path(existing_h5).resolve()
    output = Path(output_h5).resolve()
    if not original.exists():
        raise FileNotFoundError(original)
    if original == output:
        raise ValueError("Output must be a DIFFERENT HDF5 file")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    rb = np.asarray(RANGE_BINS, dtype=int)
    if np.any(rb < 0) or np.any(rb >= FFT_SIZE // 2):
        raise ValueError("Configured range bins must be within the positive FFT spectrum")
    if FFT_SIZE < ADC_SAMPLES or len(set(rb.tolist())) != len(rb):
        raise ValueError("FFT must cover the ADC samples; range bins must be unique")

    native_hz = 1.0 / FRAME_PERIOD_S
    # The alignment metadata and window indices assume 200 frames per 18 s.
    if abs(SOURCE_BLOCK_FRAMES * FRAME_PERIOD_S - 18.0) > 1e-6:
        raise ValueError("Expected 200 frames * frame period == 18.0 s. "
                         "The old window boundaries must be preserved.")

    paths = ordered_bin_files(bin_folder)
    file_counts = [_frames_per_file(p) for p in paths]
    total_complete_frames = sum(file_counts)
    raw_full_blocks = total_complete_frames // SOURCE_BLOCK_FRAMES
    raw_end_discarded = total_complete_frames % SOURCE_BLOCK_FRAMES

    with h5py.File(original, "r") as source:
        if "raw/radar/beamformed_cube" not in source:
            raise KeyError("Original HDF5 has no raw/radar/beamformed_cube")
        source_radar = source["raw/radar/beamformed_cube"]
        if source_radar.ndim < 2 or source_radar.shape[1] != SOURCE_BLOCK_FRAMES:
            raise ValueError("Existing HDF5 radar does not use 200-frame blocks")
        n_blocks = int(source_radar.shape[0])
        if "raw/radar/localization_cube" in source and (
            source["raw/radar/localization_cube"].shape[0] != n_blocks
        ):
            raise ValueError("Existing radar branches do not have equal block counts")
        if n_blocks == 0 or raw_full_blocks < n_blocks:
            raise ValueError("Raw frames cannot account for existing HDF5 blocks")
        start_utc = _utc_datetime(source.attrs["start_time_iso"])
        end_utc = _utc_datetime(source.attrs["end_time_iso"])
        # The source session records the final timestamp one second before
        # the exclusive end of its nominal 18-second blocks.
        expected_duration = n_blocks * 18.0
        metadata_duration = (end_utc - start_utc).total_seconds() + 1.0
        if abs(metadata_duration - expected_duration) > 0.1:
            raise ValueError(
                "Original HDF5 block times do not match its retained radar count: "
                f"(end-start+1)={metadata_duration:.3f}s vs "
                f"{expected_duration:.3f}s. Cannot certify alignment; "
                "not creating output."
            )
        original_has_physio = "raw/physiology" in source

    skipped_blocks = raw_full_blocks - n_blocks
    start_raw_frame = skipped_blocks * SOURCE_BLOCK_FRAMES
    kept_frame_count = n_blocks * SOURCE_BLOCK_FRAMES
    n_windows = n_blocks - 1
    if n_windows < 1:
        raise ValueError("Need at least two aligned source blocks for 36-second windows")
    if start_raw_frame + kept_frame_count > raw_full_blocks * SOURCE_BLOCK_FRAMES:
        raise AssertionError("Internal frame-index mismatch")
    print(f"Raw full 18s blocks: {raw_full_blocks}")
    print(f"Existing aligned HDF5 blocks: {n_blocks}")
    print(f"Skipping {skipped_blocks} whole blocks from the START "
          f"({start_raw_frame} frames); dropping {raw_end_discarded} "
          "final leftover complete frames")
    print(f"Aligned start UTC: {start_utc.isoformat()}")

    complex_frames = _selected_range_profiles(
        paths, file_counts, start_raw_frame, kept_frame_count,
    )
    # Unwrap across the full aligned recording, not independently per window.
    phase_native = np.unwrap(np.angle(complex_frames), axis=0).astype(np.float32)
    del complex_frames

    # Filter the full native-rate recording to avoid per-window transients.
    breathing_sos = butter(4, (0.2, 0.6), btype="bandpass",
                           fs=native_hz, output="sos")
    heartbeat_sos = butter(4, (0.8, 3.5), btype="bandpass",
                           fs=native_hz, output="sos")
    breathing = sosfiltfilt(breathing_sos, phase_native, axis=0).astype(np.float32)
    heartbeat = sosfiltfilt(heartbeat_sos, phase_native, axis=0).astype(np.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".partial")
    if temp.exists():
        raise FileExistsError(f"Remove stale partial file before retrying: {temp}")
    try:
        with h5py.File(temp, "w") as dest:
            dest.attrs["source_session_filename"] = original.name
            dest.attrs["aligned_start_time_iso"] = start_utc.isoformat()
            dest.attrs["aligned_end_time_iso"] = end_utc.isoformat()
            dest.attrs["native_frame_rate_hz"] = native_hz
            dest.attrs["model_input_rate_hz"] = native_hz
            dest.attrs["resampling_applied"] = False
            dest.attrs["native_frames_per_block"] = WINDOW_FRAMES
            dest.attrs["block_duration_s"] = 36.0
            dest.attrs["window_stride_s"] = 18.0
            dest.attrs["raw_blocks_skipped_at_start"] = skipped_blocks
            dest.attrs["raw_frames_skipped_at_start"] = start_raw_frame
            dest.attrs["raw_frames_discarded_at_end"] = raw_end_discarded
            dest.attrs["tx_count"] = NUM_TX
            dest.attrs["rx_count"] = NUM_RX
            dest.attrs["chirp_index_per_tx"] = CHIRP_INDEX
            dest.attrs["loops_per_tx"] = LOOPS_PER_TX
            dest.attrs["range_fft_size"] = FFT_SIZE
            dest.attrs["preprocessing"] = (
                "one fixed chirp per TX/RX; Hann range FFT; temporal phase unwrap; "
                "no resampling; fourth-order zero-phase Butterworth filters; "
                "breathing 0.2-0.6 Hz and heartbeat 0.8-3.5 Hz; VitaNet adaptation")
            dest.create_dataset("range_bins", data=rb.astype(np.int16))
            if original_has_physio:
                rel_source = os.path.relpath(original, start=output.parent)
                dest["source_physiology"] = h5py.ExternalLink(
                    rel_source, "/raw/physiology"
                )

            radar = dest.create_group("radar")
            ds_phase = _safe_compressed_dataset(
                radar, "phase_36s",
                (n_windows, WINDOW_FRAMES, NUM_TX * NUM_RX, len(rb)),
                chunks=(1, WINDOW_FRAMES, NUM_TX * NUM_RX, len(rb)),
            )
            model = dest.create_group("vitanet")
            ds_input = _safe_compressed_dataset(
                model, "input",
                (n_windows, WINDOW_FRAMES, NUM_TX * NUM_RX, len(rb), 2),
                chunks=(1, WINDOW_FRAMES, NUM_TX * NUM_RX, len(rb), 2),
            )
            model.attrs["axis_order"] = "sample,time,virtual_antenna,range,band"
            model.attrs["band_order"] = "breathing_0.2_0.6Hz,heartbeat_0.8_3.5Hz"
            model.attrs["window_duration_s"] = 36.0
            model.attrs["samples_per_window"] = WINDOW_FRAMES
            model.attrs["sample_rate_hz"] = native_hz
            model.attrs["original_paper_window_s"] = 6.4
            model.attrs["temporal_adaptation"] = (
                "36-second overlapping input; downstream model supports 400 native-rate samples"
            )

            window_starts = np.arange(n_windows, dtype=np.float64) * 18.0
            window_ends = window_starts + 36.0
            starts_iso = np.empty(n_windows, dtype=object)
            ends_iso = np.empty(n_windows, dtype=object)
            for block in range(n_windows):
                ds_phase[block] = phase_native[
                    block * STRIDE_FRAMES:block * STRIDE_FRAMES + WINDOW_FRAMES
                ]
                b_slice = slice(block * STRIDE_FRAMES,
                                block * STRIDE_FRAMES + WINDOW_FRAMES)
                ds_input[block] = np.stack(
                    (breathing[b_slice], heartbeat[b_slice]), axis=-1
                )
                starts_iso[block] = (
                    start_utc + timedelta(seconds=float(window_starts[block]))
                ).isoformat()
                ends_iso[block] = (
                    start_utc + timedelta(seconds=float(window_ends[block]))
                ).isoformat()

            model.create_dataset("source_block_index", data=np.arange(n_windows, dtype=np.int32))
            model.create_dataset("window_start_s", data=window_starts)
            model.create_dataset("window_end_s", data=window_ends)
            utf8 = h5py.string_dtype(encoding="utf-8")
            model.create_dataset("window_start_utc", data=starts_iso, dtype=utf8)
            model.create_dataset("window_end_utc", data=ends_iso, dtype=utf8)

        os.replace(temp, output)
    except BaseException:
        if temp.exists():
            temp.unlink()
        raise

    print(f"Saved {n_blocks} original aligned blocks and {n_windows} "
          f"VitaNet overlapping 36s inputs (400 samples at native {native_hz:.6f} Hz) to {output}")
    return output


def main() -> None:
    export_all_subjects()


if __name__ == "__main__":
    main()
