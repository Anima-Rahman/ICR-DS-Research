"""Create mmFormer-style radar inputs from ICR-DS raw ADC recordings.

For each subject, this script reads:
    sub-XXX/raw/radar/adc_data_[Raw_]N.bin
    sub-XXX/processed/sXXX_session.h5

and creates:
    sub-XXX/processed/sXXX_mmformer.h5

Each window contains 36 seconds (400 frames) of complex range-FFT data, with
a 18-second (200-frame) hop. The exporter stores the full 13-bin ROI archive
for HeartTimeMixer and a separate four-bin view for mmFormer. The source HDF5
is organised in 18-second blocks; these blocks are retained for alignment.

This is an HR-oriented adaptation, not a verbatim WaveBP/mmFormer reproduction.
It uses one fixed TDM chirp from one transmitter and all four receivers. The
input is not Doppler processed, beamformed, or resampled to 125 Hz.

The exporter infers the number of complete 18-second blocks removed from the
start of the raw recording, then verifies the inferred offset against the
source HDF5 at the first, middle, and last retained frames. Export stops if
those complex samples do not match.

Output datasets:
    radar/rx_iq_36s  (N, 4, 2, 13, 400)   full ROI archive for HeartTimeMixer
    radar/range_bin_indices              FFT bins represented in the full archive
    mmformer/input   (N, 4, 2,  4, 400)   four-bin mmFormer model input
    mmformer/range_bin_indices           FFT bins represented in mmFormer input

Requirements: numpy, h5py
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
import warnings

import h5py
import numpy as np


# -----------------------------------------------------------------------------
# Edit this path for your local ICR-DS copy.
# Expected subject layout: sub-XXX/raw/radar and sub-XXX/processed.
# -----------------------------------------------------------------------------
DATA_ROOT = Path("path/to/ICR-DS")

# Fixed radar and window configuration used for this baseline adaptation.
NUM_TX = 3
NUM_RX = 4
ADC_SAMPLES = 96
LOOPS_PER_TX = 255
TX_INDEX = 0
CHIRP_INDEX = 0

SOURCE_BLOCK_FRAMES = 200       # 18 seconds at 11.111 frames/s
WINDOW_FRAMES = 400             # 36-second model input
WINDOW_STRIDE_FRAMES = 200      # 18-second hop
FRAME_PERIOD_S = 0.09
FFT_SIZE = 96
ROI_START = 3
ROI_STOP = 15                  # inclusive
MMFORMER_RANGE_START = 7       # fixed four-bin selection: 7, 8, 9, 10
NUM_MODEL_RANGES = 4
ALIGNMENT_TOLERANCE = 0.002

FILE_RE = re.compile(r"^adc_data_(?:Raw_)?(\d+)\.bin$", re.IGNORECASE)
SUBJECT_RE = re.compile(r"^sub-(\d+)$", re.IGNORECASE)


def ordered_bin_files(folder: Path) -> list[Path]:
    """Return numbered ADC files in numeric order and warn about gaps."""
    pairs = []
    for path in folder.iterdir():
        if path.is_file():
            match = FILE_RE.fullmatch(path.name)
            if match:
                pairs.append((int(match.group(1)), path))

    if not pairs:
        raise FileNotFoundError(f"No numbered ADC .bin files found in {folder}")

    pairs.sort(key=lambda item: item[0])
    indices = [number for number, _ in pairs]
    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate ADC file indices in {folder}")
    if any(b != a + 1 for a, b in zip(indices, indices[1:])):
        warnings.warn(
            f"Missing numbered ADC files in {folder}; capture-time gaps may exist",
            stacklevel=2,
        )
    return [path for _, path in pairs]


def _int16_per_frame() -> int:
    """Number of int16 values in one complete raw radar frame."""
    return NUM_TX * NUM_RX * LOOPS_PER_TX * ADC_SAMPLES * 2


def count_frames(path: Path) -> int:
    """Count complete frames in a raw file, discarding any incomplete tail."""
    values_per_frame = _int16_per_frame()
    bytes_per_frame = values_per_frame * np.dtype(np.int16).itemsize
    count, leftover = divmod(path.stat().st_size, bytes_per_frame)
    if count == 0:
        raise ValueError(f"{path.name} contains no complete radar frames")
    if leftover:
        warnings.warn(
            f"{path.name}: discarding {leftover} bytes after its complete frames",
            stacklevel=2,
        )
    return int(count)


def _unpack_iq(chirp_packed: np.ndarray) -> np.ndarray:
    """Decode one chirp into complex samples with shape (..., 4 RX, 96 ADC)."""
    data = chirp_packed
    decoded = np.empty(
        data.shape[:-1] + (NUM_RX * ADC_SAMPLES,), dtype=np.complex64
    )
    decoded[..., 0::2] = (
        data[..., 0::4].astype(np.float32)
        + 1j * data[..., 2::4].astype(np.float32)
    )
    decoded[..., 1::2] = (
        data[..., 1::4].astype(np.float32)
        + 1j * data[..., 3::4].astype(np.float32)
    )
    return decoded.reshape(data.shape[:-1] + (NUM_RX, ADC_SAMPLES))


def one_chirp_four_rx(raw_frames: np.ndarray) -> np.ndarray:
    """Extract the fixed TX/chirp from frames; return (frames, 4 RX, 96 ADC)."""
    reshaped = raw_frames.reshape(
        raw_frames.shape[0],
        NUM_TX * LOOPS_PER_TX,
        NUM_RX * ADC_SAMPLES * 2,
    )
    selected = reshaped[:, TX_INDEX + NUM_TX * CHIRP_INDEX, :]
    return _unpack_iq(selected)


class RawFrameReader:
    """Read complete frames across the ordered ADC files as one timeline."""

    def __init__(self, files: list[Path]):
        self.files = files
        self.counts = [count_frames(path) for path in files]
        self.cumulative = np.cumsum([0] + self.counts).tolist()
        self.total = self.cumulative[-1]
        self.frame_int16 = _int16_per_frame()

    def get_frames(self, begin: int, count: int) -> np.ndarray:
        """Read a contiguous frame slice from the concatenated raw files."""
        if begin < 0 or count < 0 or begin + count > self.total:
            raise IndexError(f"Raw frame slice {begin}:{begin + count} is out of bounds")

        result = np.empty((count, self.frame_int16), dtype=np.int16)
        end = begin + count
        output_index = 0
        position = begin

        while position < end:
            file_index = bisect_right(self.cumulative, position) - 1
            local_index = position - self.cumulative[file_index]
            take = min(end - position, self.counts[file_index] - local_index)
            mapped = np.memmap(
                self.files[file_index],
                dtype=np.int16,
                mode="r",
                shape=(self.counts[file_index], self.frame_int16),
            )
            result[output_index:output_index + take] = mapped[
                local_index:local_index + take
            ]
            del mapped
            output_index += take
            position += take

        return result


def _check_utc(value: object) -> datetime:
    """Parse a timezone-aware timestamp from the aligned source HDF5."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Source HDF5 timestamp has no timezone: {value}")
    return parsed.astimezone(timezone.utc)


def _reconstruct_reference_center_angle(raw_frame: np.ndarray) -> np.ndarray:
    """Rebuild the source HDF5's centre-Doppler/centre-angle reference values.

    All chirps are used here only to verify the inferred raw-frame offset. The
    mmFormer model input itself uses only the one fixed chirp defined above.
    """
    packed = raw_frame.reshape(
        NUM_TX * LOOPS_PER_TX, NUM_RX * ADC_SAMPLES * 2
    )
    range_window = np.hanning(ADC_SAMPLES).astype(np.float32)
    doppler_window = np.hamming(LOOPS_PER_TX).astype(np.float32)

    # Virtual antenna order used by the original source HDF5's azimuth row 3.
    tx_rx_order = [(2, 2), (2, 0), (1, 2), (1, 0)]
    total = np.zeros(ROI_STOP - ROI_START + 1, dtype=np.complex128)

    for tx in (1, 2):
        tx_chirps = packed[tx::NUM_TX]
        iq = _unpack_iq(tx_chirps)
        range_fft = np.fft.fft(
            iq * range_window[None, None, :], n=FFT_SIZE, axis=-1
        )
        zero_doppler = np.sum(
            range_fft * doppler_window[:, None, None], axis=0
        )
        for antenna_tx, receiver in tx_rx_order:
            if antenna_tx == tx:
                # Preserve the original AoP polarity correction for this row.
                total -= zero_doppler[receiver, ROI_START:ROI_STOP + 1]

    # Angle FFT bin 16 is the zero-spatial-frequency sum of the four antennas.
    return total


def verify_alignment(
    reader: RawFrameReader,
    skipped_start_blocks: int,
    retained_blocks: int,
    source_cube: h5py.Dataset,
) -> dict[str, float]:
    """Check inferred raw alignment against first, middle and last source frames."""
    expected_shape = (SOURCE_BLOCK_FRAMES, 2, 9, 13, 32)
    if source_cube.ndim != 6 or source_cube.shape[1:] != expected_shape:
        raise ValueError(
            "Cannot verify alignment: source beamformed_cube does not have "
            f"the expected (blocks, {expected_shape}) shape"
        )

    first_raw_frame = skipped_start_blocks * SOURCE_BLOCK_FRAMES
    offsets = sorted(
        {
            0,
            retained_blocks * SOURCE_BLOCK_FRAMES // 2,
            retained_blocks * SOURCE_BLOCK_FRAMES - 1,
        }
    )
    errors: dict[str, float] = {}

    for offset in offsets:
        source_block, frame_in_block = divmod(offset, SOURCE_BLOCK_FRAMES)
        raw_frame = reader.get_frames(first_raw_frame + offset, 1)[0]
        calculated = _reconstruct_reference_center_angle(raw_frame)

        real = source_cube[source_block, frame_in_block, 0, 4, :, 16]
        imag = source_cube[source_block, frame_in_block, 1, 4, :, 16]
        expected = np.asarray(real, dtype=np.float64) + 1j * np.asarray(
            imag, dtype=np.float64
        )
        error = np.linalg.norm(calculated - expected) / max(
            np.linalg.norm(expected), 1.0
        )

        if not np.isfinite(error) or error > ALIGNMENT_TOLERANCE:
            raise ValueError(
                f"Raw/HDF5 alignment check failed at retained frame {offset} "
                f"(raw frame {first_raw_frame + offset}): relative complex "
                f"error {error:.5g} exceeds {ALIGNMENT_TOLERANCE:.5g}. "
                "Check ADC file order, radar configuration, TDM packing, "
                "or source-HDF5 block trimming. No output was written."
            )
        errors[str(offset)] = float(error)

    return errors


def export_subject(subject_folder: Path) -> Path:
    """Export one subject's mmFormer input without modifying source files."""
    match = SUBJECT_RE.fullmatch(subject_folder.name)
    if not match:
        raise ValueError(f"Unexpected subject folder name: {subject_folder.name}")

    subject_number = int(match.group(1))
    stem = f"s{subject_number:03d}"
    source_h5 = subject_folder / "processed" / f"{stem}_session.h5"
    raw_folder = subject_folder / "raw" / "radar"
    destination = subject_folder / "processed" / f"{stem}_mmformer.h5"

    if not source_h5.is_file():
        raise FileNotFoundError(source_h5)
    if not raw_folder.is_dir():
        raise NotADirectoryError(raw_folder)
    if destination.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {destination}"
        )

    raw_files = ordered_bin_files(raw_folder)
    reader = RawFrameReader(raw_files)
    raw_full_blocks, incomplete_tail_frames = divmod(
        reader.total, SOURCE_BLOCK_FRAMES
    )

    with h5py.File(source_h5, "r") as source:
        if "raw/radar/beamformed_cube" not in source:
            raise KeyError("Source HDF5 is missing raw/radar/beamformed_cube")
        source_cube = source["raw/radar/beamformed_cube"]
        retained_blocks = int(source_cube.shape[0])

        if source_cube.ndim != 6 or source_cube.shape[1] != SOURCE_BLOCK_FRAMES:
            raise ValueError(
                f"Source HDF5 must contain {SOURCE_BLOCK_FRAMES} frames per block"
            )
        if retained_blocks == 0 or raw_full_blocks < retained_blocks:
            raise ValueError(
                "Raw ADC recording cannot account for the retained source-HDF5 blocks"
            )
        if "raw/radar/localization_cube" in source:
            if source["raw/radar/localization_cube"].shape[0] != retained_blocks:
                raise ValueError("Source beamformed/localization block counts disagree")

        start_utc = _check_utc(source.attrs["start_time_iso"])
        end_utc = _check_utc(source.attrs["end_time_iso"])
        # Retain the source generator's one-second-inclusive metadata check.
        metadata_duration = (end_utc - start_utc).total_seconds() + 1.0
        expected_duration = 18.0 * retained_blocks
        if abs(metadata_duration - expected_duration) > 0.1:
            raise ValueError(
                "Source timestamps disagree with its 18-second block count: "
                f"{metadata_duration:.3f} s vs {expected_duration:.3f} s"
            )

        skipped_start_blocks = raw_full_blocks - retained_blocks
        number_of_windows = retained_blocks - 1
        if number_of_windows < 1:
            raise ValueError(
                "At least two source 18-second blocks are required for a 36-second window"
            )

        alignment_errors = verify_alignment(
            reader, skipped_start_blocks, retained_blocks, source_cube
        )
        has_physiology = "raw/physiology" in source

    print(f"  Complete raw frames: {reader.total}")
    print(f"  Retained source blocks: {retained_blocks}")
    print(f"  Whole 18-second blocks skipped at start: {skipped_start_blocks}")
    print(f"  Incomplete 18-second tail frames discarded: {incomplete_tail_frames}")
    print("  Raw-to-source alignment checks passed:", alignment_errors)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise FileExistsError(
            f"Temporary output already exists; remove it before retrying: {temporary}"
        )

    roi_bins = np.arange(ROI_START, ROI_STOP + 1, dtype=np.int16)
    selected_bins = np.arange(
        MMFORMER_RANGE_START,
        MMFORMER_RANGE_START + NUM_MODEL_RANGES,
        dtype=np.int16,
    )
    if (
        MMFORMER_RANGE_START < ROI_START
        or MMFORMER_RANGE_START + NUM_MODEL_RANGES - 1 > ROI_STOP
    ):
        raise ValueError("The selected mmFormer range bins must lie inside the ROI")
    selected_roi_positions = selected_bins - ROI_START

    try:
        with h5py.File(temporary, "w") as output:
            # Store filenames/relative structure, never machine-specific paths.
            output.attrs["source_session_file"] = source_h5.name
            output.attrs["source_raw_subdirectory"] = "../raw/radar"
            output.attrs["aligned_start_time_iso"] = start_utc.isoformat()
            output.attrs["aligned_end_time_iso"] = end_utc.isoformat()
            output.attrs["raw_blocks_skipped_at_start"] = skipped_start_blocks
            output.attrs["raw_frames_skipped_at_start"] = (
                skipped_start_blocks * SOURCE_BLOCK_FRAMES
            )
            output.attrs["raw_complete_frames_discarded_at_end"] = (
                incomplete_tail_frames
            )
            output.attrs["raw_frame_period_s"] = FRAME_PERIOD_S
            output.attrs["window_duration_s"] = 36.0
            output.attrs["window_stride_s"] = 18.0
            output.attrs["frames_per_window"] = WINDOW_FRAMES
            output.attrs["tx_selected"] = TX_INDEX
            output.attrs["chirp_index_selected"] = CHIRP_INDEX
            output.attrs["loops_per_tx"] = LOOPS_PER_TX
            output.attrs["receiver_count"] = NUM_RX
            output.attrs["range_fft_size"] = FFT_SIZE
            output.attrs["raw_alignment_verification_passed"] = True
            output.attrs["raw_alignment_max_relative_error"] = max(
                alignment_errors.values()
            )
            output.attrs["processing"] = (
                "One fixed TDM chirp per 0.09-s frame; one TX and four RX; "
                "96-point Hann range FFT; complex real/imaginary input; "
                "no Doppler FFT, angle FFT, beamforming, chirp aggregation, "
                "or temporal resampling."
            )
            output.attrs["method_note"] = (
                "HR-oriented mmFormer adaptation. The original WaveBP/mmFormer "
                "used a different radar setup, beamforming and 125-Hz inputs."
            )

            if has_physiology:
                relative_session_path = os.path.relpath(
                    source_h5.resolve(), start=destination.parent.resolve()
                )
                output["source_physiology"] = h5py.ExternalLink(
                    relative_session_path, "/raw/physiology"
                )

            timing = output.create_group("timing")
            window_indices = np.arange(number_of_windows, dtype=np.int32)
            timing.create_dataset("source_block_index", data=window_indices)
            timing.create_dataset(
                "raw_block_index",
                data=np.arange(
                    skipped_start_blocks,
                    skipped_start_blocks + number_of_windows,
                    dtype=np.int32,
                ),
            )
            timing.create_dataset(
                "block_start_s", data=18.0 * window_indices, dtype=np.float64
            )
            timing.create_dataset(
                "block_end_s", data=18.0 * window_indices + 36.0, dtype=np.float64
            )
            string_dtype = h5py.string_dtype("utf-8")
            timing.create_dataset(
                "block_start_utc",
                dtype=string_dtype,
                data=[
                    (start_utc + timedelta(seconds=index * 18.0)).isoformat()
                    for index in range(number_of_windows)
                ],
            )
            timing.create_dataset(
                "block_end_utc",
                dtype=string_dtype,
                data=[
                    (start_utc + timedelta(seconds=index * 18.0 + 36.0)).isoformat()
                    for index in range(number_of_windows)
                ],
            )

            # Keep the complete ROI archive: HeartTimeMixer reads this dataset.
            radar = output.create_group("radar")
            radar.create_dataset("range_bin_indices", data=roi_bins)
            full_archive = radar.create_dataset(
                "rx_iq_36s",
                shape=(number_of_windows, NUM_RX, 2, len(roi_bins), WINDOW_FRAMES),
                dtype="f4",
                chunks=(1, NUM_RX, 2, len(roi_bins), WINDOW_FRAMES),
                compression="lzf",
                shuffle=True,
            )
            full_archive.attrs["axis_order"] = (
                "block,receiver,real_imag,range_bin,time"
            )

            # Keep mmFormer as a separate four-bin view of the same FFT data.
            mmformer = output.create_group("mmformer")
            mmformer.create_dataset("range_bin_indices", data=selected_bins)
            mmformer.attrs["axis_order"] = (
                "block,receiver,real_imag,range_bin,time"
            )
            mmformer.attrs["receiver_view_input_shape"] = "(2, 4, 400)"
            mmformer.attrs["native_duration_s"] = 36.0
            model_input = mmformer.create_dataset(
                "input",
                shape=(
                    number_of_windows,
                    NUM_RX,
                    2,
                    NUM_MODEL_RANGES,
                    WINDOW_FRAMES,
                ),
                dtype="f4",
                chunks=(1, NUM_RX, 2, NUM_MODEL_RANGES, WINDOW_FRAMES),
                compression="lzf",
                shuffle=True,
            )

            range_window = np.hanning(ADC_SAMPLES).astype(np.float32)
            first_raw_frame = skipped_start_blocks * SOURCE_BLOCK_FRAMES
            chunk_size = 32

            for window_index in range(number_of_windows):
                native = np.empty(
                    (WINDOW_FRAMES, NUM_RX, len(roi_bins)),
                    dtype=np.complex64,
                )
                for local_start in range(0, WINDOW_FRAMES, chunk_size):
                    take = min(chunk_size, WINDOW_FRAMES - local_start)
                    raw_frames = reader.get_frames(
                        first_raw_frame
                        + window_index * WINDOW_STRIDE_FRAMES
                        + local_start,
                        take,
                    )
                    iq = one_chirp_four_rx(raw_frames)
                    spectrum = np.fft.fft(
                        iq * range_window[None, None, :],
                        n=FFT_SIZE,
                        axis=-1,
                    )
                    native[local_start:local_start + take] = spectrum[
                        ..., roi_bins
                    ].astype(np.complex64)

                # (time, RX, range) -> (RX, real/imag, range, time)
                real_imag = np.stack((native.real, native.imag), axis=0)
                real_imag = np.transpose(real_imag, (2, 0, 3, 1)).astype(
                    np.float32
                )
                full_archive[window_index] = real_imag
                model_input[window_index] = real_imag[
                    :, :, selected_roi_positions, :
                ]

                if (window_index + 1) % 20 == 0 or window_index + 1 == number_of_windows:
                    print(
                        f"  Exported {window_index + 1}/{number_of_windows} "
                        "36-second windows"
                    )

        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    print(f"  Saved: {destination}")
    return destination


def main() -> None:
    """Process every ICR-DS subject found under DATA_ROOT."""
    if not DATA_ROOT.is_dir():
        raise NotADirectoryError(
            f"Set DATA_ROOT to your ICR-DS directory; not found: {DATA_ROOT}"
        )

    subjects = []
    for folder in DATA_ROOT.iterdir():
        if folder.is_dir():
            match = SUBJECT_RE.fullmatch(folder.name)
            if match:
                subjects.append((int(match.group(1)), folder))
    subjects.sort(key=lambda item: item[0])

    if not subjects:
        raise FileNotFoundError(f"No sub-XXX subject folders found under {DATA_ROOT}")

    summary = {"success": 0, "skipped": 0, "failed": 0}
    for _, subject_folder in subjects:
        print(f"\n[{subject_folder.name}] Processing")
        subject_number = int(SUBJECT_RE.fullmatch(subject_folder.name).group(1))
        destination = (
            subject_folder
            / "processed"
            / f"s{subject_number:03d}_mmformer.h5"
        )

        if destination.exists():
            print(f"  SKIP: {destination.name} already exists; source files unchanged")
            summary["skipped"] += 1
            continue

        try:
            export_subject(subject_folder)
            summary["success"] += 1
        except Exception as error:
            summary["failed"] += 1
            print(
                f"  FAILED (source files unchanged): "
                f"{type(error).__name__}: {error}"
            )

    print("\nBatch finished:", summary)
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
