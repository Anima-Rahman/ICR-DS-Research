from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.signal import detrend
from tqdm import tqdm

from dataset import (
    extract_fixed_complex_roi,
    select_fixed_chest_roi,
)


# --------------------------------------------------
# Radar / window timing
# --------------------------------------------------
# The radar frame period is 90 ms, so the physical frame rate is exactly:
FRAME_PERIOD_S = 0.090
FS = 1.0 / FRAME_PERIOD_S

# One model example spans 36 s = 400 radar frames exactly.
WINDOW_FRAMES = 400
WINDOW_SECONDS = WINDOW_FRAMES * FRAME_PERIOD_S

# Existing HDF5 files may still be stored as non-overlapping 200-frame
# (18 s) chunks.  This script joins two adjacent source chunks to form each
# 36 s analysis window, so the raw radar generator does NOT have to be rerun.
#
# With 200-frame source chunks, consecutive 36 s examples use a one-chunk
# stride: [0,1], [1,2], [2,3], ... => 18 s analysis stride / 50% overlap.
LEGACY_SOURCE_WINDOW_FRAMES = 200

# --------------------------------------------------
# Fixed radar neighbourhood used for the phase spectrum
# --------------------------------------------------
NUM_ANGLES = 3
NUM_DOPPLERS = 5
LOCALIZATION_DOPPLERS = 5

# --------------------------------------------------
# Full 36 s phase spectrum
# --------------------------------------------------
# Use the ENTIRE 400-frame / 36 s phase trace for one spectrum.
#
# Fundamental Fourier spacing:
#     1 / 36 s = 0.02778 Hz ~= 1.67 bpm.
#
# NFFT=4096 zero-pads the 400-sample Hann-windowed signal to provide a smooth
# frequency grid. It does NOT create finer physical resolution than the
# 36-second observation itself.
#
# No narrow cardiac bandpass is applied before the FFT.
FFT_NFFT = 4096
SPECTRUM_FMIN_HZ = 0.10
SPECTRUM_FMAX_HZ = 4.00
SPECTRUM_EPS = 1e-12

# --------------------------------------------------
# Scalar HR target from IBI/beat timestamps
# --------------------------------------------------
# ibi_ts is interpreted exactly as beat timestamps in seconds from recording
# start.  For each 36 s radar window, count timestamps in the half-open interval
# [window_start_s, window_stop_s), then convert that count directly to bpm:
#
#     HR_bpm = beat_count * 60 / WINDOW_SECONDS
#
# With a 36 s window, one beat corresponds to 60/36 = 1.6667 bpm.
HR_FMIN_BPM = 36.0
HR_FMAX_BPM = 210.0


DATA_ROOT = Path("path/to/ICR-DS")

input_folder = DATA_ROOT


def phase_spectrum(complex_series, fs=FS):
    """Convert fixed coherent radar channels into full-window phase spectra.

    Input shape:  (angles, dopplers, 400) complex
    Output shape: (angles, dopplers, frequencies) log-power dB

    Processing per fixed channel:
        complex -> phase -> unwrap -> linear detrend -> 400-sample Hann
        -> 4096-point one-sided FFT -> power -> log10 power -> 0.1..4 Hz crop

    No cardiac bandpass is applied.
    """
    complex_series = np.asarray(complex_series)

    if complex_series.ndim != 3:
        raise ValueError(
            "Expected complex_series with shape (angles, dopplers, frames), "
            f"got {complex_series.shape}."
        )

    if complex_series.shape[-1] != WINDOW_FRAMES:
        raise ValueError(
            f"Expected {WINDOW_FRAMES} frames, got {complex_series.shape[-1]}."
        )

    # Unwrap each fixed angle/Doppler channel independently.
    phase = np.unwrap(np.angle(complex_series), axis=-1)

    # Remove only linear phase drift.
    phase = detrend(phase, axis=-1, type="linear")

    # One Hann window over the entire 36-second observation.
    hann = np.hanning(WINDOW_FRAMES).astype(np.float32)
    windowed_phase = phase * hann

    fft_values = np.fft.rfft(
        windowed_phase,
        n=FFT_NFFT,
        axis=-1,
    )

    frequencies_hz = np.fft.rfftfreq(
        FFT_NFFT,
        d=1.0 / fs,
    )

    keep = (
        (frequencies_hz >= SPECTRUM_FMIN_HZ)
        & (frequencies_hz <= SPECTRUM_FMAX_HZ)
    )

    frequencies_hz = frequencies_hz[keep]
    fft_values = fft_values[..., keep]

    power = np.abs(fft_values) ** 2
    spectrum_db = 10.0 * np.log10(power + SPECTRUM_EPS)

    return (
        spectrum_db.astype(np.float32),
        frequencies_hz.astype(np.float32),
        phase.astype(np.float32),
    )

def window_hr_from_beat_count(beat_times_s, window_start_s, window_stop_s):
    """Return one scalar HR target by counting beat timestamps in the window.

    ``beat_times_s`` must contain beat timestamps in seconds relative to the
    same recording start as the radar.

    The interval is half-open: [window_start_s, window_stop_s).  This makes the
    boundary rule deterministic: a beat exactly at the stop time belongs to
    the following window, not the current one.

    Returns
    -------
    hr_bpm : float or None
        Scalar count-based HR in beats/minute.
    beat_count : int
        Number of beat timestamps inside the 36 s analysis window.
    """
    beat_times_s = np.asarray(beat_times_s, dtype=np.float64)

    in_window = (
        np.isfinite(beat_times_s)
        & (beat_times_s >= window_start_s)
        & (beat_times_s < window_stop_s)
    )

    beat_count = int(np.count_nonzero(in_window))

    if beat_count == 0:
        return None, 0

    hr_bpm = beat_count * 60.0 / WINDOW_SECONDS

    if not (HR_FMIN_BPM <= hr_bpm <= HR_FMAX_BPM):
        return None, beat_count

    return float(hr_bpm), beat_count

def process_recording(h5_path):
    """Process one HDF5 recording into full 36 s phase spectra with scalar count-based HR targets."""
    with h5py.File(h5_path, "r") as f:
        radar_group = f["raw/radar"]

        if "beamformed_cube" not in radar_group:
            raise KeyError(f"{h5_path}: raw/radar/beamformed_cube is missing.")
        if "localization_cube" not in radar_group:
            raise KeyError(
                f"{h5_path}: raw/radar/localization_cube is missing. "
                "Regenerate radar preprocessing with the two-branch generator."
            )

        radar = radar_group["beamformed_cube"]
        localization = radar_group["localization_cube"]

        if len(radar) != len(localization):
            raise RuntimeError(
                f"{h5_path}: beamformed_cube has {len(radar)} windows but "
                f"localization_cube has {len(localization)}."
            )

        if radar.ndim != 6 or localization.ndim != 6:
            raise ValueError(
                f"{h5_path}: expected stored radar datasets with shape "
                "(windows, frames, 2, doppler, range, angle)."
            )

        if radar.shape[1:] != localization.shape[1:]:
            raise RuntimeError(
                f"{h5_path}: beamformed_cube and localization_cube shapes differ: "
                f"{radar.shape[1:]} vs {localization.shape[1:]}."
            )

        source_window_frames = int(radar.shape[1])

        if source_window_frames == LEGACY_SOURCE_WINDOW_FRAMES:
            # Existing 18 s HDF5: pair adjacent chunks to make 36 s.
            source_windows_per_analysis = 2
        elif source_window_frames == WINDOW_FRAMES:
            # Also support future HDF5 files that are written directly as 36 s.
            source_windows_per_analysis = 1
        else:
            raise ValueError(
                f"{h5_path}: each stored radar window has {source_window_frames} frames. "
                f"Expected either {LEGACY_SOURCE_WINDOW_FRAMES} (18 s source chunks) "
                f"or {WINDOW_FRAMES} (36 s source windows)."
            )

        source_window_seconds = source_window_frames * FRAME_PERIOD_S
        analysis_stride_seconds = source_window_seconds

        num_analysis_windows = (
            len(radar) - source_windows_per_analysis + 1
        )

        if num_analysis_windows <= 0:
            return None

        beat_times_s = np.asarray(
            f["raw"]["physiology"]["ibi_ts"],
            dtype=np.float64,
        )
        beat_times_s = beat_times_s[np.isfinite(beat_times_s)]
        beat_times_s = np.sort(beat_times_s)

        # Keep the old script's interpretation explicit: ibi_ts must be beat
        # timestamps in seconds relative to recording start.
        if beat_times_s.size and beat_times_s[0] < 0:
            raise ValueError(f"{h5_path}: ibi_ts contains negative timestamps.")

        all_x = []
        all_y = []
        all_beat_count = []
        all_window_start_s = []
        all_window_stop_s = []
        all_range_bins = []
        all_centre_angles_deg = []
        all_angle_degs = []
        all_doppler_offsets = []
        all_mean_magnitude = []
        all_range_confidence = []
        all_angle_confidence = []

        spectrum_frequencies_hz = None

        for analysis_idx in range(num_analysis_windows):
            # The analysis start is always tied to an exact stored radar-chunk
            # boundary, which itself is an exact multiple of the 90 ms frame period.
            window_start_s = analysis_idx * analysis_stride_seconds
            window_stop_s = window_start_s + WINDOW_SECONDS

            hr_bpm, beat_count = window_hr_from_beat_count(
                beat_times_s,
                window_start_s,
                window_stop_s,
            )

            # Skip only if the timestamp count cannot produce a plausible target.
            if hr_bpm is None:
                continue

            if source_windows_per_analysis == 1:
                localization_window = np.asarray(
                    localization[analysis_idx],
                    dtype=np.float32,
                )
                radar_window = np.asarray(
                    radar[analysis_idx],
                    dtype=np.float32,
                )
            else:
                # Join two adjacent 18 s chunks.  HDF5 chunk boundaries do not
                # alter the stored complex samples, so this does not introduce
                # a synthetic phase reset by itself.
                localization_window = np.concatenate(
                    (
                        localization[analysis_idx],
                        localization[analysis_idx + 1],
                    ),
                    axis=0,
                ).astype(np.float32, copy=False)

                radar_window = np.concatenate(
                    (
                        radar[analysis_idx],
                        radar[analysis_idx + 1],
                    ),
                    axis=0,
                ).astype(np.float32, copy=False)

            if radar_window.shape[0] != WINDOW_FRAMES:
                raise RuntimeError(
                    f"{h5_path}: assembled analysis window has "
                    f"{radar_window.shape[0]} frames instead of {WINDOW_FRAMES}."
                )

            # Localisation uses the clutter-removed branch only.
            roi = select_fixed_chest_roi(
                localization_window,
                num_angles=NUM_ANGLES,
                localization_dopplers=LOCALIZATION_DOPPLERS,
            )

            # Phase comes only from the non-clutter-removed HR branch.
            extracted = extract_fixed_complex_roi(
                radar_window,
                roi,
                num_dopplers=NUM_DOPPLERS,
            )

            spectrum_db, freqs_hz, _ = phase_spectrum(
                extracted["series"],
                fs=FS,
            )

            if spectrum_frequencies_hz is None:
                spectrum_frequencies_hz = freqs_hz
            else:
                if not np.array_equal(spectrum_frequencies_hz, freqs_hz):
                    raise RuntimeError("FFT frequency axis changed between windows.")

            # Flatten angle and Doppler only. One sample is now (15, F).
            # Channel order remains angle-major, then Doppler-minor.
            x = spectrum_db.reshape(
                NUM_ANGLES * NUM_DOPPLERS,
                spectrum_db.shape[-1],
            )

            all_x.append(torch.from_numpy(x))
            all_y.append(float(hr_bpm))
            all_beat_count.append(beat_count)
            all_window_start_s.append(window_start_s)
            all_window_stop_s.append(window_stop_s)
            all_range_bins.append(extracted["range_bin"])
            all_centre_angles_deg.append(roi["centre_angle_deg"])
            all_angle_degs.append(extracted["angle_degs"])
            all_doppler_offsets.append(extracted["doppler_offsets"])
            all_mean_magnitude.append(extracted["mean_magnitude"])
            all_range_confidence.append(roi["range_peak_to_median"])
            all_angle_confidence.append(roi["angle_peak_to_median"])

        if not all_x:
            return None

        return {
            "x": torch.stack(all_x),
            # One scalar target per 36 s sample, in bpm.
            "y": torch.tensor(all_y, dtype=torch.float32),
            # Diagnostics: exactly how the scalar target was produced.
            "beat_count": torch.tensor(all_beat_count, dtype=torch.int64),
            "window_start_s": torch.tensor(all_window_start_s, dtype=torch.float64),
            "window_stop_s": torch.tensor(all_window_stop_s, dtype=torch.float64),
            "spectrum_frequencies_hz": torch.from_numpy(
                spectrum_frequencies_hz
            ),
            # Backward-compatible alias for plotting code.
            "spectrogram_frequencies_hz": torch.from_numpy(
                spectrum_frequencies_hz
            ),
            "range_bin": torch.tensor(all_range_bins, dtype=torch.int64),
            "centre_angle_deg": torch.tensor(
                all_centre_angles_deg,
                dtype=torch.float32,
            ),
            "angle_degs": torch.from_numpy(
                np.stack(all_angle_degs).astype(np.float32)
            ),
            "doppler_offsets": torch.from_numpy(
                np.stack(all_doppler_offsets).astype(np.int64)
            ),
            "mean_magnitude": torch.from_numpy(
                np.stack(all_mean_magnitude).astype(np.float32)
            ),
            "range_peak_to_median": torch.tensor(
                all_range_confidence,
                dtype=torch.float32,
            ),
            "angle_peak_to_median": torch.tensor(
                all_angle_confidence,
                dtype=torch.float32,
            ),
            "recording": str(h5_path),
            "channel_layout": "angle-major, doppler-minor",
            "num_angles": NUM_ANGLES,
            "num_dopplers": NUM_DOPPLERS,
            "frame_period_s": FRAME_PERIOD_S,
            "fs_hz": FS,
            "window_frames": WINDOW_FRAMES,
            "window_seconds": WINDOW_SECONDS,
            "analysis_stride_seconds": analysis_stride_seconds,
            "source_window_frames": source_window_frames,
            "source_window_seconds": source_window_seconds,
            "representation": "full_36s_phase_spectrum",
            "fft_nfft": FFT_NFFT,
            "spectrum_fmin_hz": SPECTRUM_FMIN_HZ,
            "spectrum_fmax_hz": SPECTRUM_FMAX_HZ,
            "target_definition": "beat_count * 60 / window_seconds",
            "target_units": "bpm",
        }


def main():

    h5_files = sorted(
        input_folder.glob("sub-*/processed/*_session.h5")
    )

    print(f"Found {len(h5_files)} recordings.")

    sample_number = 0

    for h5_path in tqdm(h5_files):

        # Save the PT beside its corresponding HDF5 file
        out_path = h5_path.with_suffix(".pt")

        if out_path.exists():
            print(f"\nSkipping existing file: {out_path}")
            continue

        processed = process_recording(h5_path)

        if processed is None:
            print(f"\nNo valid windows: {h5_path.name}")
            continue

        torch.save(processed, out_path)

        sample_number += int(processed["x"].shape[0])

        print(f"\nSaved: {out_path}")

    print(f"\nTotal samples saved: {sample_number}")


if __name__ == "__main__":
    main()
