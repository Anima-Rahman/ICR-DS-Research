import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset


# --------------------------------------------------
# Radar cube geometry produced by timeseries_generator.py
# --------------------------------------------------
EXPECTED_FRAMES = 400
EXPECTED_IQ_CHANNELS = 2
EXPECTED_DOPPLERS = 9
EXPECTED_ANGLE_BINS = 32

FIRST_RANGE_BIN = 3
LAST_RANGE_BIN = 15
MIN_VALID_RANGE_BIN = 7
MAX_VALID_RANGE_BIN = 15

# Keep localisation deliberately narrow around zero Doppler, while the
# generator preserves a slightly wider 9-bin neighbourhood for flexibility.
LOCALIZATION_DOPPLERS = 5

# Default radar neighbourhood used for downstream phase-spectrum extraction.
# These are fixed channels for the whole 400-frame / 36 s analysis window; nothing hops.
SPECTROGRAM_ANGLES = 3
SPECTROGRAM_DOPPLERS = 5

range_bins = np.arange(FIRST_RANGE_BIN, LAST_RANGE_BIN + 1)


def angle_axis_from_fft(fft_size):
    bins = np.arange(-fft_size // 2, fft_size // 2)
    sin_theta = 2.0 * bins / fft_size
    return np.degrees(np.arcsin(np.clip(sin_theta, -1.0, 1.0)))


angle_axis = angle_axis_from_fft(EXPECTED_ANGLE_BINS)


def _complex_cube(radar_window):
    """Validate a stored radar window and return its complex cube.

    Input shape:  (F, 2, D, R, A)
    Output shape: (F, D, R, A)
    """
    radar_window = np.asarray(radar_window)

    if radar_window.ndim != 5:
        raise ValueError(
            f"Expected radar window with 5 dimensions (F,2,D,R,A), "
            f"got shape {radar_window.shape}."
        )

    F, IQ, D, R, A = radar_window.shape

    if F != EXPECTED_FRAMES:
        raise ValueError(f"Expected {EXPECTED_FRAMES} frames, got {F}.")
    if IQ != EXPECTED_IQ_CHANNELS:
        raise ValueError(f"Expected I/Q dimension of 2, got {IQ}.")
    if D != EXPECTED_DOPPLERS:
        raise ValueError(
            f"Expected {EXPECTED_DOPPLERS} stored Doppler bins, got {D}. "
            "Regenerate the radar cube after changing KEEP_DOPPLER to 9."
        )
    if R != len(range_bins):
        raise ValueError(
            f"Expected {len(range_bins)} stored range bins "
            f"({FIRST_RANGE_BIN}..{LAST_RANGE_BIN}), got {R}."
        )
    if A != EXPECTED_ANGLE_BINS:
        raise ValueError(f"Expected {EXPECTED_ANGLE_BINS} angle bins, got {A}.")

    return radar_window[:, 0] + 1j * radar_window[:, 1]


def _centre_indices(total_bins, keep_bins):
    """Return a symmetric, fixed set of indices around the centre bin."""
    if keep_bins < 1 or keep_bins > total_bins:
        raise ValueError(
            f"keep_bins must be between 1 and {total_bins}, got {keep_bins}."
        )

    # Odd counts give a truly symmetric neighbourhood containing the DC bin.
    if keep_bins % 2 == 0:
        raise ValueError(
            "Use an odd Doppler count (1, 3, 5, 7, 9) so the neighbourhood "
            "is symmetric around the centre bin."
        )

    centre = total_bins // 2
    half = keep_bins // 2
    return np.arange(centre - half, centre + half + 1, dtype=np.int64)


def _nearest_angle_indices(centre_angle, total_angles, count):
    """Return the centre angle first, then its nearest fixed neighbours."""
    if count < 1 or count > total_angles:
        raise ValueError(
            f"count must be between 1 and {total_angles}, got {count}."
        )

    ordered = sorted(
        range(total_angles),
        key=lambda a: (abs(a - centre_angle), a),
    )
    return np.asarray(ordered[:count], dtype=np.int64)


def _peak_to_median(values):
    """Simple localisation confidence diagnostic; no automatic rejection."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan

    median = float(np.median(values))
    peak = float(np.max(values))

    if median <= 0:
        return np.inf if peak > 0 else 1.0

    return peak / median


def select_fixed_chest_roi(
        localization_window,
        num_angles=SPECTROGRAM_ANGLES,
        localization_dopplers=LOCALIZATION_DOPPLERS):
    """Select one fixed chest range and a small fixed angle neighbourhood.

    Parameters
    ----------
    localization_window : ndarray
        Clutter-removed localisation cube with shape (400, 2, 9, 13, 32).
    num_angles : int
        Number of fixed neighbouring angle bins to keep around the selected
        chest centre.  Default is 3.
    localization_dopplers : int
        Number of central Doppler bins used only for localisation.  Default is
        5, matching the near-zero-Doppler localisation idea while preserving 9
        bins in the stored cube.

    Returns
    -------
    dict
        Minimal ROI description plus localisation confidence diagnostics.

    Notes
    -----
    The selected range and angle neighbourhood are fixed for all 400 frames.
    There is no frame-by-frame tracking or winner switching.
    """
    complex_cube = _complex_cube(localization_window)
    _, D, _, A = complex_cube.shape

    doppler_indices = _centre_indices(D, localization_dopplers)

    # Localisation uses power only.  Keep Doppler bins as a neighbourhood and
    # sum their power rather than selecting a different winner each frame.
    power_cube = np.abs(complex_cube[:, doppler_indices]) ** 2

    valid_range_indices = np.where(
        (range_bins >= MIN_VALID_RANGE_BIN) &
        (range_bins <= MAX_VALID_RANGE_BIN)
    )[0]

    if valid_range_indices.size == 0:
        raise ValueError(
            "No valid range bins remain after applying the configured range gate."
        )

    # One fixed range for the whole window:
    # sum over frames, localisation Dopplers, and angles.
    range_power = power_cube.sum(axis=(0, 1, 3))
    valid_range_power = range_power[valid_range_indices]

    best_range_index = int(
        valid_range_indices[np.argmax(valid_range_power)]
    )

    # One centre angle at that fixed range:
    # sum over frames and localisation Dopplers.
    angle_power = power_cube[:, :, best_range_index, :].sum(axis=(0, 1))
    centre_angle_index = int(np.argmax(angle_power))

    angle_indices = _nearest_angle_indices(
        centre_angle_index,
        A,
        num_angles,
    )

    return {
        "range_index": best_range_index,
        "range_bin": int(range_bins[best_range_index]),
        "centre_angle_index": centre_angle_index,
        "centre_angle_deg": float(angle_axis[centre_angle_index]),
        "angle_indices": angle_indices,
        "angle_degs": angle_axis[angle_indices].astype(np.float32),
        "localization_doppler_indices": doppler_indices,
        "localization_doppler_offsets": (
            doppler_indices - D // 2
        ).astype(np.int64),
        "range_peak_to_median": _peak_to_median(valid_range_power),
        "angle_peak_to_median": _peak_to_median(angle_power),
    }


def extract_fixed_complex_roi(
        radar_window,
        roi,
        num_dopplers=SPECTROGRAM_DOPPLERS):
    """Extract fixed coherent complex channels from the HR/phase cube.

    Parameters
    ----------
    radar_window : ndarray
        NON-clutter-removed HR cube with shape (400, 2, 9, 13, 32).
    roi : dict
        Output of select_fixed_chest_roi().
    num_dopplers : int
        Number of fixed central Doppler bins to retain.  Default is 5.
        Set to 1 for centre-Doppler-only experiments or 9 to keep everything.

    Returns
    -------
    dict
        ``series`` has shape (num_angles, num_dopplers, 400), complex64.
        Every channel uses the same range, angle and Doppler index for the
        complete 400-frame window.
        Phase extraction, unwrapping, linear detrending, Hann windowing,
        and FFT-based spectral analysis are performed downstream.
    """
    complex_cube = _complex_cube(radar_window)
    F, D, R, A = complex_cube.shape

    range_index = int(roi["range_index"])
    angle_indices = np.asarray(roi["angle_indices"], dtype=np.int64)
    doppler_indices = _centre_indices(D, num_dopplers)

    if not (0 <= range_index < R):
        raise IndexError(
            f"Range index {range_index} is outside stored range dimension {R}."
        )

    if np.any((angle_indices < 0) | (angle_indices >= A)):
        raise IndexError(
            f"At least one angle index is outside stored angle dimension {A}."
        )

    # Advanced indexing here gives (F, D_selected, A_selected); transpose to
    # (A_selected, D_selected, F) so each [angle, doppler] entry is one fixed
    # complex time series ready for independent phase processing.
    selected = complex_cube[:, doppler_indices, range_index, :]
    selected = selected[:, :, angle_indices]
    series = np.transpose(selected, (2, 1, 0)).astype(
        np.complex64,
        copy=False,
    )

    return {
        "series": series,
        "range_index": range_index,
        "range_bin": int(range_bins[range_index]),
        "angle_indices": angle_indices,
        "angle_degs": angle_axis[angle_indices].astype(np.float32),
        "doppler_indices": doppler_indices,
        "doppler_offsets": (doppler_indices - D // 2).astype(np.int64),
        # Preserve channel-strength information for diagnostics/weighting.
        # Do not normalize each channel to unit variance here.
        "mean_magnitude": np.mean(np.abs(series), axis=-1).astype(np.float32),
    }


class RadarDataset(Dataset):
    """Load preprocessed .pt windows. Each y value is one scalar HR target in bpm."""

    def __init__(self, processed_folder):
        self.samples = []
        self._cached_file = None
        self._cached_sample = None

        for file in sorted(Path(processed_folder).glob("*.pt")):
            data = torch.load(file, map_location="cpu")
            num_windows = data["x"].shape[0]

            for window in range(num_windows):
                self.samples.append((file, window))

        print(f"Dataset contains {len(self.samples)} windows.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file, window = self.samples[idx]

        if file != self._cached_file:
            self._cached_sample = torch.load(file, map_location="cpu")
            self._cached_file = file

        return (
            self._cached_sample["x"][window],
            self._cached_sample["y"][window],
        )
