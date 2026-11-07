import numpy as np
from pathlib import Path
import h5py
import re

DATA_ROOT = Path("path/to/ICR-DS")


def process_radar(h5_path, bin_folder):

    WINDOW_FRAMES = 200

    def list_ordered_bin_files(
            folder: Path,
            pattern=r"adc_data_(?:Raw_)?(\d+)\.bin"
    ):

        files = []
        regex = re.compile(pattern)

        for p in folder.glob("*.bin"):
            m = regex.match(p.name)
            if m:
                files.append((int(m.group(1)), p))

        if not files:
            raise RuntimeError(f"No radar bin files found in {folder}")

        files.sort(key=lambda x: x[0])

        return [p for _, p in files]

    with h5py.File(h5_path, "r+") as f:

        radar_group = f["raw/radar"]
        radar_ds = radar_group["beamformed_cube"]

        # Keep a second cube only for torso/chest localisation.  It has the same
        # layout as beamformed_cube, but is generated from a chirp-mean-subtracted
        # copy of the range FFT.  beamformed_cube itself remains untouched and
        # coherent for the later HR phase path.
        if "localization_cube" in radar_group:
            localization_ds = radar_group["localization_cube"]

            if localization_ds.shape[1:] != radar_ds.shape[1:]:
                raise RuntimeError(
                    "raw/radar/localization_cube has an incompatible shape."
                )

            if localization_ds.shape[0] != radar_ds.shape[0]:
                raise RuntimeError(
                    "beamformed_cube and localization_cube contain different "
                    "numbers of windows; regenerate them together."
                )
        else:
            # Creating a new localisation dataset beside already-populated phase
            # data would make the two window indices refer to different times.
            if radar_ds.shape[0] != 0:
                raise RuntimeError(
                    "beamformed_cube already contains windows but localization_cube "
                    "does not exist. Regenerate from an empty radar dataset so the "
                    "two branches stay aligned."
                )

            create_kwargs = {}

            if radar_ds.chunks is not None:
                create_kwargs["chunks"] = radar_ds.chunks

            if radar_ds.compression is not None:
                create_kwargs["compression"] = radar_ds.compression
                create_kwargs["compression_opts"] = radar_ds.compression_opts

            if radar_ds.shuffle:
                create_kwargs["shuffle"] = radar_ds.shuffle

            if radar_ds.fletcher32:
                create_kwargs["fletcher32"] = radar_ds.fletcher32

            localization_ds = radar_group.create_dataset(
                "localization_cube",
                shape=(0,) + radar_ds.shape[1:],
                maxshape=radar_ds.maxshape,
                dtype=radar_ds.dtype,
                **create_kwargs,
            )

        phase_frame_buffer = []
        localization_frame_buffer = []

        bin_files = list_ordered_bin_files(Path(bin_folder))

        for bin_path in bin_files:

            print(f"\nProcessing {bin_path.name}")

            #
            # Read radar
            #
            _, _, virtual_cube = read_dca1000_truncate(bin_path)

            #
            # Range FFT: create two branches from the SAME range FFT.
            #
            # HR/phase branch: preserve the coherent quasi-static return.
            range_fft_phase = range_fft_preserve_phase(
                virtual_cube
            )

            # Localisation branch: remove the chirp/slow-time mean so static
            # cabin clutter is suppressed when we later search for the torso.
            # This is a separate copy; it does not modify range_fft_phase.
            range_fft_localization = (
                range_fft_phase
                - np.mean(range_fft_phase, axis=2, keepdims=True)
            )

            #
            # Doppler FFT
            #
            # Both branches use the same Hamming window, FFT, fftshift and the
            # same five centre Doppler bins.  Their only difference so far is
            # whether the chirp mean was removed.
            rd_phase, _, _, _ = doppler_fft_and_select(
                range_fft_phase
            )

            rd_localization, _, _, _ = doppler_fft_and_select(
                range_fft_localization
            )

            #
            # Azimuth FFT
            #
            angle_phase, _ = azimuth_angle_fft(
                rd_phase,
                fft_size=32
            )

            angle_localization, _ = azimuth_angle_fft(
                rd_localization,
                fft_size=32
            )

            #
            # angle_* shape: (F, D, 4, A, R)
            #
            # Use the same fixed, complete horizontal AoP row in both branches.
            # Row 3 maps to virtual channels [10, 8, 6, 4].  There is no dynamic
            # row switching or strongest-row phase alignment.
            phase_row = angle_phase[:, :, AZIMUTH_ROW, :, :]
            localization_row = angle_localization[:, :, AZIMUTH_ROW, :, :]

            #
            # Split real / imag and store as (F, 2, D, R, A)
            #
            phase_row = np.stack(
                (
                    phase_row.real,
                    phase_row.imag,
                ),
                axis=1,
            ).astype(np.float32)

            phase_row = np.transpose(
                phase_row,
                (0, 1, 2, 4, 3)
            )

            localization_row = np.stack(
                (
                    localization_row.real,
                    localization_row.imag,
                ),
                axis=1,
            ).astype(np.float32)

            localization_row = np.transpose(
                localization_row,
                (0, 1, 2, 4, 3)
            )

            #
            # Buffer the two branches in lock-step so window n refers to the
            # same 200 radar frames in both datasets.
            #
            phase_frame_buffer.append(phase_row)
            localization_frame_buffer.append(localization_row)

            phase_buffer = np.concatenate(phase_frame_buffer, axis=0)
            localization_buffer = np.concatenate(
                localization_frame_buffer,
                axis=0
            )

            if phase_buffer.shape[0] != localization_buffer.shape[0]:
                raise RuntimeError(
                    "Phase and localisation branches lost frame alignment."
                )

            while phase_buffer.shape[0] >= WINDOW_FRAMES:
                phase_window = phase_buffer[:WINDOW_FRAMES]
                localization_window = localization_buffer[:WINDOW_FRAMES]

                radar_ds.resize(
                    radar_ds.shape[0] + 1,
                    axis=0,
                )

                localization_ds.resize(
                    localization_ds.shape[0] + 1,
                    axis=0,
                )

                radar_ds[-1] = phase_window
                localization_ds[-1] = localization_window

                phase_buffer = phase_buffer[WINDOW_FRAMES:]
                localization_buffer = localization_buffer[WINDOW_FRAMES:]

            #
            # Keep leftover frames for the next .bin file.
            #
            phase_frame_buffer = [phase_buffer]
            localization_frame_buffer = [localization_buffer]

        print("--------------------------------------")
        print("Radar preprocessing complete.")
        print(f"Phase windows stored        : {radar_ds.shape[0]}")
        print(f"Localisation windows stored : {localization_ds.shape[0]}")
        print(f"Frames left over (discarded): {phase_buffer.shape[0]}")
        print("--------------------------------------")


fs = 200 / 18.0      # Hz (≈11.11)


def read_dca1000_truncate(file_name,
                          num_adcsamples=96,
                          num_adcbits=16,
                          num_rx=4,
                          num_tx=3,
                          num_loops=255,
                          is_real=0):
    file_name = Path(file_name)
    adc_data = np.fromfile(file_name, dtype=np.int16)

    if num_adcbits != 16:
        l_max = 2 ** (num_adcbits - 1) - 1
        adc_data = adc_data.copy()
        adc_data[adc_data > l_max] -= 2 ** num_adcbits

    int16_per_frame = num_tx * num_rx * num_loops * num_adcsamples * 2
    num_frames = adc_data.size // int16_per_frame
    adc_data = adc_data[:num_frames * int16_per_frame]

    if adc_data.size == 0:
        raise ValueError("No complete frames found in file.")

    if is_real:
        complex_samples = adc_data.astype(np.float32)
        num_chirps = complex_samples.size // (num_adcsamples * num_rx)
        complex_samples = complex_samples[:num_chirps * num_adcsamples * num_rx]
        lvds = complex_samples.reshape(num_chirps, num_adcsamples * num_rx)
    else:
        if adc_data.size % 4 != 0:
            adc_data = adc_data[:(adc_data.size // 4) * 4]

        complex_samples = np.empty(adc_data.size // 2, dtype=np.complex64)
        complex_samples[0::2] = adc_data[0::4] + 1j * adc_data[2::4]
        complex_samples[1::2] = adc_data[1::4] + 1j * adc_data[3::4]

        num_chirps = complex_samples.size // (num_adcsamples * num_rx)
        complex_samples = complex_samples[:num_chirps * num_adcsamples * num_rx]
        lvds = complex_samples.reshape(num_chirps, num_adcsamples * num_rx)

    adc_cube = lvds.reshape(num_chirps, num_rx, num_adcsamples).transpose(1, 0, 2)

    chirps_per_frame = num_tx * num_loops
    num_frames = num_chirps // chirps_per_frame
    adc_cube = adc_cube[:, :num_frames * chirps_per_frame, :]
    adc_frames = adc_cube.reshape(num_rx, num_frames, chirps_per_frame, num_adcsamples)

    tx_cubes = [adc_frames[:, :, tx::num_tx, :] for tx in range(num_tx)]
    virtual_cube = np.concatenate(tx_cubes, axis=0)

    return adc_cube, adc_frames, virtual_cube


def range_fft_preserve_phase(virtual_cube, nfft=96):
    """Range FFT used by the coherent HR phase path.

    Deliberately does NOT subtract the chirp/slow-time mean.  The quasi-static
    chest return is the carrier whose inter-frame phase variation is needed for
    phase-based vital-sign extraction.
    """
    window = np.hanning(virtual_cube.shape[-1]).astype(np.float32)
    x = virtual_cube * window

    # Range FFT along fast-time axis.
    return np.fft.fft(x, n=nfft, axis=-1)


def range_fft_with_clutter_removal(virtual_cube, nfft=96):
    """Optional detection/localisation version; not used by the HR phase path."""
    range_fft = range_fft_preserve_phase(virtual_cube, nfft=nfft)

    # Static-clutter removal across slow-time/chirp axis.
    return range_fft - np.mean(range_fft, axis=2, keepdims=True)


NEAREST_RANGE_BIN = 3
FARTHEST_RANGE_BIN = 15

KEEP_DOPPLER = 9

# Fixed complete horizontal AoP row used for coherent azimuth phase.
# Row 3 maps to virtual channels [10, 8, 6, 4].
AZIMUTH_ROW = 3


def doppler_fft_and_select(range_fft, keep_doppler=KEEP_DOPPLER):

    x = np.transpose(range_fft, (1, 2, 0, 3))

    # Window across chirps before the Doppler FFT to reduce leakage between
    # neighbouring Doppler bins.  The same fixed window is used every frame.
    doppler_window = np.hamming(x.shape[1]).astype(np.float32)
    x = x * doppler_window[None, :, None, None]

    rd = np.fft.fft(x, axis=1)
    rd = np.fft.fftshift(rd, axes=1)

    n_frames, n_doppler, n_ant, n_range = rd.shape

    centre = n_doppler // 2

    doppler_start = centre - keep_doppler // 2
    doppler_stop  = centre + keep_doppler // 2 + 1

    # Range bins to keep
    fixed_range_window = np.arange(
        NEAREST_RANGE_BIN,
        FARTHEST_RANGE_BIN + 1
    )

    # Crop BOTH doppler AND range
    rd_crop = rd[
        :,
        doppler_start:doppler_stop,
        :,
        fixed_range_window
    ]

    selected_range_windows = np.tile(
        fixed_range_window,
        (n_frames, 1)
    )

    selected_dopplers = np.tile(
        np.arange(doppler_start, doppler_stop),
        (n_frames, 1)
    )

    return (
        rd_crop,
        None,
        selected_range_windows,
        selected_dopplers,
    )


def angle_axis_from_fft(fft_size):

    bins = np.arange(-fft_size // 2, fft_size // 2)

    sin_theta = 2.0 * bins / fft_size

    return np.degrees(
        np.arcsin(
            np.clip(sin_theta, -1.0, 1.0)
        )
    )


def azimuth_angle_fft(rd_cube, fft_size=32):

    cube = rd_cube.copy()

    #
    # TI AoP polarity correction
    # RX1 and RX3 require 180° inversion
    #
    invert = np.array([
        True,  False, True,  False,
        True,  False, True,  False,
        True,  False, True,  False
    ])

    cube[:, :, invert, :] *= -1

    F, D, _, R = cube.shape

    #
    # TI AoP 4x4 mapping
    #
    grid = np.zeros(
        (F, D, 4, 4, R),
        dtype=np.complex64
    )

    #
    # Row 0
    #
    grid[:, :, 0, 0, :] = cube[:, :, 3, :]
    grid[:, :, 0, 1, :] = cube[:, :, 1, :]

    #
    # Row 1
    #
    grid[:, :, 1, 0, :] = cube[:, :, 2, :]
    grid[:, :, 1, 1, :] = cube[:, :, 0, :]

    #
    # Row 2
    #
    grid[:, :, 2, 0, :] = cube[:, :, 11, :]
    grid[:, :, 2, 1, :] = cube[:, :, 9, :]
    grid[:, :, 2, 2, :] = cube[:, :, 7, :]
    grid[:, :, 2, 3, :] = cube[:, :, 5, :]

    #
    # Row 3
    #
    grid[:, :, 3, 0, :] = cube[:, :, 10, :]
    grid[:, :, 3, 1, :] = cube[:, :, 8, :]
    grid[:, :, 3, 2, :] = cube[:, :, 6, :]
    grid[:, :, 3, 3, :] = cube[:, :, 4, :]
    #
    # Horizontal FFT (azimuth)
    #
    az_fft = np.fft.fft(
        grid,
        n=fft_size,
        axis=3
    )

    az_fft = np.fft.fftshift(
        az_fft,
        axes=3
    )

    az_power = np.sum(
        np.abs(az_fft) ** 2,
        axis=2
    )

    return az_fft, az_power


def main():

    subject_folders = sorted(DATA_ROOT.glob("sub-*"))

    print(f"Found {len(subject_folders)} participant folders.")

    for subject_folder in subject_folders:

        subject_id = subject_folder.name

        # Input: raw radar BIN files
        bin_folder = subject_folder / "raw" / "radar"

        # Output target: existing HDF5 recording
        processed_folder = subject_folder / "processed"

        h5_files = sorted(processed_folder.glob("*_session.h5"))

        print(f"\nProcessing participant: {subject_id}")

        # Check input folders
        if not bin_folder.is_dir():
            print(f"  SKIPPED: Missing radar folder: {bin_folder}")
            continue

        if len(h5_files) != 1:
            print(
                f"  SKIPPED: Expected exactly one session HDF5, "
                f"found {len(h5_files)}."
            )
            continue

        h5_path = h5_files[0]

        # Check that matching BIN files exist
        bin_pattern = re.compile(
            r"adc_data_(?:Raw_)?(\d+)\.bin"
        )

        bin_files = [
            p for p in bin_folder.glob("*.bin")
            if bin_pattern.match(p.name)
        ]

        if not bin_files:
            print(f"  SKIPPED: No matching BIN files in {bin_folder}")
            continue

        # IMPORTANT: avoid accidentally appending duplicate windows
        # when rerunning the generator.
        with h5py.File(h5_path, "r") as f:

            if "raw/radar/beamformed_cube" not in f:
                raise KeyError(
                    f"{h5_path}: beamformed_cube is missing."
                )

            radar_ds = f["raw/radar/beamformed_cube"]

            if radar_ds.shape[0] != 0:
                print(
                    f"  SKIPPED: HDF5 already contains "
                    f"{radar_ds.shape[0]} radar windows."
                )
                continue

        print(f"  HDF5: {h5_path}")
        print(f"  BIN folder: {bin_folder}")
        print(f"  BIN files: {len(bin_files)}")

        # Generate both radar branches
        process_radar(
            h5_path=h5_path,
            bin_folder=bin_folder
        )

    print("\n================================")
    print("ALL PARTICIPANTS PROCESSED")
    print("================================")


if __name__ == "__main__":
    main()
