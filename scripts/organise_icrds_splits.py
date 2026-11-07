
#!/usr/bin/env python3
"""
Organise ICR-DS data into subject-level Train/Test folders.

Source:
    ICR-DS/
        sub-001/
            processed/
                s001_session.pt
                s001_hrconvnet.h5
                s001_alizadeh.h5
                s001_vitanet.h5
                s001_mmformer.h5

Destination:
    ICR-DS/
        Train/
            STAT/
            hrconvnet/
            alizadeh/
            vitanet/
            mmformer/
        Test/
            STAT/
            hrconvnet/
            alizadeh/
            vitanet/
            mmformer/

STAT uses .pt files.
All other methods use HDF5 files.

Original files are never moved or deleted.
Existing destination files are not overwritten.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path


# ---------------------------------------------------------
# Paths and subject split
# ---------------------------------------------------------

DATA_ROOT = Path("path/to/ICR-DS")

TEST_SUBJECTS = {3, 5, 8, 10, 18}
EXCLUDED_SUBJECTS = {9, 11}

SUBJECT_RE = re.compile(r"^sub-(\d+)$", re.IGNORECASE)


# ---------------------------------------------------------
# Files required by each method
# ---------------------------------------------------------

# STAT uses the preprocessed PyTorch file.
STAT_SUFFIX = "session"

# Other methods use their HDF5 files.
HDF5_METHODS = {
    "hrconvnet": "hrconvnet",
    "alizadeh": "alizadeh",
    "vitanet": "vitanet",
    "mmformer": "mmformer",
}


def output_filename(subject: int, suffix: str, extension: str) -> str:
    return f"s{subject:03d}_{suffix}.{extension}"


# ---------------------------------------------------------
# Repair HDF5 external links in copied files
# ---------------------------------------------------------

def relink_original_session(
    staged_file: Path,
    source_dir: Path,
    subject: int,
) -> bool:
    """
    Repair a source_physiology external link in a copied HDF5 file.

    The original session HDF5 file is never modified.
    """

    try:
        import h5py
    except ImportError:
        print(
            f"WARNING: h5py is not installed; "
            f"cannot check external links in {staged_file.name}"
        )
        return False

    session_file = source_dir / output_filename(
        subject, "session", "h5"
    )

    with h5py.File(staged_file, "r+") as f:

        link = f.get("source_physiology", getlink=True)

        if not isinstance(link, h5py.ExternalLink):
            return False

        if not session_file.is_file():
            raise FileNotFoundError(
                f"Original session HDF5 file not found: {session_file}"
            )

        if link.path != "/raw/physiology":
            raise ValueError(
                f"Unexpected external link destination: {link.path!r}"
            )

        relative_path = os.path.relpath(
            session_file,
            start=staged_file.parent,
        )

        del f["source_physiology"]

        f["source_physiology"] = h5py.ExternalLink(
            relative_path,
            link.path,
        )

    return True


# ---------------------------------------------------------
# Main organisation script
# ---------------------------------------------------------

def main():

    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_ROOT}"
        )

    subject_dirs = []

    for folder in sorted(DATA_ROOT.iterdir()):

        if not folder.is_dir():
            continue

        match = SUBJECT_RE.fullmatch(folder.name)

        if match:
            subject_dirs.append(
                (int(match.group(1)), folder)
            )

    if not subject_dirs:
        raise RuntimeError(
            f"No subject folders found under {DATA_ROOT}"
        )

    copied = 0
    skipped = 0
    missing = 0
    repaired = 0

    print("Organising ICR-DS dataset")
    print(f"Dataset root: {DATA_ROOT}")
    print(f"Test subjects: {sorted(TEST_SUBJECTS)}")
    print(f"Excluded subjects: {sorted(EXCLUDED_SUBJECTS)}")
    print()

    # -----------------------------------------------------
    # Process each subject
    # -----------------------------------------------------

    for subject, subject_folder in subject_dirs:

        if subject in EXCLUDED_SUBJECTS:
            print(f"EXCLUDED: sub-{subject:03d}")
            continue

        split = (
            "Test"
            if subject in TEST_SUBJECTS
            else "Train"
        )

        processed_folder = subject_folder / "processed"

        if not processed_folder.is_dir():
            print(
                f"MISSING processed folder: {processed_folder}"
            )
            missing += 1
            continue

        print(
            f"\nSubject sub-{subject:03d} -> {split}"
        )

        # -------------------------------------------------
        # STAT: copy the .pt file
        # -------------------------------------------------

        stat_filename = output_filename(
            subject, STAT_SUFFIX, "pt"
        )

        stat_source = processed_folder / stat_filename

        stat_destination = (
            DATA_ROOT / split / "STAT" / stat_filename
        )

        if not stat_source.is_file():

            print(f"  MISSING STAT: {stat_source}")
            missing += 1

        elif stat_destination.exists():

            print(f"  KEEP STAT: {stat_destination}")
            skipped += 1

        else:

            stat_destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(
                stat_source,
                stat_destination,
            )

            print(f"  COPIED STAT: {stat_destination}")
            copied += 1

        # -------------------------------------------------
        # Other methods: copy HDF5 files
        # -------------------------------------------------

        for method, suffix in HDF5_METHODS.items():

            filename = output_filename(
                subject, suffix, "h5"
            )

            source = processed_folder / filename

            destination = (
                DATA_ROOT / split / method / filename
            )

            if not source.is_file():

                print(f"  MISSING {method}: {source}")
                missing += 1
                continue

            if destination.exists():

                print(f"  KEEP {method}: {destination}")
                skipped += 1
                continue

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_file = destination.with_name(
                destination.name + ".copying"
            )

            try:

                shutil.copy2(
                    source,
                    temporary_file,
                )

                if relink_original_session(
                    temporary_file,
                    processed_folder,
                    subject,
                ):
                    repaired += 1

                os.replace(
                    temporary_file,
                    destination,
                )

                print(f"  COPIED {method}: {destination}")
                copied += 1

            except Exception as error:

                print(
                    f"  ERROR copying {source}: {error}"
                )

                if temporary_file.exists():
                    temporary_file.unlink()

                raise

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n------------------------------------")
    print("DATA ORGANISATION COMPLETE")
    print("------------------------------------")
    print(f"Files copied: {copied}")
    print(f"Existing files skipped: {skipped}")
    print(f"Missing files: {missing}")
    print(f"HDF5 external links repaired: {repaired}")
    print("------------------------------------")
    print("Original files remain untouched.")


if __name__ == "__main__":
    main()
