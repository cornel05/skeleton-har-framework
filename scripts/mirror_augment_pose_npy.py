import argparse
from pathlib import Path

import numpy as np


def mirror_coco17_sequence(sequence: np.ndarray) -> np.ndarray:
    """Mirror centered COCO17 sequence horizontally and swap left/right joints."""
    if sequence.ndim != 3:
        raise ValueError(f"Expected 3D array (T, K, C), got {sequence.shape}")
    if sequence.shape[1] < 17:
        raise ValueError(f"Expected at least 17 keypoints, got {sequence.shape}")
    if sequence.shape[2] < 2:
        raise ValueError(f"Expected at least 2 channels [x, y], got {sequence.shape}")

    mirrored = sequence.astype(np.float32, copy=True)
    # Input is expected to be centered around body origin, so x-mirror is sign inversion.
    mirrored[:, :, 0] = -mirrored[:, :, 0]

    left_right_pairs = (
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (9, 10),
        (11, 12),
        (13, 14),
        (15, 16),
    )
    for left_idx, right_idx in left_right_pairs:
        tmp = mirrored[:, left_idx, :].copy()
        mirrored[:, left_idx, :] = mirrored[:, right_idx, :]
        mirrored[:, right_idx, :] = tmp

    return mirrored


def target_path_for(source_path: Path, base_folder: Path, output_folder: Path | None, inplace: bool) -> Path:
    if inplace:
        return source_path

    rel = source_path.relative_to(base_folder)
    stemmed = rel.with_name(f"{rel.stem}_mirror{rel.suffix}")
    if output_folder is not None:
        return output_folder / stemmed

    return source_path.with_name(f"{source_path.stem}_mirror{source_path.suffix}")


def process_folder(folder: Path, output_folder: Path | None, inplace: bool) -> None:
    npy_files = sorted(folder.rglob("*.npy"))
    if not npy_files:
        print(f"No .npy files found under: {folder}")
        return

    processed = 0
    skipped = 0

    for npy_path in npy_files:
        # Avoid augmenting files that were already mirrored in prior runs.
        if npy_path.stem.endswith("_mirror"):
            continue

        try:
            data = np.load(npy_path)
            mirrored = mirror_coco17_sequence(data)

            target = target_path_for(npy_path, folder, output_folder, inplace)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, mirrored)
            processed += 1
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"[WARN] Skipped {npy_path}: {exc}")

    print(f"Processed files: {processed}")
    print(f"Skipped files: {skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply one-pass mirror augmentation to COCO17 pose .npy files. "
            "Expected format: centered skeleton sequences with x/y in channels 0/1."
        )
    )
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Root folder containing .npy pose files (searched recursively).",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help=(
            "Optional output root folder. If omitted and --inplace is not set, "
            "files are saved next to originals with suffix _mirror.npy."
        ),
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input files in place.",
    )

    args = parser.parse_args()

    if args.output_folder is not None and args.inplace:
        raise ValueError("Use either --output-folder or --inplace, not both.")

    return args


def main() -> None:
    args = parse_args()

    folder = args.folder.resolve()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist or is not a directory: {folder}")

    output_folder = args.output_folder.resolve() if args.output_folder is not None else None

    process_folder(
        folder=folder,
        output_folder=output_folder,
        inplace=args.inplace,
    )


if __name__ == "__main__":
    main()
