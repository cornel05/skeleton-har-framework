import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def compute_scale_per_frame(sequence: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute neck-to-mid-hip distance per frame for COCO17 keypoints.

    Neck is approximated as midpoint of left/right shoulders (5, 6).
    Mid-hip is midpoint of left/right hips (11, 12).

    Returns:
    - scale: shape (T,), Euclidean distance in x/y for each frame.
    - valid: shape (T,), whether scale is usable for division.
    """
    coords_xy = sequence[:, :, :2]

    neck_xy = (coords_xy[:, 5, :] + coords_xy[:, 6, :]) / 2.0
    mid_hip_xy = (coords_xy[:, 11, :] + coords_xy[:, 12, :]) / 2.0

    scale = np.linalg.norm(neck_xy - mid_hip_xy, axis=1)
    valid = scale > eps

    # If visibility/confidence is present (x, y, v), require key joints to be visible.
    if sequence.shape[2] >= 3:
        vis = sequence[:, :, 2]
        valid &= vis[:, 5] > 0.0
        valid &= vis[:, 6] > 0.0
        valid &= vis[:, 11] > 0.0
        valid &= vis[:, 12] > 0.0

    return scale, valid


def unit_scale_sequence(sequence: np.ndarray, eps: float) -> np.ndarray:
    """
    Apply per-frame unit scaling on x/y coordinates.

    Input shape is expected to be (T, K, C), where K >= 13 and C >= 3
    for COCO17 [x, y, visibility].
    """
    if sequence.ndim != 3:
        raise ValueError(f"Expected 3D array (T, K, C), got shape={sequence.shape}")
    if sequence.shape[1] < 13:
        raise ValueError(f"Expected at least 13 keypoints, got shape={sequence.shape}")
    if sequence.shape[2] < 3:
        raise ValueError(f"Expected at least 3 channels [x, y, visibility], got shape={sequence.shape}")

    out = sequence.astype(np.float32, copy=True)
    scale, valid = compute_scale_per_frame(out, eps=eps)
    safe_scale = np.maximum(scale, eps)

    if np.any(valid):
        out[valid, :, :2] /= safe_scale[valid, None, None]

    return out


def target_path_for(source_path: Path, base_folder: Path, output_folder: Path | None, inplace: bool) -> Path:
    if inplace:
        return source_path

    if output_folder is not None:
        rel = source_path.relative_to(base_folder)
        return output_folder / rel

    return source_path.with_name(f"{source_path.stem}_unit{source_path.suffix}")


def process_folder(folder: Path, output_folder: Path | None, inplace: bool, eps: float) -> None:
    npy_files = sorted(folder.rglob("*.npy"))
    if not npy_files:
        print(f"No .npy files found under: {folder}")
        return

    processed = 0
    skipped = 0

    for npy_path in npy_files:
        try:
            data = np.load(npy_path)
            scaled = unit_scale_sequence(data, eps=eps)

            target = target_path_for(npy_path, folder, output_folder, inplace)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, scaled)
            processed += 1
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"[WARN] Skipped {npy_path}: {exc}")

    print(f"Processed files: {processed}")
    print(f"Skipped files: {skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply unit scaling to pose .npy files by dividing x/y coordinates "
            "with neck-to-mid-hip distance per frame."
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
            "files are saved next to originals with suffix _unit.npy."
        ),
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input files in place.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Small positive threshold for valid scale.",
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
        eps=args.eps,
    )


if __name__ == "__main__":
    main()
