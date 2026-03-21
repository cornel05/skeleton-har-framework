import argparse
from pathlib import Path
import numpy as np
from typing import Tuple, Optional


def compute_scale_per_frame(sequence: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute neck-to-mid-hip distance per frame for COCO17 keypoints.

    Neck is approximated as midpoint of left/right shoulders (5, 6).
    Mid-hip is midpoint of left/right hips (11, 12).

    Returns:
    - scale: shape (T,), Euclidean distance in x/y for each frame.
    - valid: shape (T,), whether scale is usable for division.
    """
    if sequence.ndim == 2 and sequence.shape[1] == 34:
        sequence = sequence.reshape(sequence.shape[0], 17, 2)
    elif sequence.ndim != 3:
         raise ValueError(f"Expected (T, K, C) or (T, 34), got {sequence.shape}")

    coords_xy = sequence[:, :, :2]

    neck_xy = (coords_xy[:, 5, :] + coords_xy[:, 6, :]) / 2.0
    mid_hip_xy = (coords_xy[:, 11, :] + coords_xy[:, 12, :]) / 2.0

    scale = np.linalg.norm(neck_xy - mid_hip_xy, axis=1)
    valid = scale > eps

    if sequence.shape[2] >= 3:
        vis = sequence[:, :, 2]
        valid &= vis[:, 5] > 0.0
        valid &= vis[:, 6] > 0.0
        valid &= vis[:, 11] > 0.0
        valid &= vis[:, 12] > 0.0

    return scale, valid


def apply_unit_scale(coco: np.ndarray, eps: float) -> None:
    """Scale x/y by neck-to-mid-hip distance with epsilon-safe denominator."""
    neck = (coco[5, :2] + coco[6, :2]) / 2.0
    mid_hip = (coco[11, :2] + coco[12, :2]) / 2.0
    scale = float(np.linalg.norm(neck - mid_hip))
    safe_scale = max(scale, eps)
    coco[:, :2] /= safe_scale


def unit_scale_sequence(sequence: np.ndarray, eps: float) -> np.ndarray:
    """Apply per-frame unit scaling on x/y coordinates."""
    if sequence.ndim != 3:
        raise ValueError(f"Expected 3D array (T, K, C), got shape={sequence.shape}")
    if sequence.shape[1] < 13:
        raise ValueError(f"Expected at least 13 keypoints, got shape={sequence.shape}")
    if sequence.shape[2] < 2:
        raise ValueError(f"Expected at least 2 channels [x, y], got shape={sequence.shape}")

    out = sequence.astype(np.float32, copy=True)
    scale, valid = compute_scale_per_frame(out, eps=eps)
    safe_scale = np.maximum(scale, eps)

    if np.any(valid):
        out[valid, :, :2] /= safe_scale[valid, None, None]

    return out


def mirror_coco17_sequence(sequence: np.ndarray) -> np.ndarray:
    """Mirror centered COCO17 sequence horizontally and swap left/right joints."""
    original_shape = sequence.shape
    if sequence.ndim == 2 and sequence.shape[1] == 34:
        mirrored = sequence.reshape(sequence.shape[0], 17, 2).astype(np.float32, copy=True)
    elif sequence.ndim == 3 and sequence.shape[1] == 17:
        mirrored = sequence.astype(np.float32, copy=True)
    else:
        raise ValueError(f"Unexpected shape for mirroring: {sequence.shape}")
    
    mirrored[:, :, 0] = -mirrored[:, :, 0]

    left_right_pairs = (
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    )
    for left_idx, right_idx in left_right_pairs:
        tmp = mirrored[:, left_idx, :].copy()
        mirrored[:, left_idx, :] = mirrored[:, right_idx, :]
        mirrored[:, right_idx, :] = tmp

    if len(original_shape) == 2:
        return mirrored.reshape(original_shape[0], 34)
    return mirrored


def target_path_for(source_path: Path, base_folder: Path, output_folder: Path | None, inplace: bool) -> Path:
    if inplace:
        return source_path
    if output_folder is not None:
        rel = source_path.relative_to(base_folder)
        return output_folder / rel
    return source_path.with_name(f"{source_path.stem}_unit{source_path.suffix}")


def process_folder(folder: Path, output_folder: Path | None, inplace: bool, eps: float) -> None:
    npy_files = sorted(folder.rglob("*.npy"))
    processed = 0
    for npy_path in npy_files:
        try:
            data = np.load(npy_path)
            scaled = unit_scale_sequence(data, eps=eps)
            target = target_path_for(npy_path, folder, output_folder, inplace)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, scaled)
            processed += 1
        except Exception as exc:
            print(f"[WARN] Skipped {npy_path}: {exc}")
    print(f"Processed: {processed}")


def main():
    parser = argparse.ArgumentParser(description="Unit scale pose .npy files.")
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, default=None)
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    process_folder(
        folder=args.folder.resolve(),
        output_folder=args.output_folder.resolve() if args.output_folder else None,
        inplace=args.inplace,
        eps=args.eps,
    )


if __name__ == "__main__":
    main()
