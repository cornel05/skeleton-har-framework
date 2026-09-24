"""
Per-frame feature construction shared by training, evaluation and live inference.

Learned models get the repo's original 34-d representation (COCO-17 x/y, normalized
by image size, centered on the mid-hip, divided by the neck-to-mid-hip distance).
Two repairs relative to the original extraction scripts:
  * frames where YOLO found no person are linearly interpolated from neighbouring
    frames instead of being written as all-zero vectors;
  * values are clipped to +-FEATURE_CLIP so a foreshortened torso (tiny scale) cannot
    produce unbounded inputs.
The rule-based baseline uses the raw pixel trajectories (see `rule_signals`).
"""

from typing import Dict

import numpy as np

FEATURE_CLIP = 10.0
SCALE_EPS = 1e-6
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


def interpolate_missing(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN rows along axis 0 (edges take the nearest valid value)."""
    out = np.array(arr, dtype=np.float32, copy=True)
    flat = out.reshape(out.shape[0], -1)
    t = np.arange(flat.shape[0])
    for j in range(flat.shape[1]):
        col = flat[:, j]
        ok = ~np.isnan(col)
        if ok.all() or not ok.any():
            continue
        col[~ok] = np.interp(t[~ok], t[ok], col[ok])
    return out


def normalize_repo_features(kpts: np.ndarray, width: int, height: int) -> np.ndarray:
    """(T, 17, >=2) pixel keypoints (NaN = no detection) -> (T, 34) normalized features."""
    if kpts.shape[0] == 0:
        return np.zeros((0, 34), np.float32)
    xy = interpolate_missing(kpts[:, :, :2])
    xy = np.nan_to_num(xy, nan=0.0)  # only when no frame of the video had a detection
    xy[:, :, 0] = np.clip(xy[:, :, 0] / max(width, 1), 0.0, 1.0)
    xy[:, :, 1] = np.clip(xy[:, :, 1] / max(height, 1), 0.0, 1.0)
    mid_hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2.0
    xy -= mid_hip[:, None, :]
    neck = (xy[:, L_SHOULDER] + xy[:, R_SHOULDER]) / 2.0
    scale = np.linalg.norm(neck, axis=1)  # mid-hip is the origin now
    xy /= np.maximum(scale, SCALE_EPS)[:, None, None]
    return np.clip(xy.reshape(xy.shape[0], 34), -FEATURE_CLIP, FEATURE_CLIP).astype(np.float32)


def rule_signals(kpts: np.ndarray, box: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Raw per-frame signals for the heuristic baseline (pixel units, interpolated):
      hip_y  : mid-hip vertical position (image y grows downwards)
      box_h  : person box height
      aspect : person box width / height
    """
    xy = interpolate_missing(kpts[:, :, :2])
    b = interpolate_missing(box[:, :4])
    hip_y = (xy[:, L_HIP, 1] + xy[:, R_HIP, 1]) / 2.0
    w = np.maximum(b[:, 2] - b[:, 0], 1.0)
    h = np.maximum(b[:, 3] - b[:, 1], 1.0)
    return {
        "hip_y": np.nan_to_num(hip_y).astype(np.float32),
        "box_h": np.nan_to_num(h, nan=1.0).astype(np.float32),
        "aspect": np.nan_to_num(w / h, nan=1.0).astype(np.float32),
    }
