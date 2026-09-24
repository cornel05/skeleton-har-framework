"""
Video -> fixed-length windows with per-window labels derived from per-frame annotations.

Frame labels (official UR-Fall CSV, cam0): -1 not lying, 0 falling, 1 lying on the ground.
Window labels:
  ADL video  -> every window negative (0).
  Fall video -> positive (1) if the window holds >= min(min_fall_frames, |fall interval|)
                frames of the annotated falling interval; negative (0) if it holds no
                falling and no lying frame (pre-fall activity); otherwise ignored (-1):
                partial overlap with the fall, or post-fall lying only.
Ignored windows are excluded from training and from window-level metrics, but every
window of a test video is scored for the event-level metrics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..features import normalize_repo_features, rule_signals

NOT_LYING, FALLING, LYING, UNKNOWN = -1, 0, 1, -9
POS, NEG, IGNORE = 1, 0, -1
ONSET_HALF_WIDTH = 4  # used only when a fall video has lying frames but no 'falling' frames
VEL_LAG = 5           # frames used for the hip-velocity finite difference
EDGE = 8              # frames averaged at each window end for the aspect-ratio change


@dataclass
class Video:
    vid: str
    group: str
    label: int                       # 1 = fall video, 0 = ADL video
    feats: np.ndarray                # (T, 34)
    signals: Dict[str, np.ndarray]   # raw per-frame signals for the rule baseline
    frame_labels: np.ndarray         # (T,) official per-frame labels, UNKNOWN if absent
    fps: float
    missing_rate: float              # fraction of frames without a detected person
    fall_interval: Optional[np.ndarray] = None  # bool (T,) mask of the annotated falling interval

    def __post_init__(self):
        if self.label == 1 and self.fall_interval is None:
            self.fall_interval = fall_interval_mask(self.frame_labels)


def fall_interval_mask(frame_labels: np.ndarray) -> np.ndarray:
    mask = frame_labels == FALLING
    if mask.any():
        return mask
    lying = np.flatnonzero(frame_labels == LYING)
    mask = np.zeros_like(mask)
    if lying.size:
        t = int(lying[0])
        mask[max(0, t - ONSET_HALF_WIDTH):t + ONSET_HALF_WIDTH] = True
    return mask


def label_window(video: Video, start: int, length: int, min_fall_frames: int) -> int:
    if video.label == 0:
        return NEG
    fi = video.fall_interval
    if not fi.any():
        return IGNORE
    sl = slice(start, start + length)
    f = int(fi[sl].sum())
    lying = int(((video.frame_labels[sl] == LYING) & ~fi[sl]).sum())
    if f >= min(min_fall_frames, int(fi.sum())):
        return POS
    if f == 0 and lying == 0:
        return NEG
    return IGNORE


def rule_window_features(sig: Dict[str, np.ndarray], start: int, length: int, fps: float) -> np.ndarray:
    """[peak downward hip speed in body-heights/s, aspect-ratio change end-vs-start]."""
    sl = slice(start, start + length)
    hip, h, ar = sig["hip_y"][sl], sig["box_h"][sl], sig["aspect"][sl]
    edge = max(1, min(EDGE, len(ar) // 2))
    h_ref = max(float(np.median(h[:edge])), 1.0)
    lag = min(VEL_LAG, max(1, len(hip) - 1))
    v = (hip[lag:] - hip[:-lag]) / h_ref * (fps / lag) if len(hip) > lag else np.zeros(1)
    d_ar = float(ar[-edge:].mean() - ar[:edge].mean())
    return np.array([float(v.max()), d_ar], np.float32)


@dataclass
class WindowSet:
    X: np.ndarray        # (N, L, 34)
    mask: np.ndarray     # (N, L)
    y: np.ndarray        # (N,) POS / NEG / IGNORE
    vid: np.ndarray      # (N,) index into `videos`
    start: np.ndarray    # (N,) first frame of the window
    rule: np.ndarray     # (N, 2) heuristic features
    videos: List[str] = field(default_factory=list)
    stride: int = 0
    length: int = 0

    def select(self, vids: List[str], labeled_only: bool) -> np.ndarray:
        idx = np.flatnonzero(np.isin(self.vid, [self.videos.index(v) for v in vids]))
        if labeled_only:
            idx = idx[self.y[idx] != IGNORE]
        return idx


def build_windows(videos: List[Video], length: int, stride: int, min_fall_frames: int) -> WindowSet:
    X, M, Y, V, S, R = [], [], [], [], [], []
    for vi, v in enumerate(videos):
        T = v.feats.shape[0]
        starts = list(range(0, max(T - length, 0) + 1, stride))
        for s in starts:
            w = v.feats[s:s + length]
            m = np.ones(length, np.float32)
            if w.shape[0] < length:  # short video: zero-pad, mask marks valid frames
                m[w.shape[0]:] = 0.0
                w = np.concatenate([w, np.zeros((length - w.shape[0], w.shape[1]), np.float32)])
            X.append(w)
            M.append(m)
            Y.append(label_window(v, s, length, min_fall_frames))
            V.append(vi)
            S.append(s)
            R.append(rule_window_features(v.signals, s, length, v.fps))
    return WindowSet(
        X=np.stack(X).astype(np.float32), mask=np.stack(M), y=np.asarray(Y, np.int64),
        vid=np.asarray(V, np.int64), start=np.asarray(S, np.int64), rule=np.stack(R),
        videos=[v.vid for v in videos], stride=stride, length=length,
    )


def load_urfall_videos(pose_dir: Path, frame_labels: Dict[str, Dict[int, int]]) -> List[Video]:
    videos = []
    for path in sorted(Path(pose_dir).glob("*.npz")):
        rec = np.load(path, allow_pickle=False)
        stem = path.stem                            # fall-01-cam0-rgb
        seq = str(rec["sequence"])                  # fall-01
        kpts, box = rec["kpts"], rec["box"]
        if kpts.shape[0] == 0:
            continue
        per_frame = frame_labels.get(seq, {})
        fl = np.array([per_frame.get(int(f), UNKNOWN) for f in rec["frame_id"]], np.int64)
        videos.append(Video(
            vid=stem, group=seq, label=1 if seq.startswith("fall") else 0,
            feats=normalize_repo_features(kpts, int(rec["width"]), int(rec["height"])),
            signals=rule_signals(kpts, box), frame_labels=fl, fps=float(rec["fps"]),
            missing_rate=float(np.isnan(kpts[:, 0, 0]).mean()),
        ))
    return videos


@dataclass
class Prepared:
    """Everything the protocol needs, cached once by `run prepare` (pickled)."""
    train_ws: WindowSet
    eval_ws: WindowSet
    meta: Dict[str, dict]
    splits: Dict[str, List[Dict[str, List[str]]]]
    protocol: dict
