"""
Synthetic stand-in data with the exact on-disk format of the real pipeline (pose .npz +
official-style per-frame label CSVs). Used ONLY to smoke-test the code path end to end
(`make smoke`); nothing produced from it is a result.
"""

import argparse
from pathlib import Path

import numpy as np

# Standing COCO-17 template (pixels, feet at y=0, head up = negative y), ~250 px tall.
TEMPLATE = np.array([
    [0, -235], [-6, -242], [6, -242], [-14, -238], [14, -238],   # nose, eyes, ears
    [-30, -200], [30, -200], [-38, -150], [38, -150], [-40, -105], [40, -105],  # shoulders, elbows, wrists
    [-18, -120], [18, -120], [-20, -62], [20, -62], [-20, 0], [20, 0],  # hips, knees, ankles
], np.float32)


def _rotate(pts, angle, pivot):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]], np.float32)
    return (pts - pivot) @ R.T + pivot


def _video(kind: str, rng: np.random.Generator, W=640, H=480):
    T = int(rng.integers(90, 200))
    base_x, floor_y = float(rng.uniform(200, 440)), float(rng.uniform(380, 450))
    frames, labels = [], np.full(T, -1, np.int64)
    t0 = int(rng.integers(30, T - 50))
    dur = int(rng.integers(10, 20))
    direction = rng.choice([-1.0, 1.0])
    for t in range(T):
        pts = TEMPLATE.copy()
        x = base_x + (t * rng.uniform(0.5, 1.5) if kind == "walk" else 0.0)
        pts[:, 0] += 8 * np.sin(t / 5.0) * (np.arange(17) >= 13) * (kind == "walk")
        if kind == "fall" and t >= t0:
            a = min(1.0, (t - t0) / dur) * np.pi / 2 * direction
            pts = _rotate(pts, a, np.array([0.0, 0.0], np.float32))
            labels[t] = 0 if t < t0 + dur else 1
        elif kind == "sit" and t >= t0:
            k = min(1.0, (t - t0) / 40.0)
            pts[:13, 1] += 60 * k
        elif kind == "bend" and t0 <= t < t0 + 40:
            a = np.sin(np.pi * (t - t0) / 40.0) * np.pi / 3
            pts[:11] = _rotate(pts[:11], a, pts[11:13].mean(axis=0))
        pts = pts + np.array([x, floor_y], np.float32) + rng.normal(0, 2.0, pts.shape).astype(np.float32)
        frames.append(pts)
    xy = np.stack(frames)
    kpts = np.concatenate([xy, np.full((T, 17, 1), 0.9, np.float32)], axis=2)
    lo, hi = xy.min(axis=1) - 10, xy.max(axis=1) + 10
    box = np.concatenate([lo, hi, np.full((T, 1), 0.9, np.float32)], axis=1)
    miss = rng.random(T) < 0.03
    kpts[miss] = np.nan
    box[miss] = np.nan
    return kpts.astype(np.float32), box.astype(np.float32), labels, W, H


def generate(root: Path, n_falls: int = 30, n_adls: int = 40, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    pose, raw = root / "pose", root / "raw"
    pose.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    rows = {"falls": [], "adls": []}
    for kind_set, n, prefix in (("falls", n_falls, "fall"), ("adls", n_adls, "adl")):
        for i in range(1, n + 1):
            kind = "fall" if prefix == "fall" else ["walk", "sit", "bend"][i % 3]
            kpts, box, labels, W, H = _video(kind, rng)
            seq = f"{prefix}-{i:02d}"
            np.savez_compressed(pose / f"{seq}-cam0-rgb.npz", kpts=kpts, box=box,
                                frame_id=np.arange(1, len(labels) + 1, dtype=np.int32), width=np.int32(W),
                                height=np.int32(H), fps=np.float32(30.0), sequence=np.array(seq),
                                pose_model=np.array("synthetic"), imgsz=np.int32(0), conf_thres=np.float32(0))
            rows[kind_set] += [f"{seq},{t + 1},{lab},0,0,0,0,0,0,0,0" for t, lab in enumerate(labels)]
    for kind_set, lines in rows.items():
        (raw / f"urfall-cam0-{kind_set}.csv").write_text("\n".join(lines) + "\n")
    print(f"Synthetic data ({n_falls} falls, {n_adls} ADLs) -> {root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("dataset/synthetic"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate(a.root, seed=a.seed)
