"""
Generic YOLOv8-pose keypoint extraction for arbitrary videos (replaces the former MediaPipe
extractor, whose 33-landmark / 66-d output did not match the 34-d COCO-17 training features).

Writes, per video, `<stem>_keypoints.npz` (raw pixel keypoints + confidences + box, NaN where no
person was found) and `<stem>_keypoints.npy` (the 34-d normalized features the classifiers consume).
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from ..features import normalize_repo_features
    from .urfall import extract_sequence
except ImportError:
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from pose_estimation.features import normalize_repo_features
    from pose_estimation.preprocessing.urfall import extract_sequence

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def _frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        yield idx, frame
    cap.release()


def extract_keypoints(video_path: Path, model, imgsz: int = 320, conf: float = 0.25, device: str = "cpu") -> dict:
    rec = extract_sequence(_frames(video_path), model, device, imgsz, conf)
    cap = cv2.VideoCapture(str(video_path))
    rec["fps"] = np.float32(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    logging.info("%s | frames: %d | frames with a person: %d", video_path, len(rec["frame_id"]),
                 int((~np.isnan(rec["kpts"][:, 0, 0])).sum()) if len(rec["kpts"]) else 0)
    return rec


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="YOLOv8-pose keypoint extraction (COCO-17, 2D).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path)
    group.add_argument("--folder", type=Path, help="Process every video under this folder (recursive).")
    parser.add_argument("--output", type=Path, default=None, help="Output folder (default: next to each video).")
    parser.add_argument("--model", type=str, default="models/yolov8n-pose.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    videos = [args.video] if args.video else sorted(
        p for p in args.folder.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    for video in videos:
        rec = extract_keypoints(video, model, args.imgsz, args.conf)
        out_dir = args.output or video.parent
        if args.output and args.folder:
            out_dir = args.output / video.parent.relative_to(args.folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"{video.stem}_keypoints.npz", **rec)
        np.save(out_dir / f"{video.stem}_keypoints.npy",
                normalize_repo_features(rec["kpts"], int(rec["width"]), int(rec["height"])))
        logging.info("Saved %s", out_dir / f"{video.stem}_keypoints.np[yz]")


if __name__ == "__main__":
    main()
