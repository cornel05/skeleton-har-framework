import argparse
import csv
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from dataclasses import dataclass

try:
    from ..utils import resolve_device
    from .common import apply_unit_scale, mirror_coco17_sequence
except ImportError:
    # Allow running this file directly: python src/pose_estimation/preprocessing/le2i.py
    import sys

    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from pose_estimation.utils import resolve_device
    from pose_estimation.preprocessing.common import apply_unit_scale, mirror_coco17_sequence


@dataclass
class VideoItem:
    folder_name: str
    folder_path: Path
    video_path: Path
    label: int


def infer_label(folder_name: str) -> Optional[int]:
    lower = folder_name.lower()
    if lower.startswith("fall-"):
        return 1
    if lower.startswith("adl-"):
        return 0
    return None


def find_video_file(folder_path: Path) -> Optional[Path]:
    mp4_files = sorted(folder_path.rglob("*.mp4"))
    if not mp4_files:
        return None
    return mp4_files[0]


def discover_videos(dataset_root: Path) -> List[VideoItem]:
    items: List[VideoItem] = []

    for folder in sorted(dataset_root.iterdir()):
        if not folder.is_dir():
            continue

        label = infer_label(folder.name)
        if label is None:
            continue

        video_path = find_video_file(folder)
        if video_path is None:
            continue

        items.append(
            VideoItem(
                folder_name=folder.name,
                folder_path=folder,
                video_path=video_path,
                label=label,
            )
        )

    return items


def pick_pose_from_result(result) -> Optional[np.ndarray]:
    """
    Select one person's keypoints from a YOLOv8-pose result.

    Returns shape (17, 3): [x_px, y_px, keypoint_conf].
    """
    if result.keypoints is None or result.boxes is None:
        return None

    xy = result.keypoints.xy
    if xy is None or len(xy) == 0:
        return None

    xy_np = xy.detach().cpu().numpy()  # (N, 17, 2)
    conf_np = None
    if result.keypoints.conf is not None:
        conf_np = result.keypoints.conf.detach().cpu().numpy()  # (N, 17)

    boxes_conf = (
        result.boxes.conf.detach().cpu().numpy()
        if result.boxes.conf is not None
        else np.zeros((xy_np.shape[0],), dtype=np.float32)
    )

    n_person = xy_np.shape[0]
    if n_person == 0:
        return None

    # No external annotations in UR-Fall. Keep the highest confidence person.
    best_idx = int(np.argmax(boxes_conf))

    selected_xy = xy_np[best_idx]  # (17, 2)
    selected_conf = conf_np[best_idx] if conf_np is not None else np.ones((17,), dtype=np.float32)
    return np.concatenate([selected_xy, selected_conf[:, None]], axis=1).astype(np.float32)


def extract_skeleton_sequence(
    video_path: Path,
    model: YOLO,
    device: str,
    imgsz: int,
    conf_thres: float,
    scale_eps: float,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames <= 0:
        total_frames = 1

    sequence = np.zeros((total_frames, 34), dtype=np.float32)

    frame_idx = 0
    frame_bar = tqdm(
        total=total_frames,
        desc=f"Frames: {video_path.stem}",
        leave=False,
        unit="frame",
    )

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx >= sequence.shape[0]:
                extra = np.zeros((1, 34), dtype=np.float32)
                sequence = np.concatenate([sequence, extra], axis=0)

            results = model.predict(
                source=frame_bgr,
                device=device,
                imgsz=imgsz,
                conf=conf_thres,
                verbose=False,
            )

            if results:
                selected = pick_pose_from_result(results[0])
                if selected is not None:
                    # selected: (17, 3) => [x_px, y_px, keypoint_conf]
                    coco = np.zeros((17, 3), dtype=np.float32)
                    coco[:, 0] = np.clip(selected[:, 0] / max(width, 1), 0.0, 1.0)
                    coco[:, 1] = np.clip(selected[:, 1] / max(height, 1), 0.0, 1.0)
                    coco[:, 2] = np.clip(selected[:, 2], 0.0, 1.0)

                    # Center skeleton around the mid-hip point in x, y.
                    left_hip = coco[11, :2]
                    right_hip = coco[12, :2]
                    mid_hip = (left_hip + right_hip) / 2.0
                    coco[:, :2] -= mid_hip
                    apply_unit_scale(coco, eps=scale_eps)

                    # Flatten to (34,): concatenate all x,y coordinates
                    sequence[frame_idx] = coco[:, :2].flatten()

            frame_idx += 1
            frame_bar.update(1)
    finally:
        frame_bar.close()
        cap.release()

    if frame_idx < sequence.shape[0]:
        sequence = sequence[:frame_idx]

    return sequence


def process_dataset(
    dataset_root: Path,
    output_dir: Path,
    model_path: str,
    device: str,
    imgsz: int,
    conf_thres: float,
    scale_eps: float,
    mirror_aug: bool,
) -> None:
    items = discover_videos(dataset_root)
    if not items:
        print(f"No valid videos found under {dataset_root}")
        return

    print(f"[INFO] Loading YOLO model: {model_path}")
    print(f"[INFO] Inference device: {device}")
    model = YOLO(model_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "summary_labels.csv"

    rows: List[List[str]] = []

    video_bar = tqdm(items, desc="Videos", unit="video")
    for item in video_bar:
        video_bar.set_postfix_str(item.folder_name)
        output_path = output_dir / f"{item.folder_name}_skeleton.npy"

        try:
            skeleton = extract_skeleton_sequence(
                item.video_path,
                model=model,
                device=device,
                imgsz=imgsz,
                conf_thres=conf_thres,
                scale_eps=scale_eps,
            )
            np.save(output_path, skeleton)

            rows.append(
                [
                    item.folder_name,
                    str(item.video_path.relative_to(dataset_root)),
                    str(output_path.relative_to(output_dir)),
                    str(item.label),
                    str(skeleton.shape[0]),
                ]
            )
        except Exception as exc:
            print(f"[ERROR] Failed processing {item.video_path}: {exc}")
            continue

        if mirror_aug:
            mirror_output_path = output_dir / f"{item.folder_name}_skeleton_mirror.npy"
            try:
                mirrored = mirror_coco17_sequence(skeleton)
                np.save(mirror_output_path, mirrored)
                rows.append(
                    [
                        f"{item.folder_name}_mirror",
                        str(item.video_path.relative_to(dataset_root)),
                        str(mirror_output_path.relative_to(output_dir)),
                        str(item.label),
                        str(mirrored.shape[0]),
                    ]
                )
            except Exception as exc:
                print(f"[WARN] Failed mirror augmentation for {item.video_path}: {exc}")

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["folder_name", "video_relpath", "skeleton_file", "label", "num_frames"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} skeleton file(s) to {output_dir}")
    print(f"Saved labels summary to {summary_csv}")


def main():
    parser = argparse.ArgumentParser(description="UR-Fall dataset processing.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="models/yolov8n-pose.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--scale-eps", type=float, default=1e-6)
    parser.add_argument("--mirror-aug", action="store_true")
    args = parser.parse_args()

    device = str(resolve_device(args.device))
    process_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        model_path=args.model,
        device=device,
        imgsz=args.imgsz,
        conf_thres=args.conf,
        scale_eps=args.scale_eps,
        mirror_aug=args.mirror_aug,
    )


if __name__ == "__main__":
    main()
