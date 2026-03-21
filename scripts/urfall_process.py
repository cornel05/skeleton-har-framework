import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

try:
	import torch
except Exception:  # noqa: BLE001
	torch = None


def resolve_device(device_arg: str) -> str:
	if device_arg != "auto":
		return device_arg
	if torch is not None and torch.cuda.is_available():
		return "cuda:0"
	return "cpu"


def apply_unit_scale(coco: np.ndarray, eps: float) -> None:
	"""Scale x/y by neck-to-mid-hip distance with epsilon-safe denominator."""
	neck = (coco[5, :2] + coco[6, :2]) / 2.0
	mid_hip = (coco[11, :2] + coco[12, :2]) / 2.0
	scale = float(np.linalg.norm(neck - mid_hip))
	safe_scale = max(scale, eps)
	coco[:, :2] /= safe_scale


def mirror_coco17_sequence(sequence: np.ndarray) -> np.ndarray:
	"""Mirror centered COCO17 sequence horizontally and swap left/right joints."""
	if sequence.ndim != 2 or sequence.shape[1] != 34:
		raise ValueError(f"Expected 2D array (T, 34), got {sequence.shape}")

	# Reshape to (T, 17, 2) for processing
	mirrored = sequence.reshape(sequence.shape[0], 17, 2).astype(np.float32, copy=True)
	
	# Sequence is centered around mid-hip, so horizontal mirror is x -> -x.
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

	# Flatten back to (T, 34)
	return mirrored.reshape(mirrored.shape[0], 34)


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
			print(f"[WARN] No .mp4 found in {folder}")
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

	resolved_device = resolve_device(device)
	print(f"[INFO] Loading YOLO model: {model_path}")
	print(f"[INFO] Inference device: {resolved_device}")
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
				device=resolved_device,
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
		except Exception as exc:  # noqa: BLE001
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
			except Exception as exc:  # noqa: BLE001
				print(f"[WARN] Failed mirror augmentation for {item.video_path}: {exc}")

	with summary_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["folder_name", "video_relpath", "skeleton_file", "label", "num_frames"])
		writer.writerows(rows)

	print(f"Saved {len(rows)} skeleton file(s) to {output_dir}")
	print(f"Saved labels summary to {summary_csv}")


def parse_args() -> argparse.Namespace:
	project_root = Path(__file__).resolve().parent.parent
	default_dataset_root = project_root / "dataset" / "ur-fall-detection-dataset" / "UR_fall_detection_dataset_cam0_rgb"
	default_output_dir = project_root / "processed_data"

	parser = argparse.ArgumentParser(
		description="Preprocess UR Fall Detection videos into COCO17 skeleton sequences using YOLOv8-pose."
	)
	parser.add_argument(
		"--dataset-root",
		type=Path,
		default=default_dataset_root,
		help="Path to UR_fall_detection_dataset_cam0_rgb root.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=default_output_dir,
		help="Directory to save skeleton .npy files and summary_labels.csv.",
	)
	parser.add_argument(
		"--model",
		type=str,
		default="models/yolov8n-pose.pt",
		help="YOLOv8-pose model path or name (e.g. yolov8n-pose.pt).",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="auto",
		help='Inference device: "auto", "cuda:0", or "cpu".',
	)
	parser.add_argument(
		"--imgsz",
		type=int,
		default=320,
		help="YOLO inference image size.",
	)
	parser.add_argument(
		"--conf",
		type=float,
		default=0.25,
		help="YOLO confidence threshold.",
	)
	parser.add_argument(
		"--scale-eps",
		type=float,
		default=1e-6,
		help="Epsilon floor for neck-to-mid-hip scale denominator.",
	)
	parser.add_argument(
		"--mirror-aug",
		action="store_true",
		help="Also save mirrored skeleton sequence for each video.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	dataset_root = args.dataset_root.resolve()
	output_dir = args.output_dir.resolve()

	if not dataset_root.exists() or not dataset_root.is_dir():
		raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {dataset_root}")

	process_dataset(
		dataset_root=dataset_root,
		output_dir=output_dir,
		model_path=args.model,
		device=args.device,
		imgsz=args.imgsz,
		conf_thres=args.conf,
		scale_eps=args.scale_eps,
		mirror_aug=args.mirror_aug,
	)


if __name__ == "__main__":
	main()
