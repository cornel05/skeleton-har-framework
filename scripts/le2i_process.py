import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm  # type: ignore[reportMissingModuleSource]
from ultralytics import YOLO

try:
	import torch
except Exception:  # noqa: BLE001
	torch = None


@dataclass
class AnnotationData:
	fall_start: Optional[int]
	fall_end: Optional[int]
	bboxes: Dict[int, Tuple[int, int, int, int]]


def _to_int(value: str) -> Optional[int]:
	try:
		return int(float(value.strip()))
	except (ValueError, TypeError):
		return None


def parse_annotation_file(annotation_path: Path) -> AnnotationData:
	"""
	Parse LE2I annotation text file.

	Expected format:
	- Line 1: fall start frame
	- Line 2: fall end frame
	- Line 3+: frame_num, state, xmin, ymin, xmax, ymax
	"""
	lines = [line.strip() for line in annotation_path.read_text(encoding="utf-8").splitlines() if line.strip()]

	fall_start = _to_int(lines[0]) if len(lines) >= 1 else None
	fall_end = _to_int(lines[1]) if len(lines) >= 2 else None

	bboxes: Dict[int, Tuple[int, int, int, int]] = {}
	for raw in lines[2:]:
		# Support both comma-separated and whitespace-separated files.
		parts = [p.strip() for p in raw.replace("\t", " ").split(",")]
		if len(parts) == 1:
			parts = [p for p in raw.split() if p]
		if len(parts) < 6:
			continue

		frame_num = _to_int(parts[0])
		xmin = _to_int(parts[2])
		ymin = _to_int(parts[3])
		xmax = _to_int(parts[4])
		ymax = _to_int(parts[5])

		if None in (frame_num, xmin, ymin, xmax, ymax):
			continue

		bboxes[frame_num] = (xmin, ymin, xmax, ymax)

	return AnnotationData(fall_start=fall_start, fall_end=fall_end, bboxes=bboxes)


def find_annotation_dir(videos_dir: Path) -> Optional[Path]:
	parent = videos_dir.parent
	for name in ("Annotation_files", "Annotations_files"):
		candidate = parent / name
		if candidate.is_dir():
			return candidate
	return None


def resolve_annotation_file(video_path: Path, annotation_dir: Path) -> Optional[Path]:
	exact = annotation_dir / f"{video_path.stem}.txt"
	if exact.exists():
		return exact

	# Fallback for naming inconsistencies.
	matching = sorted(annotation_dir.glob(f"*{video_path.stem}*.txt"))
	if matching:
		return matching[0]

	txt_files = sorted(annotation_dir.glob("*.txt"))
	if len(txt_files) == 1:
		return txt_files[0]

	return None


def discover_videos(dataset_root: Path) -> List[Tuple[Path, Optional[Path]]]:
	"""Return list of (video_path, annotation_path_or_none)."""
	items: List[Tuple[Path, Optional[Path]]] = []

	for videos_dir in dataset_root.rglob("Videos"):
		if not videos_dir.is_dir():
			continue

		annotation_dir = find_annotation_dir(videos_dir)
		avi_files = sorted(videos_dir.glob("*.avi"))

		for video_path in avi_files:
			annotation_path = None
			if annotation_dir is not None:
				annotation_path = resolve_annotation_file(video_path, annotation_dir)
			items.append((video_path, annotation_path))

	return items


def _bbox_is_zero(bbox: Tuple[int, int, int, int]) -> bool:
	return bbox[0] == 0 and bbox[1] == 0 and bbox[2] == 0 and bbox[3] == 0


def _bbox_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
	ax1, ay1, ax2, ay2 = box_a
	bx1, by1, bx2, by2 = box_b

	inter_x1 = max(ax1, bx1)
	inter_y1 = max(ay1, by1)
	inter_x2 = min(ax2, bx2)
	inter_y2 = min(ay2, by2)

	inter_w = max(0.0, inter_x2 - inter_x1)
	inter_h = max(0.0, inter_y2 - inter_y1)
	inter_area = inter_w * inter_h

	area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
	area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
	union = area_a + area_b - inter_area
	if union <= 0.0:
		return 0.0
	return inter_area / union


def _frame_bbox_for_index(annotation: AnnotationData, frame_idx_zero_based: int) -> Optional[Tuple[int, int, int, int]]:
	# LE2I annotations are often 1-based; keep a fallback for 0-based files.
	one_based = frame_idx_zero_based + 1
	return annotation.bboxes.get(one_based, annotation.bboxes.get(frame_idx_zero_based))


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


def pick_pose_from_result(
	result,
	annotation_bbox: Optional[Tuple[int, int, int, int]],
) -> Optional[np.ndarray]:
	"""
	Select one person's keypoints from a YOLOv8-pose result.

	Returns shape (17, 3): [x_px, y_px, visibility_conf].
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

	boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()  # (N, 4)
	boxes_conf = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.zeros((xy_np.shape[0],))

	n_person = xy_np.shape[0]
	if n_person == 0:
		return None

	if annotation_bbox is not None and not _bbox_is_zero(annotation_bbox):
		ann = tuple(float(v) for v in annotation_bbox)
		candidates: List[Tuple[int, float, float]] = []
		for i in range(n_person):
			iou = _bbox_iou(tuple(float(v) for v in boxes_xyxy[i]), ann)
			if iou > 0.0:
				candidates.append((i, iou, float(boxes_conf[i])))

		if not candidates:
			return None

		# Prioritize overlap with annotation, then detector confidence.
		candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
		best_idx = candidates[0][0]
	else:
		best_idx = int(np.argmax(boxes_conf))

	selected_xy = xy_np[best_idx]  # (17, 2)
	selected_conf = conf_np[best_idx] if conf_np is not None else np.ones((17,), dtype=np.float32)
	return np.concatenate([selected_xy, selected_conf[:, None]], axis=1).astype(np.float32)


def extract_video_pose(
	video_path: Path,
	annotation: Optional[AnnotationData],
	output_path: Path,
	model: YOLO,
	device: str,
	imgsz: int,
	conf_thres: float,
	scale_eps: float,
) -> int:
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
				# Grow in rare cases where metadata frame count is inaccurate.
				extra = np.zeros((1, 34), dtype=np.float32)
				sequence = np.concatenate([sequence, extra], axis=0)

			use_zero = False
			current_bbox = None
			if annotation is not None:
				current_bbox = _frame_bbox_for_index(annotation, frame_idx)
				if current_bbox is not None and _bbox_is_zero(current_bbox):
					use_zero = True

			if not use_zero:
				results = model.predict(
					source=frame_bgr,
					device=device,
					imgsz=imgsz,
					conf=conf_thres,
					verbose=False,
				)
				if results:
					selected = pick_pose_from_result(results[0], current_bbox)
					if selected is not None:
						# selected: (17, 3) => [x_px, y_px, keypoint_conf]
						coco = np.zeros((17, 3), dtype=np.float32)
						coco[:, 0] = np.clip(selected[:, 0] / max(width, 1), 0.0, 1.0)
						coco[:, 1] = np.clip(selected[:, 1] / max(height, 1), 0.0, 1.0)
						coco[:, 2] = np.clip(selected[:, 2], 0.0, 1.0)

						# Center all joints around the mid-hip point in x/y.
						l_hip = coco[11, :2]
						r_hip = coco[12, :2]
						mid_hip = (l_hip + r_hip) / 2.0
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

	output_path.parent.mkdir(parents=True, exist_ok=True)
	np.save(output_path, sequence)
	return int(sequence.shape[0])


def build_output_name(dataset_root: Path, video_path: Path) -> str:
	rel = video_path.relative_to(dataset_root)
	# Keep folder context in filename to avoid collisions across locations.
	# Example: Coffee_room_01_Coffee_room_01_Videos_video1.npy
	safe_stem = "_".join(rel.with_suffix("").parts)
	return f"{safe_stem}.npy"


def process_dataset(
	dataset_root: Path,
	output_dir: Path,
	metadata_path: Path,
	model_path: str,
	device: str,
	imgsz: int,
	conf_thres: float,
	scale_eps: float,
	mirror_aug: bool,
) -> None:
	videos = discover_videos(dataset_root)
	if not videos:
		print(f"No videos found under: {dataset_root}")
		return

	resolved_device = resolve_device(device)
	print(f"[INFO] Loading YOLO model: {model_path}")
	print(f"[INFO] Inference device: {resolved_device}")
	model = YOLO(model_path)

	metadata: List[dict] = []

	video_bar = tqdm(videos, desc="Videos", unit="video")
	for video_path, annotation_path in video_bar:
		video_bar.set_postfix_str(video_path.stem)

		annotation = None
		fall_start = None
		fall_end = None

		if annotation_path is None:
			print(f"[WARN] Missing annotation for {video_path}")
		else:
			try:
				annotation = parse_annotation_file(annotation_path)
				fall_start = annotation.fall_start
				fall_end = annotation.fall_end
			except Exception as exc:  # noqa: BLE001
				print(f"[WARN] Failed to parse annotation {annotation_path}: {exc}")

		out_name = build_output_name(dataset_root, video_path)
		out_path = output_dir / out_name

		try:
			num_frames = extract_video_pose(
				video_path=video_path,
				annotation=annotation,
				output_path=out_path,
				model=model,
				device=resolved_device,
				imgsz=imgsz,
				conf_thres=conf_thres,
				scale_eps=scale_eps,
			)
			metadata.append(
				{
					"npy_file": str(out_path.relative_to(output_dir)),
					"video_file": str(video_path.relative_to(dataset_root)),
					"annotation_file": (
						str(annotation_path.relative_to(dataset_root)) if annotation_path is not None else None
					),
					"fall_start": fall_start,
					"fall_end": fall_end,
					"num_frames": num_frames,
				}
			)

			if mirror_aug:
				try:
					sequence = np.load(out_path)
					mirrored = mirror_coco17_sequence(sequence)
					mirror_name = f"{out_path.stem}_mirror{out_path.suffix}"
					mirror_path = out_path.with_name(mirror_name)
					np.save(mirror_path, mirrored)

					metadata.append(
						{
							"npy_file": str(mirror_path.relative_to(output_dir)),
							"video_file": str(video_path.relative_to(dataset_root)),
							"annotation_file": (
								str(annotation_path.relative_to(dataset_root)) if annotation_path is not None else None
							),
							"fall_start": fall_start,
							"fall_end": fall_end,
							"num_frames": int(mirrored.shape[0]),
						}
					)
				except Exception as exc:  # noqa: BLE001
					print(f"[WARN] Failed mirror augmentation for {video_path}: {exc}")
		except Exception as exc:  # noqa: BLE001
			print(f"[WARN] Failed to process {video_path}: {exc}")

	metadata_path.parent.mkdir(parents=True, exist_ok=True)
	metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
	print(f"Saved metadata to: {metadata_path}")
	print(f"Processed videos: {len(metadata)} / {len(videos)}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Extract LE2I skeleton sequences (COCO17) from videos using YOLOv8-pose."
	)
	parser.add_argument(
		"--dataset-root",
		type=Path,
		default=Path("dataset/le2i"),
		help="Path to LE2I dataset root.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("dataset/le2i/pose_npy"),
		help="Directory where .npy files will be saved.",
	)
	parser.add_argument(
		"--metadata",
		type=Path,
		default=Path("dataset/le2i/pose_npy/metadata.json"),
		help="Path to output metadata JSON file.",
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
	process_dataset(
		dataset_root=args.dataset_root,
		output_dir=args.output_dir,
		metadata_path=args.metadata,
		model_path=args.model,
		device=args.device,
		imgsz=args.imgsz,
		conf_thres=args.conf,
		scale_eps=args.scale_eps,
		mirror_aug=args.mirror_aug,
	)


if __name__ == "__main__":
	main()
