"""
Inference script for skeleton-based fall detection.
Supports both video files and image streams (folders of images).
"""

import argparse
import os
from pathlib import Path
from typing import Any, Tuple, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from .dataset import SkeletonDataset
    from .model import SkeletonLSTM, SkeletonLSTMWithAttention, build_model
    from .config import DATASET_CFG, MODEL_CFG, INFERENCE_CFG, TRAINING_CFG
    from .features import normalize_repo_features
except ImportError:
    from dataset import SkeletonDataset
    from model import SkeletonLSTM, SkeletonLSTMWithAttention, build_model
    from features import normalize_repo_features
    try:
        from config import DATASET_CFG, MODEL_CFG, INFERENCE_CFG, TRAINING_CFG
    except ImportError:
        DATASET_CFG = {}
        MODEL_CFG = {}
        INFERENCE_CFG = {}
        TRAINING_CFG = {}


DEFAULT_POSE_MODEL = "models/yolov8n-pose.pt"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _yolo_keypoints(frames, pose_model_path: str, imgsz: int = 320, conf: float = 0.25):
    """
    Run YOLOv8-pose on an iterable of BGR frames exactly as the training-data extraction does
    (highest-confidence person, same imgsz/conf) and return raw (T, 17, 3) keypoints (NaN = no person).
    """
    from ultralytics import YOLO
    try:
        from .preprocessing.urfall import pick_pose_from_result
    except ImportError:
        from preprocessing.urfall import pick_pose_from_result

    model = YOLO(pose_model_path)
    kpts, width, height, total = [], 0, 0, 0
    for frame in frames:
        total += 1
        height, width = frame.shape[:2]
        picked = pick_pose_from_result(model.predict(frame, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0])
        kpts.append(picked[0] if picked is not None else np.full((17, 3), np.nan, np.float32))
    arr = np.stack(kpts) if kpts else np.zeros((0, 17, 3), np.float32)
    return arr, width, height, total


def _features_from_frames(frames, pose_model_path: str) -> Tuple[np.ndarray, int, int]:
    kpts, width, height, total = _yolo_keypoints(frames, pose_model_path)
    valid = int((~np.isnan(kpts[:, 0, 0])).sum()) if len(kpts) else 0
    if valid == 0:
        return np.zeros((0, 34), np.float32), total, 0
    return normalize_repo_features(kpts, width, height), total, valid


def _extract_keypoints_from_video(video_path: str, pose_model_path: str = DEFAULT_POSE_MODEL
                                  ) -> Tuple[np.ndarray, float, int, int]:
    """Per-frame 34-d features for a video file (YOLOv8-pose, same normalization as training)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def frames():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()

    feats, total, valid = _features_from_frames(frames(), pose_model_path)
    return feats, float(fps), total, valid


def _extract_keypoints_from_image_stream(folder_path: str, fps: float = 30.0,
                                         pose_model_path: str = DEFAULT_POSE_MODEL
                                         ) -> Tuple[np.ndarray, float, int, int]:
    """Per-frame 34-d features for a folder of images sorted by name."""
    import cv2

    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder_path}")
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise ValueError(f"No images found in: {folder_path}")
    frames = (img for img in (cv2.imread(str(f)) for f in files) if img is not None)
    feats, total, valid = _features_from_frames(frames, pose_model_path)
    return feats, float(fps), total, valid


def _load_checkpoint_safely(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return {"state_dict": checkpoint["state_dict"], "meta": checkpoint}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return {"state_dict": checkpoint["model_state_dict"], "meta": checkpoint}
    if isinstance(checkpoint, dict):
        return {"state_dict": checkpoint, "meta": {}}

    raise ValueError("Unsupported checkpoint format. Expected state_dict-like object.")


def _infer_model_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    if "lstm.weight_ih_l0" not in state_dict:
        raise KeyError("Checkpoint missing 'lstm.weight_ih_l0'; cannot infer model config.")

    input_dim = int(state_dict["lstm.weight_ih_l0"].shape[1])
    hidden_size = int(state_dict["lstm.weight_hh_l0"].shape[1])
    num_classes = int(state_dict["classifier.weight"].shape[0])

    num_layers = 0
    while f"lstm.weight_ih_l{num_layers}" in state_dict:
        num_layers += 1
    num_layers = max(1, num_layers)

    bidirectional = "lstm.weight_ih_l0_reverse" in state_dict

    return {
        "input_dim": input_dim,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": 0.0,
        "num_classes": num_classes,
        "bidirectional": bidirectional,
    }


def load_model_for_inference(
    checkpoint_path: str,
    device: torch.device
) -> nn.Module:
    """Load a trained SkeletonLSTM or SkeletonLSTMWithAttention from checkpoint."""
    loaded = _load_checkpoint_safely(checkpoint_path, device)
    state_dict = loaded["state_dict"]

    if "lstm.weight_ih_l0" not in state_dict:  # GRU / TCN / ST-GCN checkpoints from the evaluation protocol
        kind = "gru" if "gru.weight_ih_l0" in state_dict else "stgcn" if any(
            k.startswith("blocks.") for k in state_dict) else "tcn"
        model = build_model(kind, input_dim=34)
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval()

    model_config = _infer_model_config_from_state_dict(state_dict)
    uses_attention = any(k.startswith("attention.") for k in state_dict.keys())

    if uses_attention:
        model = SkeletonLSTMWithAttention(**model_config)
    else:
        model = SkeletonLSTM(**model_config)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _match_feature_dimension(sequence: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Align input feature dimension to model's expected input_dim.
    """
    if sequence.shape[1] == target_dim:
        return sequence
    if sequence.shape[1] > target_dim:
        return sequence[:, :target_dim]

    padded = np.zeros((sequence.shape[0], target_dim), dtype=np.float32)
    padded[:, :sequence.shape[1]] = sequence
    return padded


def predict_fall(
    model: nn.Module,
    source_path: str,
    is_image_stream: bool = False,
    sequence_length: int = INFERENCE_CFG.get("sequence_length", 32),
    threshold: float = INFERENCE_CFG.get("threshold", 0.5),
    stride: Optional[int] = INFERENCE_CFG.get("stride", 16),
    device: Optional[torch.device] = None,
    fps: float = 30.0,
    pose_model_path: str = DEFAULT_POSE_MODEL,
) -> Dict[str, Any]:
    """
    Run offline fall detection on a video or an image folder.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")

    if stride is None or stride <= 0:
        stride = max(1, sequence_length // 2)

    if is_image_stream:
        sequence, fps, total_frames, valid_pose_frames = _extract_keypoints_from_image_stream(
            source_path, fps=fps, pose_model_path=pose_model_path)
    else:
        sequence, fps, total_frames, valid_pose_frames = _extract_keypoints_from_video(
            source_path, pose_model_path=pose_model_path)

    if valid_pose_frames == 0:
        return {
            "source_path": source_path,
            "detected_fall": False,
            "fall_probability": 0.0,
            "window_probabilities": [],
            "fps": fps,
            "total_frames": total_frames,
            "valid_pose_frames": 0,
            "note": "No person detected in source.",
        }

    input_dim = int(model.input_dim) if hasattr(model, "input_dim") else sequence.shape[1]
    sequence = _match_feature_dimension(sequence, input_dim)

    windows: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    window_ranges: List[Tuple[int, int]] = []

    if sequence.shape[0] < sequence_length:
        padded = np.zeros((sequence_length, sequence.shape[1]), dtype=np.float32)
        padded[:sequence.shape[0]] = sequence
        mask = np.zeros(sequence_length, dtype=np.float32)
        mask[:sequence.shape[0]] = 1.0
        windows.append(padded)
        masks.append(mask)
        window_ranges.append((0, sequence.shape[0]))
    else:
        for start in range(0, sequence.shape[0] - sequence_length + 1, stride):
            end = start + sequence_length
            windows.append(sequence[start:end])
            masks.append(np.ones(sequence_length, dtype=np.float32))
            window_ranges.append((start, end))

    sequences_tensor = torch.from_numpy(np.stack(windows)).float().to(device)
    masks_tensor = torch.from_numpy(np.stack(masks)).float().to(device)

    with torch.no_grad():
        _, probabilities = model.predict(sequences_tensor, masks_tensor)

    fall_probs = probabilities[:, 1].detach().cpu().numpy()
    max_prob = float(fall_probs.max()) if len(fall_probs) > 0 else 0.0
    detected_fall = bool(max_prob >= threshold)

    window_summaries: List[Dict[str, Any]] = []
    for idx, prob in enumerate(fall_probs.tolist()):
        start_frame, end_frame = window_ranges[idx]
        window_summaries.append(
            {
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_time_sec": float(start_frame / fps),
                "end_time_sec": float(end_frame / fps),
                "fall_probability": float(prob),
            }
        )

    return {
        "source_path": source_path,
        "detected_fall": detected_fall,
        "fall_probability": max_prob,
        "window_probabilities": window_summaries,
        "fps": fps,
        "total_frames": total_frames,
        "valid_pose_frames": valid_pose_frames,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test skeleton-based fall detector")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to a video file or a folder of images.",
    )
    parser.add_argument(
        "--is-image-stream",
        action="store_true",
        help="If set, source is treated as a folder of images.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=TRAINING_CFG.get("checkpoint_path", "checkpoints/best_model.pt"),
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=INFERENCE_CFG.get("sequence_length", 32),
        help="Sequence length used for inference windows.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=INFERENCE_CFG.get("threshold", 0.5),
        help="Fall probability threshold for final decision.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=INFERENCE_CFG.get("stride", 16),
        help="Sliding-window stride for inference.",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        default=DEFAULT_POSE_MODEL,
        help="YOLOv8-pose weights (must match the extractor used for training data).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS to assume for image stream.",
    )

    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.isfile(args.model_path):
        # Try relative to project root if not absolute
        print(f"[WARN] Model checkpoint not found at {args.model_path}, trying to resolve...")
        
    model = load_model_for_inference(args.model_path, device=device)
    result = predict_fall(
        model=model,
        source_path=args.source,
        is_image_stream=args.is_image_stream,
        sequence_length=args.sequence_length,
        threshold=args.threshold,
        stride=args.stride,
        device=device,
        fps=args.fps,
        pose_model_path=args.pose_model,
    )

    print("=== Inference Result ===")
    print(f"Source: {result['source_path']}")
    print(f"Total frames: {result['total_frames']}")
    print(f"Frames with detected pose: {result['valid_pose_frames']}")
    print(f"Final fall probability (max over windows): {result['fall_probability']:.4f}")
    print(f"Detected fall: {result['detected_fall']}")

    top_windows = sorted(
        result["window_probabilities"],
        key=lambda x: x["fall_probability"],
        reverse=True,
    )[:5]
    if top_windows:
        print("Top windows:")
        for item in top_windows:
            print(
                f"  frames [{item['start_frame']}, {item['end_frame']}) | "
                f"time [{item['start_time_sec']:.2f}s, {item['end_time_sec']:.2f}s) | "
                f"fall_prob={item['fall_probability']:.4f}"
            )
