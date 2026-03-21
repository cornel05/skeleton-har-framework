"""
Example usage and training pipeline for skeleton-based action recognition.

Demonstrates how to use the SkeletonDataset, DataLoader, and SkeletonLSTM
for training a fall detection model.
"""

import argparse
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, Tuple, Dict, List, Optional

try:
    # Works when executed as a module: python -m src.pose_estimation.training
    from .dataset import SkeletonDataset, collate_fn_skeleton
    from .model import SkeletonLSTM, SkeletonLSTMWithAttention
except ImportError:
    # Fallback for direct script execution: python src/pose_estimation/training.py
    from dataset import SkeletonDataset, collate_fn_skeleton
    from model import SkeletonLSTM, SkeletonLSTMWithAttention


def log_dataset_samples_for_testing(
    dataset: SkeletonDataset,
    output_txt_path: str,
    max_samples: int = 20
) -> None:
    """
    Log representative dataset samples into a text file for later testing.

    The output contains file name and label so you can quickly pick examples
    for offline validation.
    """
    if len(dataset) == 0:
        return

    max_samples = max(1, min(max_samples, len(dataset)))

    # Keep a roughly balanced subset by class when possible.
    class_0_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
    class_1_indices = [i for i, label in enumerate(dataset.labels) if label == 1]

    selected_indices: List[int] = []
    half = max_samples // 2
    selected_indices.extend(class_0_indices[:half])
    selected_indices.extend(class_1_indices[:half])

    # Fill remaining slots from the full index order.
    if len(selected_indices) < max_samples:
        for i in range(len(dataset)):
            if i not in selected_indices:
                selected_indices.append(i)
            if len(selected_indices) >= max_samples:
                break

    output_path = Path(output_txt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fp:
        fp.write("# Sample files for testing\n")
        fp.write("# Format: relative_path,label\n")
        for idx in selected_indices:
            rel_path = dataset.file_list[idx]
            label = dataset.labels[idx]
            fp.write(f"{rel_path},{label}\n")

    print(f"Logged {len(selected_indices)} dataset sample(s) to: {output_path}")


def _infer_label_from_filename(file_name: str) -> int:
    """Infer binary label from filename using simple fall/non-fall heuristics."""
    name = file_name.lower()
    if "fall" in name or name.startswith("f-"):
        return 1
    return 0


def create_dataset_from_directory(
    root_dir: str,
    sequence_length: int = 32,
    batch_size: int = 8,
    max_files: Optional[int] = None,
    sample_log_path: Optional[str] = "dataset/testing_samples.txt",
    sample_log_count: int = 20
) -> Tuple[SkeletonDataset, DataLoader]:
    """
    Build a dataset/dataloader from all skeleton files in a folder.

    Labels are inferred from filename:
        - contains 'fall' -> 1
        - otherwise -> 0
    """
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root_dir}")

    all_files = sorted([p.name for p in root.glob("*.npy")])
    if not all_files:
        raise ValueError(f"No .npy skeleton files found in: {root_dir}")

    if max_files is not None and max_files > 0:
        all_files = all_files[:max_files]

    labels = [_infer_label_from_filename(name) for name in all_files]

    dataset = SkeletonDataset(
        root_dir=str(root),
        file_paths=all_files,
        labels=labels,
        sequence_length=sequence_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_skeleton,
        num_workers=0
    )

    if sample_log_path:
        log_dataset_samples_for_testing(
            dataset=dataset,
            output_txt_path=sample_log_path,
            max_samples=sample_log_count,
        )

    return dataset, dataloader


def create_dummy_dataset_example() -> Tuple[SkeletonDataset, DataLoader]:
    """
    Example: Create a dataset from dummy skeleton files.
    
    In practice, you would:
    1. Scan your data directory for .npy files
    2. Organize them into train/val/test splits
    3. Assign labels based on ground truth (fall vs non-fall)
    
    Returns:
        Tuple of (dataset, dataloader)
    """
    # Example file structure:
    # data/
    #   ├── adl-01-cam0-rgb_skeleton.npy  (non-fall, label=0)
    #   ├── adl-02-cam0-rgb_skeleton.npy  (non-fall, label=0)
    #   ├── fall-01-cam0-rgb_skeleton.npy (fall, label=1)
    #   └── ...
    
    # File paths relative to root_dir
    file_paths = [
        "adl-01-cam0-rgb_skeleton.npy",
        "adl-02-cam0-rgb_skeleton.npy",
        "fall-01-cam0-rgb_skeleton.npy",
        "fall-02-cam0-rgb_skeleton.npy",
    ]
    
    # Labels: 0=ADL (non-fall), 1=Fall
    labels = [0, 0, 1, 1]
    
    # Create dataset
    dataset = SkeletonDataset(
        root_dir="dataset/pose_npy",
        file_paths=file_paths,
        labels=labels,
        sequence_length=32  # Fixed temporal length
    )
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn_skeleton,
        num_workers=0  # Increase for multi-GPU training
    )

    # Write a small sample list that can be reused for quick testing.
    log_dataset_samples_for_testing(
        dataset=dataset,
        output_txt_path="dataset/testing_samples.txt",
        max_samples=4,
    )
    
    return dataset, dataloader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: SkeletonLSTM model
        dataloader: Training dataloader
        optimizer: Optimizer (Adam, SGD, etc.)
        loss_fn: Loss function (CrossEntropyLoss for classification)
        device: Computation device (cpu or cuda)
    
    Returns:
        Average loss over the epoch
    """
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for sequences, masks, labels in dataloader:
        # Move data to device (GPU or CPU)
        sequences = sequences.to(device)  # (B, T, D)
        masks = masks.to(device)  # (B, T)
        labels = labels.to(device)  # (B,)
        
        # Forward pass
        logits = model(sequences, masks)  # (B, num_classes)
        
        # Compute loss
        loss = loss_fn(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (optional, helps with stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimization step
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model on validation/test set.
    
    Args:
        model: SkeletonLSTM model
        dataloader: Validation/test dataloader
        loss_fn: Loss function
        device: Computation device
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for sequences, masks, labels in dataloader:
            sequences = sequences.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(sequences, masks)
            
            # Compute loss
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.shape[0]
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: torch.device = torch.device("cpu")
) -> Dict[str, List[float]]:
    """
    Complete training loop with validation.
    
    Args:
        model: SkeletonLSTM model
        train_dataloader: Training data
        val_dataloader: Validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Computation device
    
    Returns:
        Dictionary with training history:
            - 'train_loss': List of average train losses per epoch
            - 'val_loss': List of average val losses per epoch
            - 'val_accuracy': List of validation accuracies per epoch
    
    Example:
        >>> history = train_model(
        ...     model,
        ...     train_dl,
        ...     val_dl,
        ...     num_epochs=50,
        ...     learning_rate=0.001
        ... )
        >>> print(f"Best accuracy: {max(history['val_accuracy']):.4f}")
    """
    model = model.to(device)
    
    # Optimizer: Adam is a good choice for LSTM models
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function for binary classification
    loss_fn = nn.CrossEntropyLoss()
    
    # Optional: Learning rate scheduler for decay
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_accuracy = 0.0
    patience_counter = 0
    patience = 10  # Early stopping patience
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(
            model, train_dataloader, optimizer, loss_fn, device
        )
        history['train_loss'].append(train_loss)
        
        # Evaluate
        val_loss, val_accuracy = evaluate(
            model, val_dataloader, loss_fn, device
        )
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Logging
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )
        
        # Early stopping
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    return history


def inference_example() -> None:
    """
    Example: Run inference on a single batch.
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = SkeletonLSTM(
        input_dim=34,  # COCO keypoints
        hidden_size=64,
        num_classes=2
    ).to(device)
    
    # Create dummy batch
    batch_size = 4
    seq_length = 32
    feature_dim = 34
    
    sequences = torch.randn(batch_size, seq_length, feature_dim).to(device)
    masks = torch.ones(batch_size, seq_length).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(sequences, masks)
        predictions, probabilities = model.predict(sequences, masks)
    
    # Print results
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions: {predictions}")
    print(f"Fall probabilities: {probabilities[:, 1]}")


def _extract_mediapipe_keypoints_from_video(
    video_path: str
) -> Tuple[np.ndarray, float, int, int]:
    """
    Extract 2D pose keypoints per frame from a video using MediaPipe Pose.

    Returns:
        keypoints: (L, D) array
        fps: video frame rate
        total_frames: total frames read
        valid_pose_frames: frames where a pose was found
    """
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Video inference needs opencv-python and mediapipe. "
            "Install them in your environment first."
        ) from exc

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    mp_pose = mp.solutions.pose
    sequence: List[np.ndarray] = []
    total_frames = 0

    with mp_pose.Pose() as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            total_frames += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            if result.pose_landmarks:
                keypoints: List[float] = []
                for landmark in result.pose_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y])
                sequence.append(np.asarray(keypoints, dtype=np.float32))

    cap.release()

    if not sequence:
        return np.zeros((0, 0), dtype=np.float32), float(fps), total_frames, 0

    keypoints_array = np.stack(sequence, axis=0).astype(np.float32)
    return keypoints_array, float(fps), total_frames, len(sequence)


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

    - If sequence has more features: truncate.
    - If sequence has fewer features: right-pad with zeros.
    """
    if sequence.shape[1] == target_dim:
        return sequence
    if sequence.shape[1] > target_dim:
        return sequence[:, :target_dim]

    padded = np.zeros((sequence.shape[0], target_dim), dtype=np.float32)
    padded[:, :sequence.shape[1]] = sequence
    return padded


def predict_fall_from_video(
    model: nn.Module,
    video_path: str,
    sequence_length: int = 32,
    threshold: float = 0.5,
    stride: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Run offline fall detection on a complete (non-streaming) video.

    Returns prediction summary with per-window probabilities.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")

    if stride is None or stride <= 0:
        stride = max(1, sequence_length // 2)

    sequence, fps, total_frames, valid_pose_frames = _extract_mediapipe_keypoints_from_video(video_path)

    if valid_pose_frames == 0:
        return {
            "video_path": video_path,
            "detected_fall": False,
            "fall_probability": 0.0,
            "window_probabilities": [],
            "fps": fps,
            "total_frames": total_frames,
            "valid_pose_frames": 0,
            "note": "No pose landmarks detected in video.",
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
        "video_path": video_path,
        "detected_fall": detected_fall,
        "fall_probability": max_prob,
        "window_probabilities": window_summaries,
        "fps": fps,
        "total_frames": total_frames,
        "valid_pose_frames": valid_pose_frames,
    }


# ============================================================================
# Training script template
# ============================================================================

if __name__ == "__main__":
    """
    Main training script.
    
    To adapt for your dataset:
    1. Update file_paths and labels in create_dummy_dataset_example()
    2. Verify input_dim matches your skeleton keypoints (34 for COCO)
    3. Adjust sequence_length, batch_size, learning_rate as needed
    """
    
    parser = argparse.ArgumentParser(description="Train and test skeleton-based fall detector")
    parser.add_argument(
        "--mode",
        choices=["train", "test-video"],
        default="train",
        help="Run training loop or test a full offline video.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset/pose_npy",
        help="Directory containing .npy skeleton files.",
    )
    parser.add_argument(
        "--sample-log-path",
        type=str,
        default="dataset/testing_samples.txt",
        help="Path to save sample list for testing.",
    )
    parser.add_argument(
        "--sample-log-count",
        type=int,
        default=20,
        help="Number of dataset samples to log into the sample file.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to a non-streaming video file for inference.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="best_model.pt",
        help="Path to trained checkpoint (for test-video mode).",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=32,
        help="Sequence length used for training and inference windows.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fall probability threshold for final decision.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Sliding-window stride for video inference.",
    )

    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == "test-video":
        if not args.video_path:
            raise ValueError("--video-path is required when --mode test-video")

        if not os.path.isfile(args.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

        model = load_model_for_inference(args.model_path, device=device)
        result = predict_fall_from_video(
            model=model,
            video_path=args.video_path,
            sequence_length=args.sequence_length,
            threshold=args.threshold,
            stride=args.stride,
            device=device,
        )

        print("\n=== Video Inference Result ===")
        print(f"Video: {result['video_path']}")
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
    else:
        # Create datasets and log representative samples for testing.
        train_dataset, train_dataloader = create_dataset_from_directory(
            root_dir=args.dataset_dir,
            sequence_length=args.sequence_length,
            batch_size=8,
            sample_log_path=args.sample_log_path,
            sample_log_count=args.sample_log_count,
        )
        val_dataset, val_dataloader = create_dataset_from_directory(
            root_dir=args.dataset_dir,
            sequence_length=args.sequence_length,
            batch_size=8,
            sample_log_path=None,
        )

        # Model architecture
        feature_dim = train_dataset.get_feature_dim()

        model = SkeletonLSTM(
            input_dim=feature_dim,
            hidden_size=64,
            num_layers=1,
            dropout=0.2,
            num_classes=2,
            bidirectional=False,
        )

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train
        history = train_model(
            model,
            train_dataloader,
            val_dataloader,
            num_epochs=50,
            learning_rate=0.001,
            device=device,
        )

        print("\nTraining complete!")
        print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
