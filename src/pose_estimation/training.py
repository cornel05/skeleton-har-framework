"""
Training pipeline for skeleton-based action recognition.
Handles dataset splitting, loading, and model training.
"""

import argparse
from pathlib import Path
import os
import shutil
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, Tuple, Dict, List, Optional
from collections import Counter

try:
    # Works when executed as a module: python -m src.pose_estimation.training
    from .dataset import SkeletonDataset, collate_fn_skeleton
    from .model import SkeletonLSTM, SkeletonLSTMWithAttention
    from .config import DATASET_CFG, MODEL_CFG, TRAINING_CFG
    from .utils import train_val_test_split_grouped, compute_metrics
except ImportError:
    # Fallback for direct script execution: python src/pose_estimation/training.py
    from dataset import SkeletonDataset, collate_fn_skeleton
    from model import SkeletonLSTM, SkeletonLSTMWithAttention
    try:
        from config import DATASET_CFG, MODEL_CFG, TRAINING_CFG
    except ImportError:
        DATASET_CFG = {}
        MODEL_CFG = {}
        TRAINING_CFG = {}
    try:
        from utils import train_val_test_split_grouped, compute_metrics
    except ImportError:
        def train_val_test_split_grouped(*args, **kwargs):
             raise ImportError("utils.py not found.")
        def compute_metrics(*args, **kwargs):
            raise ImportError("utils.py not found.")


def copy_dataset_files_to_dir(dataset: SkeletonDataset, target_dir: str) -> None:
    """
    Copy all .npy files in the dataset to a target directory.
    Checks if the target directory is not empty and appends files.
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Check if directory is empty (informational only based on user request)
    is_empty = not any(target_path.iterdir())
    if is_empty:
        print(f"Testing dataset folder '{target_dir}' is empty. Initializing with validation/test samples.")
    else:
        print(f"Testing dataset folder '{target_dir}' is not empty. Appending validation/test samples.")

    count = 0
    for src_file in dataset.abs_file_list:
        dest_file = target_path / os.path.basename(src_file)
        if not dest_file.exists():
            shutil.copy2(src_file, dest_file)
            count += 1
    
    if count > 0:
        print(f"Copied {count} new .npy files to '{target_dir}'.")
    else:
        print(f"All files already exist in '{target_dir}'.")


def log_dataset_samples_for_testing(
    dataset: SkeletonDataset,
    output_txt_path: str,
    max_samples: Optional[int] = None
) -> None:
    """
    Log dataset samples into a text file for later testing.
    """
    if len(dataset) == 0:
        return

    if max_samples is None:
        selected_indices = list(range(len(dataset)))
    else:
        max_samples = max(1, min(max_samples, len(dataset)))
        class_0_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
        class_1_indices = [i for i, label in enumerate(dataset.labels) if label == 1]

        selected_indices: List[int] = []
        half = max_samples // 2
        selected_indices.extend(class_0_indices[:half])
        selected_indices.extend(class_1_indices[:half])

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
    # Prioritize explicit segmented suffixes for LE2I
    if "_fall" in name:
        return 1
    if "_adl" in name or "_preadl" in name or "_postadl" in name:
        return 0
    # Fallback to UR Fall heuristics
    if "fall" in name or name.startswith("f-"):
        return 1
    return 0


def _group_key_from_filename(file_name: str) -> str:
    """Return grouping key so mirrored/original variants stay in the same split."""
    stem = Path(file_name).stem
    # Remove mirror suffix to keep original and mirror in the same split
    group = stem.replace("_mirror", "")
    # Note: We keep the _fall, _adl, etc. suffixes in the group key
    # so that different segments of the same video can be in different splits.
    # Frame leakage is avoided because the segments are non-overlapping.
    return group


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for sequences, masks, labels in dataloader:
        sequences = sequences.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        logits = model(sequences, masks)
        loss = loss_fn(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(1, num_batches)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for sequences, masks, labels in dataloader:
            sequences = sequences.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            logits = model(sequences, masks)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.shape[0]
    
    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = correct_predictions / max(1, total_samples)
    
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    train_labels: List[int],
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: torch.device = torch.device("cpu"),
    patience: int = 10,
    checkpoint_path: str = "best_model.pt"
) -> Dict[str, List[float]]:
    """Complete training loop with validation."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    class_counts = Counter(train_labels)
    num_classes = int(getattr(model, "num_classes", 2))
    total_samples = float(max(1, len(train_labels)))
    class_weights = []
    for cls_idx in range(num_classes):
        cls_count = float(class_counts.get(cls_idx, 0))
        if cls_count <= 0:
            class_weights.append(1.0)
        else:
            class_weights.append(total_samples / (num_classes * cls_count))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"Class weights for loss: {class_weights_tensor.detach().cpu().numpy().tolist()}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_accuracy = 0.0
    patience_counter = 0
    
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, loss_fn, device)
        val_loss, val_accuracy = evaluate(model, val_dataloader, loss_fn, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    return history


def evaluate_with_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Dict[str, Any]:
    """Evaluate model and compute detailed binary classification metrics."""
    model.eval()
    total_loss = 0.0
    all_predictions: List[int] = []
    all_labels: List[int] = []
    all_probabilities: List[np.ndarray] = []

    with torch.no_grad():
        for sequences, masks, labels in dataloader:
            sequences = sequences.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            logits = model(sequences, masks)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_probabilities.extend(probabilities.detach().cpu().numpy())

    avg_loss = total_loss / max(1, len(dataloader))
    probabilities_np = np.asarray(all_probabilities)
    predictions_np = np.asarray(all_predictions)
    labels_np = np.asarray(all_labels)

    metrics = compute_metrics(
        predictions=predictions_np,
        probabilities=probabilities_np,
        ground_truth=labels_np,
    )

    metrics["loss"] = avg_loss
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train skeleton-based fall detector")
    parser.add_argument(
        "--dataset-dir", type=str, default=DATASET_CFG.get("root_dir", "dataset/pose_npy"),
        help="Directory containing .npy skeleton files."
    )
    parser.add_argument(
        "--epochs", type=int, default=TRAINING_CFG.get("num_epochs", 50),
        help="Number of training epochs."
    )
    parser.add_argument(
        "--lr", type=float, default=TRAINING_CFG.get("learning_rate", 0.001),
        help="Learning rate for optimization."
    )
    parser.add_argument(
        "--model-path", type=str, default=TRAINING_CFG.get("checkpoint_path", "checkpoints/best_model.pt"),
        help="Path to save the best model."
    )
    parser.add_argument(
        "--testing-dataset-dir", type=str, default=DATASET_CFG.get("testing_dataset_dir", "dataset/testing_dataset"),
        help="Directory to collect validation .npy files."
    )
    parser.add_argument(
        "--sample-log-path", type=str, default=DATASET_CFG.get("sample_log_path", "dataset/testing_samples.txt"),
        help="Path to save sample list for testing."
    )
    parser.add_argument(
        "--sample-log-count", default=DATASET_CFG.get("sample_log_count"),
        help="Number of samples to log (null for all)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for split reproducibility and dataloader randomness."
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Discover all files
    dataset_root = Path(args.dataset_dir)
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {args.dataset_dir}")

    all_files = sorted([p.name for p in dataset_root.glob("*.npy")])
    if not all_files:
        raise ValueError(f"No .npy skeleton files found in: {args.dataset_dir}")

    all_labels = [_infer_label_from_filename(name) for name in all_files]

    # 2. Group-aware split (prevents mirror/original leakage across splits)
    all_group_ids = [_group_key_from_filename(name) for name in all_files]
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = train_val_test_split_grouped(
        file_list=all_files,
        labels=all_labels,
        group_ids=all_group_ids,
        train_ratio=0.8,
        val_ratio=0.15,
        random_seed=args.seed,
    )

    print(
        "Split summary | "
        f"Train: {len(train_files)} {dict(Counter(train_labels))} | "
        f"Val: {len(val_files)} {dict(Counter(val_labels))} | "
        f"Test: {len(test_files)} {dict(Counter(test_labels))}"
    )

    # 3. Create dataset objects
    train_dataset = SkeletonDataset(
        root_dir=str(dataset_root),
        file_paths=train_files,
        labels=train_labels,
        sampling_mode="random",
    )
    val_dataset = SkeletonDataset(
        root_dir=str(dataset_root),
        file_paths=val_files,
        labels=val_labels,
        sampling_mode="center",
    )
    
    test_dataset = None
    if test_files:
        test_dataset = SkeletonDataset(
            root_dir=str(dataset_root),
            file_paths=test_files,
            labels=test_labels,
            sampling_mode="center",
        )

    # 4. Create dataloaders
    batch_size = DATASET_CFG.get("batch_size", 8)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_skeleton)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_skeleton)
    test_dataloader = None
    if test_dataset is not None and len(test_dataset) > 0:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_skeleton)

    # 5. Log and copy validation/test files
    if args.sample_log_path:
        log_dataset_samples_for_testing(dataset=val_dataset, output_txt_path=args.sample_log_path, max_samples=args.sample_log_count)

    if args.testing_dataset_dir:
        copy_dataset_files_to_dir(val_dataset, args.testing_dataset_dir)
        if test_dataset:
            copy_dataset_files_to_dir(test_dataset, args.testing_dataset_dir)

    # 6. Model setup and training
    feature_dim = train_dataset.get_feature_dim()
    model_type = MODEL_CFG.get("type", "lstm")
    model_params = {
        "input_dim": feature_dim,
        "hidden_size": MODEL_CFG.get("hidden_size", 64),
        "num_layers": MODEL_CFG.get("num_layers", 1),
        "dropout": MODEL_CFG.get("dropout", 0.2),
        "num_classes": MODEL_CFG.get("num_classes", 2),
        "bidirectional": MODEL_CFG.get("bidirectional", False),
    }

    if model_type == "lstm_attention":
        model_params["attention_context"] = MODEL_CFG.get("attention_context", 5)
        model = SkeletonLSTMWithAttention(**model_params)
    else:
        model = SkeletonLSTM(**model_params)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    history = train_model(
        model, train_dataloader, val_dataloader,
        train_labels=train_labels,
        num_epochs=args.epochs, learning_rate=args.lr, device=device,
        patience=TRAINING_CFG.get("patience", 10), checkpoint_path=args.model_path
    )

    print(f"\nTraining complete! Best validation accuracy: {max(history['val_accuracy']):.4f}")

    # 7. Strict holdout test evaluation from best checkpoint
    if test_dataloader is not None:
        best_state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(best_state_dict)
        model.to(device)

        test_metrics = evaluate_with_metrics(
            model=model,
            dataloader=test_dataloader,
            loss_fn=nn.CrossEntropyLoss(),
            device=device,
        )

        cm = test_metrics["confusion_matrix"]
        print("\nTest set evaluation (best checkpoint):")
        print(
            f"Test Loss: {test_metrics['loss']:.4f} | "
            f"Accuracy: {test_metrics['accuracy']:.4f} | "
            f"Precision: {test_metrics['precision']:.4f} | "
            f"Recall: {test_metrics['recall']:.4f} | "
            f"F1: {test_metrics['f1']:.4f} | "
            f"ROC-AUC: {test_metrics['roc_auc']:.4f}"
        )
        print("Confusion Matrix [[TN, FP], [FN, TP]]:")
        print(cm)
