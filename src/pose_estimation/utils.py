"""
Utility functions for skeleton-based action recognition.

Provides data processing helpers, evaluation metrics, and visualization tools.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def resolve_device(device_arg: str = "auto") -> torch.device:
    """
    Resolve torch device from string argument.
    
    Args:
        device_arg: "auto", "cuda", "cpu", or specific device index (e.g., "cuda:0")
    
    Returns:
        torch.device object
    """
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


# ============================================================================
# Data Processing Utilities
# ============================================================================

def normalize_skeleton(skeleton: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize skeleton keypoints to a standard range.
    
    Args:
        skeleton (np.ndarray): Shape (L, D) skeleton sequence
        method: Normalization method
            - "minmax": Normalize to [0, 1]
            - "zscore": Standardize to mean=0, std=1
            - "robust": Robust scaling using median and IQR
    
    Returns:
        Normalized skeleton array
    """
    if method == "minmax":
        # Per-dimension min-max normalization
        min_vals = skeleton.min(axis=0, keepdims=True)
        max_vals = skeleton.max(axis=0, keepdims=True)
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        return (skeleton - min_vals) / range_vals
    
    elif method == "zscore":
        # Per-dimension standardization
        mean = skeleton.mean(axis=0, keepdims=True)
        std = skeleton.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (skeleton - mean) / std
    
    elif method == "robust":
        # Robust scaling using median and IQR
        median = np.median(skeleton, axis=0, keepdims=True)
        q25 = np.percentile(skeleton, 25, axis=0, keepdims=True)
        q75 = np.percentile(skeleton, 75, axis=0, keepdims=True)
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0
        return (skeleton - median) / iqr
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def remove_static_joints(
    skeleton: np.ndarray,
    threshold: float = 0.1
) -> np.ndarray:
    """
    Remove joints that don't move significantly (static in scene).
    
    Identifies joints with low temporal variance and sets them to zero.
    Useful for filtering tracking artifacts or scene-fixed markers.
    
    Args:
        skeleton (np.ndarray): Shape (L, D) skeleton sequence
        threshold: Variance threshold below which joint is considered static
    
    Returns:
        Skeleton with static joints zeroed out
    """
    skeleton_out = skeleton.copy()
    
    # Compute temporal variance per joint
    variances = skeleton.var(axis=0)  # Shape: (D,)
    
    # Identify static joints
    static_mask = variances < threshold
    
    # Zero out static joints
    skeleton_out[:, static_mask] = 0.0
    
    return skeleton_out


def interpolate_missing_frames(
    skeleton: np.ndarray,
    missing_indicator: np.ndarray
) -> np.ndarray:
    """
    Interpolate missing keypoints (e.g., when pose detector fails).
    
    Args:
        skeleton (np.ndarray): Shape (L, D) skeleton sequence
        missing_indicator (np.ndarray): Shape (L,) binary array
            - 1: frame has missing keypoints
            - 0: frame is valid
    
    Returns:
        Skeleton with missing frames interpolated
    """
    skeleton_out = skeleton.copy()
    
    for dim in range(skeleton.shape[1]):
        # Find valid and invalid frame indices
        valid_idx = np.where(missing_indicator == 0)[0]
        invalid_idx = np.where(missing_indicator == 1)[0]
        
        if len(valid_idx) > 0 and len(invalid_idx) > 0:
            # Linear interpolation
            skeleton_out[invalid_idx, dim] = np.interp(
                invalid_idx,
                valid_idx,
                skeleton[valid_idx, dim]
            )
    
    return skeleton_out


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_metrics(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for binary classification.
    
    Args:
        predictions (np.ndarray): Predicted class indices, shape (N,)
        probabilities (np.ndarray): Predicted probabilities, shape (N, 2)
        ground_truth (np.ndarray): True labels, shape (N,)
    
    Returns:
        Dictionary containing:
            - accuracy: Classification accuracy
            - precision: Precision for positive class (fall)
            - recall: Recall for positive class (fall)
            - f1: F1-score for positive class (fall)
            - roc_auc: ROC-AUC score
            - confusion_matrix: Confusion matrix
    """
    # Compute confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1-score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(ground_truth, probabilities[:, 1])
    except:
        roc_auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def sensitivity_specificity(
    predictions: np.ndarray,
    ground_truth: np.ndarray
) -> Tuple[float, float]:
    """
    Compute sensitivity and specificity.
    
    For fall detection:
        - Sensitivity (recall): Ability to detect falls (TP / (TP + FN))
        - Specificity: Ability to correctly identify non-falls (TN / (TN + FP))
    
    Args:
        predictions (np.ndarray): Predicted class indices
        ground_truth (np.ndarray): True labels
    
    Returns:
        Tuple of (sensitivity, specificity)
    """
    cm = confusion_matrix(ground_truth, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return sensitivity, specificity


# ============================================================================
# Visualization
# ============================================================================

def plot_training_history(history: Dict[str, List[float]]) -> None:
    """
    Plot training history.
    
    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 'val_accuracy'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['val_accuracy'], label='Val Accuracy', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None
) -> None:
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix, shape (2, 2)
        class_names: List of class names (default: ['Non-fall', 'Fall'])
    """
    if class_names is None:
        class_names = ['Non-fall', 'Fall']
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot heatmap
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    
    # Labels
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14
            )
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(
    ground_truth: np.ndarray,
    probabilities: np.ndarray
) -> None:
    """
    Plot ROC curve.
    
    Args:
        ground_truth (np.ndarray): True labels
        probabilities (np.ndarray): Predicted probabilities for positive class
    """
    fpr, tpr, _ = roc_curve(ground_truth, probabilities)
    auc = roc_auc_score(ground_truth, probabilities)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.3)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Fall Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


# ============================================================================
# Data Splitting Utilities
# ============================================================================

def train_val_test_split(
    file_list: List[str],
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[Tuple[List, List], Tuple[List, List], Tuple[List, List]]:
    """
    Split data into train, validation, and test sets with stratification.
    
    Ensures class distribution is maintained across splits.
    Args:
        file_list: List of file paths
        labels: List of labels
        train_ratio: Proportion for training (default: 0.7)
        val_ratio: Proportion for validation (default: 0.15)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of ((train_files, train_labels), (val_files, val_labels), (test_files, test_labels))

    Example:
        >>> (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = \
        ...     train_val_test_split(file_list, labels)
    """
    if len(file_list) != len(labels):
        raise ValueError("file_list and labels must have the same length")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    indices = np.arange(len(file_list))
    labels_arr = np.asarray(labels)

    stratify_labels = labels_arr if len(np.unique(labels_arr)) > 1 else None

    # First split: train vs (val+test)
    train_indices, temp_indices, train_labels_arr, temp_labels_arr = train_test_split(
        indices,
        labels_arr,
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True,
        stratify=stratify_labels,
    )

    # Second split: val vs test from the remaining pool
    remaining_ratio = 1.0 - train_ratio
    if val_ratio == 0.0:
        val_indices = np.array([], dtype=int)
        val_labels_arr = np.array([], dtype=labels_arr.dtype)
        test_indices = temp_indices
        test_labels_arr = temp_labels_arr
    else:
        val_within_remaining = val_ratio / remaining_ratio
        n_val = int(round(len(temp_indices) * val_within_remaining))
        n_val = max(1, min(n_val, len(temp_indices) - 1))
        temp_stratify = temp_labels_arr if len(np.unique(temp_labels_arr)) > 1 else None
        val_indices, test_indices, val_labels_arr, test_labels_arr = train_test_split(
            temp_indices,
            temp_labels_arr,
            train_size=n_val,
            random_state=random_seed,
            shuffle=True,
            stratify=temp_stratify,
        )
    
    # Extract files and labels
    train_files = [file_list[int(i)] for i in train_indices]
    train_labels = [int(label) for label in train_labels_arr]
    
    val_files = [file_list[int(i)] for i in val_indices]
    val_labels = [int(label) for label in val_labels_arr]
    
    test_files = [file_list[int(i)] for i in test_indices]
    test_labels = [int(label) for label in test_labels_arr]
    
    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)


# ============================================================================
# Model Analysis Utilities
# ============================================================================

def compute_model_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device = torch.device("cpu")
) -> int:
    """
    Estimate model FLOPs (floating point operations).
    
    Note: This is an approximation. Consider using fvcore.nn.flop_count
    for more accurate measurements.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (B, T, D)
        device: Computation device
    
    Returns:
        Estimated number of FLOPs
    """
    from fvcore.nn import FlopCountAnalysis
    
    dummy_input = torch.randn(input_shape, device=device)
    flops = FlopCountAnalysis(model, dummy_input).total()
    return flops


def compute_inference_time(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_runs: int = 100,
    device: torch.device = torch.device("cpu")
) -> Dict[str, float]:
    """
    Measure inference time.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        num_runs: Number of inference runs for averaging
        device: Computation device
    
    Returns:
        Dictionary with timing statistics (ms)
            - mean: Average inference time
            - min: Minimum inference time
            - max: Maximum inference time
            - std: Standard deviation
    """
    import time
    
    model = model.to(device)
    model.eval()
    
    times = []
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            x = torch.randn(input_shape, device=device)
            _ = model(x)
        
        # Measurement
        for _ in range(num_runs):
            x = torch.randn(input_shape, device=device)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            _ = model(x)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end = time.time()
            
            times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean': np.mean(times),
        'min': np.min(times),
        'max': np.max(times),
        'std': np.std(times)
    }
