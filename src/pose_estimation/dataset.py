"""
Skeleton-based action recognition dataset module.

Handles loading, preprocessing, and batching of skeleton sequences with variable lengths.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Union, Optional


class SkeletonDataset(Dataset):
    """
    PyTorch Dataset for skeleton-based action recognition.
    
    Loads skeleton keypoint sequences from .npy files and handles variable-length
    sequences through windowing and padding.
    
    Attributes:
        root_dir (str): Root directory containing skeleton .npy files
        sequence_length (int): Target sequence length (T)
        file_list (List[str]): List of .npy file paths
        labels (List[int]): Labels corresponding to each file (0=non-fall, 1=fall)
    """
    
    def __init__(
        self,
        root_dir: str,
        file_paths: List[str],
        labels: List[int],
        sequence_length: int = 32
    ):
        """
        Initialize the skeleton dataset.
        
        Args:
            root_dir: Root directory for file path resolution
            file_paths: List of relative paths to .npy skeleton files
            labels: List of labels (0=non-fall, 1=fall) corresponding to file_paths
            sequence_length: Target temporal length T for all sequences
            
        Raises:
            AssertionError: If lengths of file_paths and labels don't match
            FileNotFoundError: If root_dir doesn't exist
        """
        assert len(file_paths) == len(labels), \
            f"Number of files ({len(file_paths)}) must match labels ({len(labels)})"
        
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")
        
        self.root_dir = root_dir
        self.file_list = file_paths
        self.labels = labels
        self.sequence_length = sequence_length
        
        # Convert to absolute paths
        self.abs_file_list = [
            os.path.join(root_dir, f) for f in file_paths
        ]
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Load and preprocess a single skeleton sequence.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple containing:
                - sequence (torch.Tensor): Shape (T, D), skeleton keypoint sequence
                - mask (torch.Tensor): Shape (T,), binary mask (1=valid, 0=padded)
                - label (int): Action label (0=non-fall, 1=fall)
                
        Raises:
            FileNotFoundError: If .npy file doesn't exist
            ValueError: If loaded array has incorrect shape
        """
        file_path = self.abs_file_list[idx]
        label = self.labels[idx]
        
        # Load skeleton sequence from numpy file
        try:
            skeleton = np.load(file_path)  # Shape: (L, D)
        except FileNotFoundError:
            raise FileNotFoundError(f"Skeleton file not found: {file_path}")
        
        if skeleton.ndim != 2:
            raise ValueError(
                f"Expected 2D skeleton array, got shape {skeleton.shape}. "
                f"File: {file_path}"
            )
        
        sequence_len, feat_dim = skeleton.shape
        
        # Handle variable-length sequences
        if sequence_len >= self.sequence_length:
            # Randomly sample a contiguous window of length T
            sequence, mask = self._sample_window(skeleton)
        else:
            # Pad with zeros to reach target length T
            sequence, mask = self._pad_sequence(skeleton)
        
        # Convert to torch tensors
        sequence_tensor = torch.from_numpy(sequence).float()  # Shape: (T, D)
        mask_tensor = torch.from_numpy(mask).float()  # Shape: (T,)
        
        return sequence_tensor, mask_tensor, label
    
    def _sample_window(self, skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly sample a contiguous window from a long sequence.
        
        Args:
            skeleton: Shape (L, D) where L >= T
            
        Returns:
            Tuple of:
                - sampled_skeleton (np.ndarray): Shape (T, D)
                - mask (np.ndarray): Shape (T,) filled with ones
        """
        seq_len, feat_dim = skeleton.shape
        
        # Calculate random start position
        max_start = seq_len - self.sequence_length
        start_idx = np.random.randint(0, max_start + 1)
        
        # Extract window
        # Window: [start_idx, start_idx + T)
        sampled_skeleton = skeleton[start_idx:start_idx + self.sequence_length]
        
        # All timesteps are valid
        mask = np.ones(self.sequence_length, dtype=np.float32)
        
        return sampled_skeleton.astype(np.float32), mask
    
    def _pad_sequence(self, skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad a short sequence with zeros to reach target length.
        
        Args:
            skeleton: Shape (L, D) where L < T
            
        Returns:
            Tuple of:
                - padded_skeleton (np.ndarray): Shape (T, D)
                - mask (np.ndarray): Shape (T,) with 1s for valid, 0s for padded
        """
        seq_len, feat_dim = skeleton.shape
        
        # Create padded array
        padded_skeleton = np.zeros(
            (self.sequence_length, feat_dim),
            dtype=np.float32
        )
        
        # Copy original sequence
        padded_skeleton[:seq_len] = skeleton
        
        # Create mask: 1 for original frames, 0 for padding
        mask = np.zeros(self.sequence_length, dtype=np.float32)
        mask[:seq_len] = 1.0
        
        return padded_skeleton, mask
    
    def get_feature_dim(self) -> int:
        """
        Get feature dimension by loading first sample.
        
        Returns:
            Feature dimension D
        """
        skeleton = np.load(self.abs_file_list[0])
        return skeleton.shape[1]


def collate_fn_skeleton(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for skeleton dataset batches.
    
    Stacks sequences and masks into batch tensors. Handles variable sequence
    lengths by preserving padding information in masks.
    
    Args:
        batch: List of (sequence, mask, label) tuples from __getitem__
        
    Returns:
        Tuple containing:
            - sequences (torch.Tensor): Shape (B, T, D), batched sequences
            - masks (torch.Tensor): Shape (B, T), batched validity masks
            - labels (torch.Tensor): Shape (B,), batched labels
            
    Example:
        >>> dataloader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     collate_fn=collate_fn_skeleton
        ... )
        >>> sequences, masks, labels = next(iter(dataloader))
        >>> print(sequences.shape)  # (32, T, D)
    """
    sequences, masks, labels = zip(*batch)
    
    # Stack along new batch dimension
    sequences_batch = torch.stack(sequences, dim=0)  # (B, T, D)
    masks_batch = torch.stack(masks, dim=0)  # (B, T)
    labels_batch = torch.tensor(labels, dtype=torch.long)  # (B,)
    
    return sequences_batch, masks_batch, labels_batch
