"""
LSTM-based skeleton action recognition model.

Implements sequence-to-classification architecture with masking support for
variable-length skeleton sequences.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional

try:
    from .config import MODEL_CFG
except ImportError:
    MODEL_CFG = {}


class SkeletonLSTM(nn.Module):
    """
    LSTM model for skeleton-based action recognition with masking.
    
    Architecture:
        - LSTM layer: Processes sequences with batch_first=True
        - Attention pooling: Extracts the last valid timestep using mask
        - Classification head: Binary classifier (fall vs non-fall)
    
    Key features:
        - Handles variable-length sequences via batch packing
        - Uses validity masks to extract last valid output
        - Dropout for regularization
    
    Attributes:
        input_dim (int): Feature dimension D of skeleton keypoints
        hidden_size (int): LSTM hidden state dimension
        num_layers (int): Number of stacked LSTM layers
        dropout (float): Dropout probability
        num_classes (int): Number of output classes
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = MODEL_CFG.get("hidden_size", 64),
        num_layers: int = MODEL_CFG.get("num_layers", 1),
        dropout: float = MODEL_CFG.get("dropout", 0.5),
        num_classes: int = MODEL_CFG.get("num_classes", 2),
        bidirectional: bool = MODEL_CFG.get("bidirectional", False)
    ):
        """
        Initialize the skeleton LSTM model.
        
        Args:
            input_dim: Feature dimension D (e.g., 34 for COCO keypoints)
            hidden_size: LSTM hidden state dimension (default: 64)
            num_layers: Number of stacked LSTM layers (default: 1)
            dropout: Dropout rate applied to LSTM outputs (default: 0.5)
            num_classes: Number of classification classes (default: 2 for binary)
            bidirectional: Whether to use bidirectional LSTM (default: False)
            
        Example:
            >>> model = SkeletonLSTM(
            ...     input_dim=34,
            ...     hidden_size=64,
            ...     num_classes=2
            ... )
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # Compute actual output size considering bidirectionality
        self.lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # LSTM layer with batch_first=True
        # Input shape: (B, T, D)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Classification head
        # Input: last valid LSTM output (lstm_output_size,)
        # Output: class logits (num_classes,)
        self.classifier = nn.Linear(self.lstm_output_size, num_classes)
    
    def forward(
        self,
        sequences: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            sequences (torch.Tensor): Shape (B, T, D)
                - B: batch size
                - T: sequence length
                - D: feature dimension
            
            masks (torch.Tensor): Shape (B, T)
                - Binary validity mask (1 for valid, 0 for padded)
                - Used to identify last valid timestep
        
        Returns:
            logits (torch.Tensor): Shape (B, num_classes)
                - Raw output logits for classification
                - Apply softmax for probabilities or cross-entropy loss
        
        Raises:
            AssertionError: If masks contain all zeros (no valid timesteps)
        
        Example:
            >>> sequences = torch.randn(32, 20, 34)  # (B, T, D)
            >>> masks = torch.ones(32, 20)  # (B, T)
            >>> logits = model(sequences, masks)
            >>> print(logits.shape)
            torch.Size([32, 2])
        """
        # Compute sequence lengths from masks
        # Shape: (B,) with values in range [1, T]
        seq_lengths = self._get_lengths_from_mask(masks)
        
        # Pack padded sequences for efficient LSTM computation
        # This prevents LSTM from processing padding tokens
        packed_sequences = pack_padded_sequence(
            sequences,
            seq_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # LSTM forward pass
        # packed_output contains concatenated outputs for non-padded timesteps
        # h_n shape: (num_layers * num_directions, B, hidden_size)
        # c_n shape: (num_layers * num_directions, B, hidden_size)
        packed_output, (h_n, c_n) = self.lstm(packed_sequences)
        
        # Unpack sequences back to (B, T, lstm_output_size)
        # Padded positions will have zero tensors
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True
        )
        
        # Extract last valid timestep output for each sequence
        # Shape: (B, lstm_output_size)
        last_valid_output = self._extract_last_valid_output(output, masks)
        
        # Classification layer
        # Shape: (B, num_classes)
        logits = self.classifier(last_valid_output)
        
        return logits
    
    def _get_lengths_from_mask(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute actual sequence lengths from validity masks.
        
        For a mask like [1, 1, 1, 0, 0], extracts length=3.
        Ensures at least length 1 to avoid LSTM errors.
        
        Args:
            masks (torch.Tensor): Shape (B, T), binary validity mask
        
        Returns:
            lengths (torch.Tensor): Shape (B,), sequence lengths
        
        Raises:
            AssertionError: If any sequence has all zeros (no valid timesteps)
        """
        # Sum mask along time dimension: (B, T) -> (B,)
        lengths = torch.sum(masks, dim=1).long()
        
        # Ensure minimum length of 1 to avoid LSTM errors
        lengths = torch.clamp(lengths, min=1)
        
        # Debug: ensure no sequences are completely masked out
        assert torch.all(lengths > 0), \
            "All sequences must have at least one valid timestep"
        
        return lengths
    
    def _extract_last_valid_output(
        self,
        output: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract the output at the last valid timestep for each sequence.
        
        Instead of using the final timestep output (which may be padding),
        this uses the mask to find the true last valid frame.
        
        For example:
            - Sequence: [out0, out1, out2, pad, pad] with mask [1, 1, 1, 0, 0]
            - Returns: out2 (not the 4th element which is padding)
        
        Args:
            output (torch.Tensor): Shape (B, T, lstm_output_size)
                LSTM output for all timesteps
            masks (torch.Tensor): Shape (B, T)
                Binary validity mask
        
        Returns:
            last_valid (torch.Tensor): Shape (B, lstm_output_size)
                LSTM output at the last valid timestep for each sequence
        """
        batch_size = output.shape[0]
        
        # Find last valid index for each sequence
        # mask * (indices) -> (B, T) with non-zero values only for valid indices
        # argmax along T dimension gives last non-zero position
        # Shape after argmax: (B,)
        last_valid_indices = torch.argmax(
            masks * torch.arange(masks.shape[1], device=masks.device).float(),
            dim=1
        )
        
        # Create batch indices
        batch_indices = torch.arange(batch_size, device=output.device)
        
        # Index into output: output[batch_idx, time_idx, :]
        # This gives shape (B, lstm_output_size)
        last_valid_output = output[batch_indices, last_valid_indices]
        
        return last_valid_output
    
    def predict(
        self,
        sequences: torch.Tensor,
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get class predictions and confidence scores.
        
        Args:
            sequences (torch.Tensor): Shape (B, T, D)
            masks (torch.Tensor): Shape (B, T)
        
        Returns:
            Tuple containing:
                - predictions (torch.Tensor): Shape (B,), predicted class indices
                - probabilities (torch.Tensor): Shape (B, num_classes), softmax probabilities
        
        Example:
            >>> predictions, probs = model.predict(sequences, masks)
            >>> print(predictions)  # [0, 1, 0, ...]
            >>> print(probs[:, 1])  # Probability of fall class
        """
        with torch.no_grad():
            logits = self.forward(sequences, masks)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        return predictions, probabilities


class SkeletonLSTMWithAttention(SkeletonLSTM):
    """
    LSTM model with attention-based pooling over last timesteps.
    
    Instead of using only the last valid output, this variant computes
    a weighted average of the last N timesteps using attention weights.
    
    Useful for action recognition where the decision boundary might involve
    multiple frames rather than just the terminal frame.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = MODEL_CFG.get("hidden_size", 64),
        num_layers: int = MODEL_CFG.get("num_layers", 1),
        dropout: float = MODEL_CFG.get("dropout", 0.5),
        num_classes: int = MODEL_CFG.get("num_classes", 2),
        bidirectional: bool = MODEL_CFG.get("bidirectional", False),
        attention_context: int = MODEL_CFG.get("attention_context", 5)
    ):
        """
        Initialize attention-based LSTM model.
        
        Args:
            attention_context: Number of last timesteps to attend over
        """
        super().__init__(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
            bidirectional=bidirectional
        )
        
        self.attention_context = attention_context
        
        # Attention scoring network
        self.attention = nn.Sequential(
            nn.Linear(self.lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(
        self,
        sequences: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with attention-based aggregation.
        
        Args:
            sequences (torch.Tensor): Shape (B, T, D)
            masks (torch.Tensor): Shape (B, T)
        
        Returns:
            logits (torch.Tensor): Shape (B, num_classes)
        """
        # Compute sequence lengths from masks
        seq_lengths = self._get_lengths_from_mask(masks)
        
        # Pack and forward through LSTM
        packed_sequences = pack_padded_sequence(
            sequences,
            seq_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        packed_output, _ = self.lstm(packed_sequences)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply attention over last context frames
        attended_output = self._attention_pooling(output, masks)
        
        # Classification
        logits = self.classifier(attended_output)
        
        return logits
    
    def _attention_pooling(
        self,
        output: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention-based pooling over the last context frames.
        
        Args:
            output (torch.Tensor): Shape (B, T, lstm_output_size)
            masks (torch.Tensor): Shape (B, T)
        
        Returns:
            attended (torch.Tensor): Shape (B, lstm_output_size)
        """
        batch_size = output.shape[0]
        seq_len = output.shape[1]
        
        # Get last valid index for each sequence
        last_valid_indices = torch.argmax(
            masks * torch.arange(seq_len, device=masks.device).float(),
            dim=1
        )
        
        # Extract context window: [max(0, last - context), last]
        attended_outputs = []
        
        for b in range(batch_size):
            last_idx = last_valid_indices[b].item()
            start_idx = max(0, last_idx - self.attention_context + 1)
            
            # Extract context window
            context = output[b, start_idx:last_idx + 1, :]  # (context_len, lstm_output_size)
            
            # Compute attention scores
            scores = self.attention(context)  # (context_len, 1)
            scores = torch.softmax(scores, dim=0)
            
            # Weighted average
            attended = torch.sum(context * scores, dim=0)  # (lstm_output_size,)
            attended_outputs.append(attended)
        
        # Stack: (B, lstm_output_size)
        attended_output = torch.stack(attended_outputs, dim=0)
        
        return attended_output


class SkeletonGRU(nn.Module):
    """GRU counterpart of SkeletonLSTM (same hidden size / layers, last valid output -> linear)."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = MODEL_CFG.get("hidden_size", 64),
        num_layers: int = MODEL_CFG.get("num_layers", 1),
        dropout: float = MODEL_CFG.get("dropout", 0.2),
        num_classes: int = MODEL_CFG.get("num_classes", 2),
        bidirectional: bool = MODEL_CFG.get("bidirectional", False),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        out = hidden_size * (2 if bidirectional else 1)
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        self.classifier = nn.Linear(out, num_classes)

    def forward(self, sequences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        lengths = torch.clamp(masks.sum(dim=1).long(), min=1)
        packed = pack_padded_sequence(sequences, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = pad_packed_sequence(self.gru(packed)[0], batch_first=True)
        last = output[torch.arange(output.shape[0], device=output.device), lengths - 1]
        return self.classifier(last)

    def predict(self, sequences, masks):
        with torch.no_grad():
            probs = torch.softmax(self.forward(sequences, masks), dim=1)
        return probs.argmax(dim=1), probs


class _TemporalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation)
        self.norm = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return torch.relu(self.skip(x) + self.drop(torch.relu(self.norm(self.conv(x)))))


class SkeletonTCN(nn.Module):
    """Small non-causal dilated 1D-CNN over time (receptive field 31 frames), masked mean pooling."""

    def __init__(self, input_dim: int, channels: int = 48, kernel: int = 3,
                 dilations=(1, 2, 4, 8), dropout: float = 0.2, num_classes: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        layers, c = [], input_dim
        for d in dilations:
            layers.append(_TemporalBlock(c, channels, kernel, d, dropout))
            c = channels
        self.net = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, sequences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        h = self.net((sequences * masks.unsqueeze(-1)).transpose(1, 2))  # (B, C, T)
        m = masks.unsqueeze(1)
        pooled = (h * m).sum(dim=2) / m.sum(dim=2).clamp(min=1.0)
        return self.classifier(pooled)

    def predict(self, sequences, masks):
        with torch.no_grad():
            probs = torch.softmax(self.forward(sequences, masks), dim=1)
        return probs.argmax(dim=1), probs


COCO17_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
                (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


class _STGCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, A: torch.Tensor, t_kernel: int, dropout: float):
        super().__init__()
        self.register_buffer("A", A)
        self.edge_importance = nn.Parameter(torch.ones_like(A))
        self.gcn = nn.Conv2d(c_in, c_out, 1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(c_out), nn.ReLU(),
            nn.Conv2d(c_out, c_out, (t_kernel, 1), padding=((t_kernel - 1) // 2, 0)),
            nn.BatchNorm2d(c_out), nn.Dropout(dropout),
        )
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):  # x: (B, C, T, V)
        g = torch.einsum("bctv,vw->bctw", self.gcn(x), self.A * self.edge_importance)
        return torch.relu(self.tcn(g) + self.skip(x))


class SkeletonSTGCN(nn.Module):
    """Small ST-GCN (single-partition normalized COCO-17 graph, 3 blocks)."""

    def __init__(self, input_dim: int = 34, channels=(32, 32, 64), t_kernel: int = 9,
                 dropout: float = 0.2, num_classes: int = 2, num_joints: int = 17):
        super().__init__()
        assert input_dim == num_joints * 2, "ST-GCN expects (x, y) per COCO-17 joint"
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_joints = num_joints
        A = torch.eye(num_joints)
        for i, j in COCO17_EDGES:
            A[i, j] = A[j, i] = 1.0
        d = A.sum(dim=1).pow(-0.5)
        A = d[:, None] * A * d[None, :]
        self.data_bn = nn.BatchNorm1d(input_dim)
        blocks, c = [], 2
        for c_out in channels:
            blocks.append(_STGCNBlock(c, c_out, A.clone(), t_kernel, dropout))
            c = c_out
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(c, num_classes)

    def forward(self, sequences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        B, T, _ = sequences.shape
        x = self.data_bn((sequences * masks.unsqueeze(-1)).transpose(1, 2)).transpose(1, 2)
        x = x.reshape(B, T, self.num_joints, 2).permute(0, 3, 1, 2)  # (B, 2, T, V)
        h = self.blocks(x).mean(dim=3)                                # (B, C, T)
        m = masks.unsqueeze(1)
        pooled = (h * m).sum(dim=2) / m.sum(dim=2).clamp(min=1.0)
        return self.classifier(pooled)

    def predict(self, sequences, masks):
        with torch.no_grad():
            probs = torch.softmax(self.forward(sequences, masks), dim=1)
        return probs.argmax(dim=1), probs


MODEL_NAMES = ("lstm", "lstm_attention", "gru", "tcn", "stgcn")


def build_model(name: str, input_dim: int = 34, cfg: Optional[dict] = None) -> nn.Module:
    """Factory used by the evaluation protocol. Recurrent models use the repo config sizes."""
    cfg = dict(MODEL_CFG if cfg is None else cfg)
    rnn = dict(hidden_size=cfg.get("hidden_size", 64), num_layers=cfg.get("num_layers", 1),
               dropout=cfg.get("dropout", 0.2), num_classes=cfg.get("num_classes", 2),
               bidirectional=cfg.get("bidirectional", False))
    if name == "lstm":
        return SkeletonLSTM(input_dim=input_dim, **rnn)
    if name == "lstm_attention":
        return SkeletonLSTMWithAttention(input_dim=input_dim, attention_context=cfg.get("attention_context", 5), **rnn)
    if name == "gru":
        return SkeletonGRU(input_dim=input_dim, **rnn)
    if name == "tcn":
        return SkeletonTCN(input_dim=input_dim, dropout=cfg.get("dropout", 0.2), num_classes=rnn["num_classes"])
    if name == "stgcn":
        return SkeletonSTGCN(input_dim=input_dim, dropout=cfg.get("dropout", 0.2), num_classes=rnn["num_classes"])
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")


class FullWindowExport(nn.Module):
    """
    ONNX-friendly wrapper for full-length windows (mask of ones), which is the only case
    the sliding-window inference produces. Avoids pack_padded_sequence during export.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        if isinstance(m, SkeletonLSTMWithAttention):
            out, _ = m.lstm(x)
            ctx = out[:, -m.attention_context:, :]
            w = torch.softmax(m.attention(ctx), dim=1)
            return torch.softmax(m.classifier((ctx * w).sum(dim=1)), dim=1)
        if isinstance(m, SkeletonLSTM):
            return torch.softmax(m.classifier(m.lstm(x)[0][:, -1]), dim=1)
        if isinstance(m, SkeletonGRU):
            return torch.softmax(m.classifier(m.gru(x)[0][:, -1]), dim=1)
        ones = torch.ones(x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)
        return torch.softmax(m(x, ones), dim=1)
