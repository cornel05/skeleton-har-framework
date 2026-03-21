"""
Validation and unit tests for the skeleton action recognition pipeline.

Run with:
  python test_pipeline.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tempfile
import os
from pathlib import Path

from src.pose_estimation.dataset import SkeletonDataset, collate_fn_skeleton
from src.pose_estimation.model import SkeletonLSTM, SkeletonLSTMWithAttention
from src.pose_estimation.utils import (
    normalize_skeleton,
    train_val_test_split,
    compute_metrics,
    sensitivity_specificity
)


def create_dummy_skeleton(seq_len: int = 50, feat_dim: int = 34) -> np.ndarray:
    """Create a dummy skeleton array for testing."""
    return np.random.randn(seq_len, feat_dim).astype(np.float32)


class TestSkeletonDataset:
    """Tests for SkeletonDataset class."""
    
    def test_dataset_creation(self):
        """Test dataset can be created and loaded."""
        print("\n[TEST] Dataset Creation")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy skeleton files
            skeleton1 = create_dummy_skeleton(50, 34)
            skeleton2 = create_dummy_skeleton(30, 34)
            skeleton3 = create_dummy_skeleton(60, 34)
            
            np.save(os.path.join(tmpdir, "adl-01.npy"), skeleton1)
            np.save(os.path.join(tmpdir, "adl-02.npy"), skeleton2)
            np.save(os.path.join(tmpdir, "fall-01.npy"), skeleton3)
            
            # Create dataset
            file_paths = ["adl-01.npy", "adl-02.npy", "fall-01.npy"]
            labels = [0, 0, 1]
            
            dataset = SkeletonDataset(
                root_dir=tmpdir,
                file_paths=file_paths,
                labels=labels,
                sequence_length=32
            )
            
            # Test length
            assert len(dataset) == 3, "Dataset length should be 3"
            print("  ✓ Dataset length correct")
            
            # Test getitem
            sequence, mask, label = dataset[0]
            assert sequence.shape == (32, 34), f"Expected (32, 34), got {sequence.shape}"
            assert mask.shape == (32,), f"Expected (32,), got {mask.shape}"
            assert label in [0, 1], "Label should be 0 or 1"
            print("  ✓ Sample shapes correct")
            
            # Test long sequence (>T)
            sequence, mask, label = dataset[0]
            assert torch.sum(mask) <= 32, "Mask should have at most T ones"
            print("  ✓ Long sequence windowing works")
            
            # Test short sequence (<T)
            sequence, mask, label = dataset[1]
            valid_count = torch.sum(mask).item()
            assert valid_count <= 30, "Valid timesteps should be ≤ original length"
            assert valid_count > 0, "Valid timesteps should be > 0"
            print("  ✓ Short sequence padding works")
            
            # Test feature dimension
            feat_dim = dataset.get_feature_dim()
            assert feat_dim == 34, f"Feature dim should be 34, got {feat_dim}"
            print("  ✓ Feature dimension extraction works")
    
    def test_collate_function(self):
        """Test custom collate function."""
        print("\n[TEST] Collate Function")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            for i in range(3):
                skeleton = create_dummy_skeleton(40 + i*10, 34)
                np.save(os.path.join(tmpdir, f"test-{i:02d}.npy"), skeleton)
            
            dataset = SkeletonDataset(
                root_dir=tmpdir,
                file_paths=[f"test-{i:02d}.npy" for i in range(3)],
                labels=[0, 1, 0],
                sequence_length=32
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                collate_fn=collate_fn_skeleton
            )
            
            sequences, masks, labels = next(iter(dataloader))
            
            assert sequences.shape == (2, 32, 34), f"Expected (2, 32, 34), got {sequences.shape}"
            assert masks.shape == (2, 32), f"Expected (2, 32), got {masks.shape}"
            assert labels.shape == (2,), f"Expected (2,), got {labels.shape}"
            print("  ✓ Batch shapes correct")
            
            assert sequences.dtype == torch.float32, "Sequences should be float32"
            assert masks.dtype == torch.float32, "Masks should be float32"
            assert labels.dtype == torch.long, "Labels should be long"
            print("  ✓ Data types correct")


class TestSkeletonLSTM:
    """Tests for SkeletonLSTM model."""
    
    def test_model_creation(self):
        """Test model can be created."""
        print("\n[TEST] Model Creation")
        
        model = SkeletonLSTM(
            input_dim=34,
            hidden_size=64,
            num_layers=1,
            dropout=0.5,
            num_classes=2
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0, "Model should have parameters"
        print(f"  ✓ Model created with {num_params:,} parameters")
    
    def test_forward_pass(self):
        """Test forward pass through model."""
        print("\n[TEST] Forward Pass")
        
        model = SkeletonLSTM(
            input_dim=34,
            hidden_size=64,
            num_classes=2
        )
        
        # Create dummy batch
        batch_size = 4
        seq_length = 32
        feat_dim = 34
        
        sequences = torch.randn(batch_size, seq_length, feat_dim)
        masks = torch.ones(batch_size, seq_length)
        
        # Forward pass
        logits = model(sequences, masks)
        
        assert logits.shape == (batch_size, 2), f"Expected shape (4, 2), got {logits.shape}"
        assert not torch.isnan(logits).any(), "Logits should not contain NaN"
        print("  ✓ Forward pass produces correct output shape")
    
    def test_variable_length_sequences(self):
        """Test model handles variable-length sequences."""
        print("\n[TEST] Variable-Length Sequences")
        
        model = SkeletonLSTM(
            input_dim=34,
            hidden_size=64,
            num_classes=2
        )
        
        # Create batch with different padding amounts
        batch_size = 4
        seq_length = 32
        feat_dim = 34
        
        sequences = torch.randn(batch_size, seq_length, feat_dim)
        
        # Different valid lengths for each sequence
        masks = torch.ones(batch_size, seq_length)
        masks[0, 20:] = 0  # 20 valid timesteps
        masks[1, 25:] = 0  # 25 valid timesteps
        masks[2, 10:] = 0  # 10 valid timesteps
        # Sequence 3 has all valid
        
        logits = model(sequences, masks)
        
        assert logits.shape == (batch_size, 2), "Output shape should match batch size"
        print("  ✓ Variable-length sequences handled correctly")
    
    def test_predict_method(self):
        """Test predict method."""
        print("\n[TEST] Predict Method")
        
        model = SkeletonLSTM(
            input_dim=34,
            hidden_size=64,
            num_classes=2
        )
        
        sequences = torch.randn(4, 32, 34)
        masks = torch.ones(4, 32)
        
        predictions, probabilities = model.predict(sequences, masks)
        
        assert predictions.shape == (4,), "Predictions should have shape (4,)"
        assert probabilities.shape == (4, 2), "Probabilities should have shape (4, 2)"
        assert torch.all((probabilities >= 0) & (probabilities <= 1)), "Probabilities should be in [0,1]"
        assert torch.allclose(torch.sum(probabilities, dim=1), torch.ones(4)), "Probabilities should sum to 1"
        print("  ✓ Predict method works correctly")
    
    def test_bidirectional_lstm(self):
        """Test bidirectional LSTM."""
        print("\n[TEST] Bidirectional LSTM")
        
        model = SkeletonLSTM(
            input_dim=34,
            hidden_size=64,
            bidirectional=True
        )
        
        sequences = torch.randn(4, 32, 34)
        masks = torch.ones(4, 32)
        
        logits = model(sequences, masks)
        
        assert logits.shape == (4, 2), "Output shape should be (4, 2)"
        print("  ✓ Bidirectional LSTM works correctly")
    
    def test_multi_layer_lstm(self):
        """Test multi-layer LSTM."""
        print("\n[TEST] Multi-Layer LSTM")
        
        model = SkeletonLSTM(
            input_dim=34,
            hidden_size=64,
            num_layers=2,
            dropout=0.3
        )
        
        sequences = torch.randn(4, 32, 34)
        masks = torch.ones(4, 32)
        
        logits = model(sequences, masks)
        
        assert logits.shape == (4, 2), "Output shape should be (4, 2)"
        print("  ✓ Multi-layer LSTM works correctly")


class TestAttentionModel:
    """Tests for attention-based LSTM."""
    
    def test_attention_lstm(self):
        """Test SkeletonLSTMWithAttention."""
        print("\n[TEST] Attention-Based LSTM")
        
        model = SkeletonLSTMWithAttention(
            input_dim=34,
            hidden_size=64,
            attention_context=5,
            num_classes=2
        )
        
        sequences = torch.randn(4, 32, 34)
        masks = torch.ones(4, 32)
        
        logits = model(sequences, masks)
        
        assert logits.shape == (4, 2), "Output shape should be (4, 2)"
        print("  ✓ Attention-based LSTM works correctly")


class TestUtils:
    """Tests for utility functions."""
    
    def test_normalization(self):
        """Test data normalization."""
        print("\n[TEST] Data Normalization")
        
        skeleton = create_dummy_skeleton(50, 34)
        
        # Test minmax normalization
        normalized = normalize_skeleton(skeleton, method="minmax")
        assert normalized.min() >= -0.01, "MinMax should have min ≥ 0"
        assert normalized.max() <= 1.01, "MinMax should have max ≤ 1"
        print("  ✓ MinMax normalization works")
        
        # Test zscore normalization
        normalized = normalize_skeleton(skeleton, method="zscore")
        assert abs(normalized.mean()) < 0.1, "ZScore mean should be ~0"
        assert abs(normalized.std() - 1.0) < 0.1, "ZScore std should be ~1"
        print("  ✓ ZScore normalization works")
        
        # Test robust normalization
        normalized = normalize_skeleton(skeleton, method="robust")
        assert normalized.shape == skeleton.shape, "Shape should be preserved"
        print("  ✓ Robust normalization works")
    
    def test_train_val_test_split(self):
        """Test data splitting."""
        print("\n[TEST] Train/Val/Test Split")
        
        file_list = [f"file-{i:02d}.npy" for i in range(20)]
        labels = [0] * 10 + [1] * 10
        
        (train_files, train_labels), \
        (val_files, val_labels), \
        (test_files, test_labels) = train_val_test_split(
            file_list, labels,
            train_ratio=0.7,
            val_ratio=0.15
        )
        
        total = len(train_files) + len(val_files) + len(test_files)
        assert total == 20, "All samples should be used"
        
        train_ratio = len(train_files) / 20
        val_ratio = len(val_files) / 20
        assert 0.65 < train_ratio < 0.75, "Train ratio should be ~70%"
        assert 0.10 < val_ratio < 0.20, "Val ratio should be ~15%"
        print("  ✓ Split ratios correct")
        
        # Check stratification
        train_class_balance = sum(train_labels) / len(train_labels)
        original_class_balance = sum(labels) / len(labels)
        assert abs(train_class_balance - original_class_balance) < 0.15, "Class distribution should be preserved"
        print("  ✓ Stratification preserves class distribution")
    
    def test_compute_metrics(self):
        """Test metric computation."""
        print("\n[TEST] Metrics Computation")
        
        predictions = np.array([0, 1, 1, 0, 1, 0])
        ground_truth = np.array([0, 1, 0, 0, 1, 1])
        probabilities = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.4, 0.6]
        ])
        
        metrics = compute_metrics(predictions, probabilities, ground_truth)
        
        assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be in [0, 1]"
        assert 0 <= metrics['precision'] <= 1, "Precision should be in [0, 1]"
        assert 0 <= metrics['recall'] <= 1, "Recall should be in [0, 1]"
        assert 0 <= metrics['f1'] <= 1, "F1 should be in [0, 1]"
        assert 0 <= metrics['roc_auc'] <= 1, "ROC-AUC should be in [0, 1]"
        print(f"  ✓ Metrics computed: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
    
    def test_sensitivity_specificity(self):
        """Test sensitivity and specificity."""
        print("\n[TEST] Sensitivity/Specificity")
        
        predictions = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        ground_truth = np.array([0, 1, 0, 0, 1, 1, 1, 0])
        
        sensitivity, specificity = sensitivity_specificity(predictions, ground_truth)
        
        assert 0 <= sensitivity <= 1, "Sensitivity should be in [0, 1]"
        assert 0 <= specificity <= 1, "Specificity should be in [0, 1]"
        print(f"  ✓ Computed: Sensitivity={sensitivity:.3f}, Specificity={specificity:.3f}")


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self):
        """Test complete pipeline."""
        print("\n[TEST] End-to-End Pipeline")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy data
            for i in range(8):
                skeleton = create_dummy_skeleton(30 + i*5, 34)
                label = "adl" if i < 4 else "fall"
                np.save(os.path.join(tmpdir, f"{label}-{i:02d}.npy"), skeleton)
            
            # Create dataset
            file_paths = [f"adl-{i:02d}.npy" for i in range(4)] + \
                        [f"fall-{i:02d}.npy" for i in range(4)]
            labels = [0, 0, 0, 0, 1, 1, 1, 1]
            
            dataset = SkeletonDataset(
                root_dir=tmpdir,
                file_paths=file_paths,
                labels=labels,
                sequence_length=32
            )
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=4,
                collate_fn=collate_fn_skeleton
            )
            
            # Create model
            model = SkeletonLSTM(
                input_dim=34,
                hidden_size=64,
                num_classes=2
            )
            
            # Test forward pass
            sequences, masks, batch_labels = next(iter(dataloader))
            logits = model(sequences, masks)
            
            # Test loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, batch_labels)
            
            assert not torch.isnan(loss), "Loss should not be NaN"
            print(f"  ✓ Full pipeline works (loss={loss:.4f})")
            
            # Test backward pass
            loss.backward()
            print("  ✓ Backward pass successful")
            
            # Check gradients
            has_gradients = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.parameters()
            )
            assert has_gradients, "Model should have non-zero gradients"
            print("  ✓ Gradients computed correctly")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("SKELETON ACTION RECOGNITION PIPELINE - TEST SUITE")
    print("="*70)
    
    try:
        # Dataset tests
        TestSkeletonDataset().test_dataset_creation()
        TestSkeletonDataset().test_collate_function()
        
        # Model tests
        TestSkeletonLSTM().test_model_creation()
        TestSkeletonLSTM().test_forward_pass()
        TestSkeletonLSTM().test_variable_length_sequences()
        TestSkeletonLSTM().test_predict_method()
        TestSkeletonLSTM().test_bidirectional_lstm()
        TestSkeletonLSTM().test_multi_layer_lstm()
        
        # Attention model tests
        TestAttentionModel().test_attention_lstm()
        
        # Utils tests
        TestUtils().test_normalization()
        TestUtils().test_train_val_test_split()
        TestUtils().test_compute_metrics()
        TestUtils().test_sensitivity_specificity()
        
        # End-to-end tests
        TestEndToEnd().test_full_pipeline()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
