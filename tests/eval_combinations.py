#!/usr/bin/env python3
"""
Comprehensive evaluation script to test multiple hyperparameter combinations
and gather the best evaluation results for the fall detection model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pose_estimation.dataset import SkeletonDataset, collate_fn_skeleton
from pose_estimation.model import SkeletonLSTM
from itertools import product
import time

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
def load_train_val_dataset(train_ratio=0.8):
    """Load dataset and split into train/val."""
    root_dir = "dataset/pose_npy"
    
    # Get all files
    from pathlib import Path
    root = Path(root_dir)
    all_files = sorted([p.name for p in root.glob("*.npy")])
    
    # Infer labels
    labels = [1 if "fall" in f.lower() or f.lower().startswith("f-") else 0 for f in all_files]
    
    # Create dataset
    dataset = SkeletonDataset(
        root_dir=str(root),
        file_paths=all_files,
        labels=labels,
        sequence_length=32
    )
    
    # Split
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    feature_dim = dataset.get_feature_dim()
    
    return train_dataset, val_dataset, feature_dim


def evaluate(model, dataloader, loss_fn, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, masks, labels in dataloader:
            sequences = sequences.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            logits = model(sequences, masks)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.shape[0]
    
    return total_loss / len(dataloader), correct / total


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
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
    
    return total_loss / len(dataloader)


# Load data
print("Loading dataset...")
train_dataset, val_dataset, feature_dim = load_train_val_dataset()

# Create dataloaders
train_dataloader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, 
    collate_fn=collate_fn_skeleton, num_workers=0
)
val_dataloader = DataLoader(
    val_dataset, batch_size=8, shuffle=False, 
    collate_fn=collate_fn_skeleton, num_workers=0
)

print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val")
print(f"Feature dimension: {feature_dim}")

# Hyperparameter combinations to test
hidden_sizes = [32, 64, 128]
learning_rates = [0.001, 0.0005]
num_layers_list = [1, 2]
dropout_values = [0.2, 0.5]

results = []

# Test each combination
combinations = list(product(hidden_sizes, learning_rates, num_layers_list, dropout_values))
print(f"\nTesting {len(combinations)} hyperparameter combinations...")
print("=" * 80)

for idx, (hidden_size, lr, num_layers, dropout) in enumerate(combinations, 1):
    print(f"\n[{idx}/{len(combinations)}] Testing: hidden={hidden_size}, lr={lr}, layers={num_layers}, dropout={dropout}")
    
    # Create model
    model = SkeletonLSTM(
        input_dim=feature_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=2,
        bidirectional=False
    ).to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    # Train for a few epochs (since training is fast)
    num_epochs = 15
    best_val_acc = 0.0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_dataloader, loss_fn, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    
    elapsed_time = time.time() - start_time
    
    # Final evaluation
    final_val_loss, final_val_acc = evaluate(model, val_dataloader, loss_fn, device)
    
    result = {
        'hidden_size': hidden_size,
        'learning_rate': lr,
        'num_layers': num_layers,
        'dropout': dropout,
        'best_val_accuracy': best_val_acc,
        'final_val_accuracy': final_val_acc,
        'final_val_loss': final_val_loss,
        'training_time': elapsed_time
    }
    results.append(result)
    
    print(f"  ✓ Best accuracy: {best_val_acc:.4f}, Final accuracy: {final_val_acc:.4f}, Time: {elapsed_time:.2f}s")

# Sort by best accuracy
results.sort(key=lambda x: x['best_val_accuracy'], reverse=True)

print("\n" + "=" * 80)
print("EVALUATION RESULTS (sorted by best validation accuracy)")
print("=" * 80)
print(f"{'Rank':<5} {'Hidden':<8} {'LR':<10} {'Layers':<8} {'Dropout':<10} {'Best Acc':<12} {'Final Acc':<12} {'Time(s)':<8}")
print("-" * 80)

for idx, result in enumerate(results[:10], 1):
    print(f"{idx:<5} {result['hidden_size']:<8} {result['learning_rate']:<10.5f} {result['num_layers']:<8} "
          f"{result['dropout']:<10.1f} {result['best_val_accuracy']:<12.4f} {result['final_val_accuracy']:<12.4f} "
          f"{result['training_time']:<8.2f}")

# Print the best result
best = results[0]
print("\n" + "=" * 80)
print("BEST MODEL CONFIGURATION")
print("=" * 80)
print(f"Hidden size: {best['hidden_size']}")
print(f"Learning rate: {best['learning_rate']}")
print(f"Number of layers: {best['num_layers']}")
print(f"Dropout: {best['dropout']}")
print(f"Best validation accuracy: {best['best_val_accuracy']:.4f} ({best['best_val_accuracy']*100:.2f}%)")
print(f"Final validation accuracy: {best['final_val_accuracy']:.4f} ({best['final_val_accuracy']*100:.2f}%)")
print(f"Validation loss: {best['final_val_loss']:.4f}")
print(f"Training time: {best['training_time']:.2f}s")
