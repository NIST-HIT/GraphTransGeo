#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Test script for advanced training strategies

import os
import torch
import numpy as np
import random
from lib.training.advanced_strategies import (
    HaversineLoss,
    CombinedLoss,
    EarlyStopping,
    ModelCheckpoint,
    AdversarialTrainer
)
from lib.model.improved_architecture import ImprovedGraphTransGeoGCN

def test_haversine_loss():
    """Test the Haversine loss function"""
    print("Testing Haversine Loss...")
    
    # Create random coordinates
    batch_size = 10
    pred = torch.rand(batch_size, 2) * 180 - 90  # Random coordinates in [-90, 90]
    target = torch.rand(batch_size, 2) * 180 - 90
    
    # Create Haversine loss function
    loss_fn = HaversineLoss()
    
    # Calculate loss
    loss = loss_fn(pred, target)
    
    print(f"  Predicted coordinates shape: {pred.shape}")
    print(f"  Target coordinates shape: {target.shape}")
    print(f"  Loss: {loss.item():.4f}")

def test_combined_loss():
    """Test the combined loss function"""
    print("\nTesting Combined Loss...")
    
    # Create random coordinates
    batch_size = 10
    pred = torch.rand(batch_size, 2) * 180 - 90
    target = torch.rand(batch_size, 2) * 180 - 90
    adv_loss = torch.tensor(0.1)
    
    # Create combined loss function
    loss_fn = CombinedLoss(hav_weight=0.3, adv_weight=0.01)
    
    # Calculate loss
    loss, components = loss_fn(pred, target, adv_loss)
    
    print(f"  Predicted coordinates shape: {pred.shape}")
    print(f"  Target coordinates shape: {target.shape}")
    print(f"  Combined loss: {loss.item():.4f}")
    print(f"  MSE component: {components['mse']:.4f}")
    print(f"  Haversine component: {components['haversine']:.4f}")
    print(f"  Adversarial component: {components['adversarial']:.4f}")

def test_early_stopping():
    """Test the early stopping handler"""
    print("\nTesting Early Stopping...")
    
    # Create early stopping handler
    early_stopping = EarlyStopping(patience=3, min_epochs=5, delta=0.001)
    
    # Simulate training
    val_losses = [0.5, 0.4, 0.3, 0.29, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33]
    
    for epoch, val_loss in enumerate(val_losses):
        should_stop, is_best = early_stopping(epoch, val_loss)
        
        print(f"  Epoch {epoch}, Val Loss: {val_loss:.4f}, Is Best: {is_best}, Should Stop: {should_stop}")
        
        if should_stop:
            print(f"  Early stopping at epoch {epoch}")
            break

def test_model_checkpoint(tmp_dir):
    """Test the model checkpoint handler"""
    print("\nTesting Model Checkpoint...")
    
    # Create model
    model = ImprovedGraphTransGeoGCN(input_dim=30, hidden_dim=64, output_dim=2)
    
    # Create model checkpoint handler
    checkpoint = ModelCheckpoint(save_dir=tmp_dir, model_name="test_model", verbose=True)
    
    # Save checkpoints
    checkpoint(model, epoch=1, is_best=False, val_loss=0.5)
    checkpoint(model, epoch=2, is_best=True, val_loss=0.4)
    
    # Check if files exist
    latest_path = os.path.join(tmp_dir, "test_model_latest.pth")
    epoch1_path = os.path.join(tmp_dir, "test_model_epoch_1.pth")
    epoch2_path = os.path.join(tmp_dir, "test_model_epoch_2.pth")
    best_path = os.path.join(tmp_dir, "test_model_best.pth")
    
    print(f"  Latest checkpoint exists: {os.path.exists(latest_path)}")
    print(f"  Epoch 1 checkpoint exists: {os.path.exists(epoch1_path)}")
    print(f"  Epoch 2 checkpoint exists: {os.path.exists(epoch2_path)}")
    print(f"  Best checkpoint exists: {os.path.exists(best_path)}")

def test_adversarial_trainer():
    """Test the adversarial trainer"""
    print("\nTesting Adversarial Trainer...")
    
    # Create model
    model = ImprovedGraphTransGeoGCN(input_dim=30, hidden_dim=64, output_dim=2)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create loss function
    loss_fn = CombinedLoss(hav_weight=0.3, adv_weight=0.01)
    
    # Create adversarial trainer
    trainer = AdversarialTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=torch.device("cpu"),
        epsilon=0.01,
        alpha=0.01,
        beta=0.5,
        clip_grad=1.0
    )
    
    print(f"  Trainer created successfully")
    print(f"  Epsilon: {trainer.epsilon}")
    print(f"  Alpha: {trainer.alpha}")
    print(f"  Beta: {trainer.beta}")
    print(f"  Clip grad: {trainer.clip_grad}")
    print(f"  Learning rate: {trainer.optimizer.param_groups[0]['lr']}")

def main():
    """Main function to run all tests"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Create temporary directory for model checkpoints
    tmp_dir = "tmp_checkpoints"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Run tests
    test_haversine_loss()
    test_combined_loss()
    test_early_stopping()
    test_model_checkpoint(tmp_dir)
    test_adversarial_trainer()
    
    # Clean up
    for file in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, file))
    os.rmdir(tmp_dir)
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()
