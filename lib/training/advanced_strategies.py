#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Advanced training strategies for GraphTransGeo++

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import math
import os
import time
import logging

class HaversineLoss(nn.Module):
    """
    Haversine distance loss function for geographic coordinates
    """
    def __init__(self, radius=6371.0):
        super(HaversineLoss, self).__init__()
        self.radius = radius  # Earth radius in kilometers
    
    def forward(self, pred, target):
        """
        Calculate Haversine distance between predicted and target coordinates
        
        Args:
            pred: Predicted coordinates [batch_size, 2] (latitude, longitude)
            target: Target coordinates [batch_size, 2] (latitude, longitude)
            
        Returns:
            loss: Mean Haversine distance [1]
        """
        # Convert to radians
        pred_lat, pred_lon = pred[:, 0] * math.pi / 180.0, pred[:, 1] * math.pi / 180.0
        target_lat, target_lon = target[:, 0] * math.pi / 180.0, target[:, 1] * math.pi / 180.0
        
        # Haversine formula
        dlon = target_lon - pred_lon
        dlat = target_lat - pred_lat
        
        a = torch.sin(dlat / 2.0) ** 2 + torch.cos(pred_lat) * torch.cos(target_lat) * torch.sin(dlon / 2.0) ** 2
        c = 2 * torch.asin(torch.sqrt(a))
        distance = self.radius * c  # Distance in kilometers
        
        # Mean distance as loss
        return torch.mean(distance)

class CombinedLoss(nn.Module):
    """
    Combined loss function with MSE, Haversine, and adversarial components
    """
    def __init__(self, hav_weight=0.3, adv_weight=0.01):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.haversine_loss = HaversineLoss()
        self.hav_weight = hav_weight
        self.adv_weight = adv_weight
    
    def forward(self, pred, target, adv_loss=None):
        """
        Calculate combined loss
        
        Args:
            pred: Predicted coordinates [batch_size, 2]
            target: Target coordinates [batch_size, 2]
            adv_loss: Adversarial loss component (optional)
            
        Returns:
            loss: Combined loss [1]
            components: Dictionary of loss components
        """
        # Calculate MSE loss
        mse = self.mse_loss(pred, target)
        
        # Calculate Haversine loss
        hav = self.haversine_loss(pred, target)
        
        # Combine losses
        combined = (1 - self.hav_weight) * mse + self.hav_weight * hav
        
        # Add adversarial loss if provided
        if adv_loss is not None:
            combined = combined + self.adv_weight * adv_loss
        
        # Return loss and components
        components = {
            'mse': mse.item(),
            'haversine': hav.item(),
            'adversarial': adv_loss.item() if adv_loss is not None else 0.0,
            'combined': combined.item()
        }
        
        return combined, components

class EarlyStopping:
    """
    Early stopping handler for training
    """
    def __init__(self, patience=10, min_epochs=30, delta=0.0001):
        self.patience = patience
        self.min_epochs = min_epochs
        self.delta = delta
        self.best_val_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False
    
    def __call__(self, epoch, val_loss):
        """
        Check if training should stop
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            
        Returns:
            should_stop: Whether training should stop
            is_best: Whether current model is the best so far
        """
        is_best = False
        
        # Check if validation loss improved
        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            is_best = True
        else:
            self.counter += 1
        
        # Check if should stop
        if self.counter >= self.patience and epoch >= self.min_epochs:
            self.should_stop = True
        
        return self.should_stop, is_best

class ModelCheckpoint:
    """
    Model checkpoint handler for saving best models
    """
    def __init__(self, save_dir, model_name, verbose=True):
        self.save_dir = save_dir
        self.model_name = model_name
        self.verbose = verbose
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def __call__(self, model, epoch, is_best=False, val_loss=None):
        """
        Save model checkpoint
        
        Args:
            model: Model to save
            epoch: Current epoch
            is_best: Whether current model is the best so far
            val_loss: Validation loss (optional)
        """
        # Save latest model
        latest_path = os.path.join(self.save_dir, f"{self.model_name}_latest.pth")
        torch.save(model.state_dict(), latest_path)
        
        # Save epoch model
        epoch_path = os.path.join(self.save_dir, f"{self.model_name}_epoch_{epoch}.pth")
        torch.save(model.state_dict(), epoch_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, f"{self.model_name}_best.pth")
            torch.save(model.state_dict(), best_path)
            
            if self.verbose and val_loss is not None:
                print(f"Saved best model at epoch {epoch} with validation loss {val_loss:.6f}")

class AdversarialTrainer:
    """
    Adversarial trainer for GraphTransGeo++
    """
    def __init__(self, model, optimizer, loss_fn, device, epsilon=0.01, alpha=0.01, beta=0.5, clip_grad=1.0):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.clip_grad = clip_grad
        
        # Create learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    def train_epoch(self, data_loader, epoch):
        """
        Train for one epoch with adversarial training
        
        Args:
            data_loader: Data loader for training
            epoch: Current epoch
            
        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_haversine = 0.0
        total_adv_loss = 0.0
        num_batches = 0
        
        # Training loop
        for batch in data_loader:
            # Move batch to device
            batch = batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with adversarial training
            if hasattr(self.model, 'adversarial_loss'):
                # Model has built-in adversarial training
                pred, adv_loss = self.model.adversarial_loss(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                    alpha=self.alpha, beta=self.beta
                )
            else:
                # Standard forward pass
                pred = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                adv_loss = torch.tensor(0.0, device=self.device)
            
            # Calculate loss
            loss, components = self.loss_fn(pred, batch.y, adv_loss)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_mse += components['mse']
            total_haversine += components['haversine']
            total_adv_loss += components['adversarial']
            num_batches += 1
        
        # Update learning rate
        self.scheduler.step(epoch)
        
        # Calculate average metrics
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_haversine = total_haversine / num_batches
        avg_adv_loss = total_adv_loss / num_batches
        
        # Return metrics
        metrics = {
            'loss': avg_loss,
            'mse': avg_mse,
            'haversine': avg_haversine,
            'adversarial': avg_adv_loss,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def validate(self, data_loader):
        """
        Validate model on validation set
        
        Args:
            data_loader: Data loader for validation
            
        Returns:
            metrics: Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_haversine = 0.0
        num_batches = 0
        
        # Validation loop
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = batch.to(self.device)
                
                # Forward pass
                pred = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                # Calculate loss
                loss, components = self.loss_fn(pred, batch.y)
                
                # Update metrics
                total_loss += loss.item()
                total_mse += components['mse']
                total_haversine += components['haversine']
                num_batches += 1
        
        # Calculate average metrics
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_haversine = total_haversine / num_batches
        
        # Return metrics
        metrics = {
            'loss': avg_loss,
            'mse': avg_mse,
            'haversine': avg_haversine
        }
        
        return metrics
