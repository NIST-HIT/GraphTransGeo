#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script to verify the effect of normalization on loss values

import os
import numpy as np
import torch
import torch.nn as nn
import math
import logging
from gcn_data_loader import load_dataset, GraphDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class HaversineLoss(nn.Module):
    def __init__(self):
        super(HaversineLoss, self).__init__()
    
    def forward(self, pred, target):
        # Convert to radians
        pred_lat = pred[:, 0] * math.pi / 180
        pred_lon = pred[:, 1] * math.pi / 180
        target_lat = target[:, 0] * math.pi / 180
        target_lon = target[:, 1] * math.pi / 180
        
        # Calculate Haversine distance
        dlon = target_lon - pred_lon
        dlat = target_lat - pred_lat
        
        a = torch.sin(dlat/2)**2 + torch.cos(pred_lat) * torch.cos(target_lat) * torch.sin(dlon/2)**2
        c = 2 * torch.asin(torch.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        # Calculate loss
        loss = c * r
        
        return loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, hav_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.haversine_loss = HaversineLoss()
        self.hav_weight = hav_weight
    
    def forward(self, pred, target):
        # Calculate MSE loss
        mse = self.mse_loss(pred, target)
        
        # Calculate Haversine loss
        hav = self.haversine_loss(pred, target)
        
        # Calculate combined loss
        loss = (1 - self.hav_weight) * mse + self.hav_weight * hav
        
        return loss, mse, hav

def main():
    # Load dataset
    dataset_name = 'New_York'
    data_dir = 'asset/data'
    
    # Load dataset without normalization
    train_dataset, val_dataset, test_dataset, input_dim = load_dataset(
        dataset_name, data_dir, seed=42
    )
    
    # Create loss function
    criterion = CombinedLoss(hav_weight=0.3)
    
    # Calculate loss on unnormalized data
    unnorm_losses = []
    unnorm_preds = []
    unnorm_targets = []
    
    for i in range(min(10, len(train_dataset))):
        data = train_dataset[i]
        pred = data.y + torch.randn_like(data.y) * 0.1  # Simulate predictions with small error
        loss, mse, hav = criterion(pred, data.y)
        unnorm_losses.append((loss.item(), mse.item(), hav.item()))
        unnorm_preds.append(pred.detach().numpy())
        unnorm_targets.append(data.y.detach().numpy())
    
    # Calculate metrics for unnormalized data
    unnorm_preds = np.concatenate(unnorm_preds, axis=0)
    unnorm_targets = np.concatenate(unnorm_targets, axis=0)
    
    avg_unnorm_loss = np.mean([x[0] for x in unnorm_losses])
    avg_unnorm_mse = np.mean([x[1] for x in unnorm_losses])
    avg_unnorm_hav = np.mean([x[2] for x in unnorm_losses])
    
    # Calculate MSE and MAE manually
    unnorm_mse = np.mean((unnorm_preds - unnorm_targets) ** 2)
    unnorm_mae = np.mean(np.abs(unnorm_preds - unnorm_targets))
    
    logging.info(f'Unnormalized - Loss: {avg_unnorm_loss:.6f}, MSE: {avg_unnorm_mse:.6f}, Haversine: {avg_unnorm_hav:.6f}')
    logging.info(f'Unnormalized - Manual MSE: {unnorm_mse:.6f}, MAE: {unnorm_mae:.6f}')
    
    # Normalize data manually for testing
    all_y = torch.cat([data.y for data in train_dataset], dim=0)
    y_min = all_y.min(dim=0)[0]
    y_max = all_y.max(dim=0)[0]
    
    # Calculate loss on normalized data
    norm_losses = []
    norm_preds = []
    norm_targets = []
    denorm_preds = []
    denorm_targets = []
    
    for i in range(min(10, len(train_dataset))):
        data = train_dataset[i]
        
        # Normalize target
        y_norm = (data.y - y_min) / (y_max - y_min + 1e-12)
        
        # Simulate predictions with small error
        pred_norm = y_norm + torch.randn_like(y_norm) * 0.1
        
        # Calculate loss
        loss, mse, hav = criterion(pred_norm, y_norm)
        norm_losses.append((loss.item(), mse.item(), hav.item()))
        
        # Store normalized predictions and targets
        norm_preds.append(pred_norm.detach().numpy())
        norm_targets.append(y_norm.detach().numpy())
        
        # Denormalize for real-world metrics
        pred_denorm = pred_norm * (y_max - y_min) + y_min
        y_denorm = y_norm * (y_max - y_min) + y_min
        
        # Store denormalized predictions and targets
        denorm_preds.append(pred_denorm.detach().numpy())
        denorm_targets.append(y_denorm.detach().numpy())
    
    # Calculate metrics for normalized data
    norm_preds = np.concatenate(norm_preds, axis=0)
    norm_targets = np.concatenate(norm_targets, axis=0)
    
    avg_norm_loss = np.mean([x[0] for x in norm_losses])
    avg_norm_mse = np.mean([x[1] for x in norm_losses])
    avg_norm_hav = np.mean([x[2] for x in norm_losses])
    
    # Calculate MSE and MAE manually for normalized data
    norm_mse = np.mean((norm_preds - norm_targets) ** 2)
    norm_mae = np.mean(np.abs(norm_preds - norm_targets))
    
    logging.info(f'Normalized - Loss: {avg_norm_loss:.6f}, MSE: {avg_norm_mse:.6f}, Haversine: {avg_norm_hav:.6f}')
    logging.info(f'Normalized - Manual MSE: {norm_mse:.6f}, MAE: {norm_mae:.6f}')
    logging.info(f'Reduction factor: {avg_unnorm_loss / avg_norm_loss:.2f}x')
    
    # Calculate metrics for denormalized data
    denorm_preds = np.concatenate(denorm_preds, axis=0)
    denorm_targets = np.concatenate(denorm_targets, axis=0)
    
    # Calculate MSE and MAE for denormalized data
    denorm_mse = np.mean((denorm_preds - denorm_targets) ** 2)
    denorm_mae = np.mean(np.abs(denorm_preds - denorm_targets))
    
    logging.info(f'Denormalized - MSE: {denorm_mse:.6f}, MAE: {denorm_mae:.6f}')

if __name__ == '__main__':
    main()
