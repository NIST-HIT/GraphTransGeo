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

def test_dataset(dataset_name, data_dir='asset/data', seed=42):
    """
    Test normalization effects on a specific dataset
    
    Args:
        dataset_name: Name of the dataset to test
        data_dir: Directory containing the dataset
        seed: Random seed for reproducibility
        
    Returns:
        results: Dictionary containing test results
    """
    logging.info(f"\n{'='*20} Testing {dataset_name} Dataset {'='*20}")
    
    # Load dataset with normalization
    train_dataset, val_dataset, test_dataset, input_dim = load_dataset(
        dataset_name, data_dir, seed=seed
    )
    
    # Create loss function
    criterion = CombinedLoss(hav_weight=0.3)
    
    # Calculate loss on unnormalized data
    unnorm_losses = []
    unnorm_preds = []
    unnorm_targets = []
    
    for i in range(min(10, len(train_dataset))):
        data = train_dataset[i]
        
        # Get original coordinates (denormalized)
        y_min = train_dataset.y_min
        y_max = train_dataset.y_max
        y_orig = data.y * (y_max - y_min) + y_min
        
        # Simulate predictions with small error on original scale
        pred_orig = y_orig + torch.randn_like(y_orig) * 0.1
        
        # Calculate loss on unnormalized data
        loss, mse, hav = criterion(pred_orig, y_orig)
        unnorm_losses.append((loss.item(), mse.item(), hav.item()))
        unnorm_preds.append(pred_orig.detach().numpy())
        unnorm_targets.append(y_orig.detach().numpy())
    
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
    
    # Calculate loss on normalized data
    norm_losses = []
    norm_preds = []
    norm_targets = []
    denorm_preds = []
    denorm_targets = []
    
    for i in range(min(10, len(train_dataset))):
        data = train_dataset[i]
        
        # Simulate predictions with small error on normalized scale
        pred_norm = data.y + torch.randn_like(data.y) * 0.1
        
        # Calculate loss on normalized data
        loss, mse, hav = criterion(pred_norm, data.y)
        norm_losses.append((loss.item(), mse.item(), hav.item()))
        
        # Store normalized predictions and targets
        norm_preds.append(pred_norm.detach().numpy())
        norm_targets.append(data.y.detach().numpy())
        
        # Denormalize for real-world metrics
        y_min = train_dataset.y_min
        y_max = train_dataset.y_max
        pred_denorm = pred_norm * (y_max - y_min) + y_min
        y_denorm = data.y * (y_max - y_min) + y_min
        
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
    
    # Calculate reduction factor
    reduction_factor = avg_unnorm_loss / avg_norm_loss
    logging.info(f'Reduction factor: {reduction_factor:.2f}x')
    
    # Calculate metrics for denormalized data
    denorm_preds = np.concatenate(denorm_preds, axis=0)
    denorm_targets = np.concatenate(denorm_targets, axis=0)
    
    # Calculate MSE and MAE for denormalized data
    denorm_mse = np.mean((denorm_preds - denorm_targets) ** 2)
    denorm_mae = np.mean(np.abs(denorm_preds - denorm_targets))
    
    logging.info(f'Denormalized - MSE: {denorm_mse:.6f}, MAE: {denorm_mae:.6f}')
    
    # Return results
    return {
        'dataset': dataset_name,
        'input_dim': input_dim,
        'unnormalized': {
            'loss': avg_unnorm_loss,
            'mse': avg_unnorm_mse,
            'hav': avg_unnorm_hav,
            'mae': unnorm_mae
        },
        'normalized': {
            'loss': avg_norm_loss,
            'mse': avg_norm_mse,
            'hav': avg_norm_hav,
            'mae': norm_mae
        },
        'denormalized': {
            'mse': denorm_mse,
            'mae': denorm_mae
        },
        'reduction_factor': reduction_factor,
        'y_min': train_dataset.y_min.tolist(),
        'y_max': train_dataset.y_max.tolist()
    }

def main():
    # Test multiple datasets
    dataset_names = ["New_York", "Los_Angeles", "Shanghai"]
    data_dir = 'asset/data'
    
    # Store results for all datasets
    all_results = {}
    
    # Test each dataset
    for dataset_name in dataset_names:
        try:
            results = test_dataset(dataset_name, data_dir)
            all_results[dataset_name] = results
        except Exception as e:
            logging.error(f"Error testing {dataset_name}: {e}")
    
    # Print summary
    logging.info("\n" + "="*60)
    logging.info("SUMMARY OF NORMALIZATION EFFECTS ACROSS DATASETS")
    logging.info("="*60)
    
    for dataset_name, results in all_results.items():
        logging.info(f"\n{dataset_name} (Input Dim: {results['input_dim']}):")
        logging.info(f"  Coordinate Range: {results['y_min']} to {results['y_max']}")
        logging.info(f"  Unnormalized Loss: {results['unnormalized']['loss']:.6f}")
        logging.info(f"  Normalized Loss: {results['normalized']['loss']:.6f}")
        logging.info(f"  Reduction Factor: {results['reduction_factor']:.2f}x")
        logging.info(f"  Denormalized MAE: {results['denormalized']['mae']:.6f}")
    
    logging.info("\n" + "="*60)
    
    # Calculate average reduction factor
    avg_reduction = np.mean([r['reduction_factor'] for r in all_results.values()])
    logging.info(f"Average Reduction Factor Across All Datasets: {avg_reduction:.2f}x")

if __name__ == '__main__':
    main()
