#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Visualize results of the optimized GraphTransGeo model

import os
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse

def parse_log_file(log_file):
    """
    Parse training log file to extract loss values
    
    Args:
        log_file: Path to log file
        
    Returns:
        epochs: List of epoch numbers
        train_losses: List of training losses
        train_mse_losses: List of training MSE losses
        train_haversine_losses: List of training Haversine losses
        val_losses: List of validation losses
        val_mse_losses: List of validation MSE losses
        val_haversine_losses: List of validation Haversine losses
    """
    epochs = []
    train_losses = []
    train_mse_losses = []
    train_haversine_losses = []
    val_losses = []
    val_mse_losses = []
    val_haversine_losses = []
    
    # Regular expression pattern to match log lines
    pattern = r'Epoch (\d+)/\d+ - Train Loss: ([\d\.]+), Train MSE: ([\d\.]+), Train Haversine: ([\d\.]+), Val Loss: ([\d\.]+), Val MSE: ([\d\.]+), Val Haversine: ([\d\.]+)'
    
    with open(log_file, 'r') as f:
        for line in f:
            match = re.match(pattern, line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                train_mse = float(match.group(3))
                train_haversine = float(match.group(4))
                val_loss = float(match.group(5))
                val_mse = float(match.group(6))
                val_haversine = float(match.group(7))
                
                epochs.append(epoch)
                train_losses.append(train_loss)
                train_mse_losses.append(train_mse)
                train_haversine_losses.append(train_haversine)
                val_losses.append(val_loss)
                val_mse_losses.append(val_mse)
                val_haversine_losses.append(val_haversine)
    
    return epochs, train_losses, train_mse_losses, train_haversine_losses, val_losses, val_mse_losses, val_haversine_losses

def parse_test_log_file(log_file):
    """
    Parse test log file to extract metrics
    
    Args:
        log_file: Path to log file
        
    Returns:
        mse: Mean squared error
        mae: Mean absolute error
        median_error: Median distance error
    """
    mse = None
    mae = None
    median_error = None
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'Test MSE:' in line:
                mse = float(line.split(':')[1].strip().split()[0])
            elif 'Test MAE:' in line:
                mae = float(line.split(':')[1].strip().split()[0])
            elif 'Median Distance Error:' in line:
                median_error = float(line.split(':')[1].strip().split()[0])
    
    return mse, mae, median_error

def plot_training_curves(epochs, train_losses, val_losses, train_mse_losses, val_mse_losses, 
                         train_haversine_losses, val_haversine_losses, output_file):
    """
    Plot training curves
    
    Args:
        epochs: List of epoch numbers
        train_losses: List of training losses
        val_losses: List of validation losses
        train_mse_losses: List of training MSE losses
        val_mse_losses: List of validation MSE losses
        train_haversine_losses: List of training Haversine losses
        val_haversine_losses: List of validation Haversine losses
        output_file: Path to save the plot
    """
    plt.figure(figsize=(15, 15))
    
    # Plot total loss
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot MSE loss
    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_mse_losses, 'b-', label='Train MSE')
    plt.plot(epochs, val_mse_losses, 'r-', label='Val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training and Validation MSE')
    plt.legend()
    plt.grid(True)
    
    # Plot Haversine loss
    plt.subplot(3, 1, 3)
    plt.plot(epochs, train_haversine_losses, 'b-', label='Train Haversine')
    plt.plot(epochs, val_haversine_losses, 'r-', label='Val Haversine')
    plt.xlabel('Epoch')
    plt.ylabel('Haversine Loss (km)')
    plt.title('Training and Validation Haversine Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_model_comparison(datasets, gcn_metrics, mlp_metrics, metric_name, output_file):
    """
    Plot model comparison
    
    Args:
        datasets: List of dataset names
        gcn_metrics: List of GCN metrics
        mlp_metrics: List of MLP metrics
        metric_name: Name of the metric
        output_file: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    x = np.arange(len(datasets))
    width = 0.35
    
    plt.bar(x - width/2, gcn_metrics, width, label='GCN')
    plt.bar(x + width/2, mlp_metrics, width, label='MLP')
    
    plt.xlabel('Datasets')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Comparison Across Datasets')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(True, axis='y')
    
    # Use log scale if values vary significantly
    if max(gcn_metrics + mlp_metrics) / (min(gcn_metrics + mlp_metrics) + 1e-10) > 100:
        plt.yscale('log')
        plt.title(f'{metric_name} Comparison Across Datasets (Log Scale)')
    
    plt.savefig(output_file)
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize results of the optimized GCN model')
    parser.add_argument('--dataset', type=str, default='New_York', help='Dataset name')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs('asset/figures', exist_ok=True)
    
    # Parse training log file
    log_file = f'asset/log/{args.dataset}_training_gcn_optimized.log'
    if os.path.exists(log_file):
        epochs, train_losses, train_mse_losses, train_haversine_losses, val_losses, val_mse_losses, val_haversine_losses = parse_log_file(log_file)
        
        # Plot training curves
        output_file = f'asset/figures/{args.dataset}_training_curves_gcn_optimized.png'
        plot_training_curves(epochs, train_losses, val_losses, train_mse_losses, val_mse_losses, 
                            train_haversine_losses, val_haversine_losses, output_file)
        
        print(f'Training curves saved to {output_file}')
    else:
        print(f'Training log file not found: {log_file}')
    
    # Parse test log files for all datasets
    datasets = ['New_York', 'Shanghai', 'Los_Angeles']
    gcn_mse = []
    gcn_mae = []
    gcn_median = []
    
    for dataset in datasets:
        test_log_file = f'asset/log/{dataset}_test_gcn_optimized.log'
        if os.path.exists(test_log_file):
            mse, mae, median_error = parse_test_log_file(test_log_file)
            if mse is not None:
                gcn_mse.append(mse)
            else:
                gcn_mse.append(0)
            
            if mae is not None:
                gcn_mae.append(mae)
            else:
                gcn_mae.append(0)
            
            if median_error is not None:
                gcn_median.append(median_error)
            else:
                gcn_median.append(0)
        else:
            gcn_mse.append(0)
            gcn_mae.append(0)
            gcn_median.append(0)
    
    # MLP metrics (from previous results)
    mlp_mse = [3.51, 7859.51, 7569.97]
    mlp_mae = [1.49, 76.32, 76.15]
    mlp_median = [224.83, 12953.86, 12573.21]
    
    # Plot model comparison
    if any(gcn_mse):
        plot_model_comparison(datasets, gcn_mse, mlp_mse, 'MSE', 'asset/figures/mse_comparison_gcn_optimized.png')
        print('MSE comparison saved to asset/figures/mse_comparison_gcn_optimized.png')
    
    if any(gcn_mae):
        plot_model_comparison(datasets, gcn_mae, mlp_mae, 'MAE', 'asset/figures/mae_comparison_gcn_optimized.png')
        print('MAE comparison saved to asset/figures/mae_comparison_gcn_optimized.png')
    
    if any(gcn_median):
        plot_model_comparison(datasets, gcn_median, mlp_median, 'Median Distance Error (km)', 'asset/figures/median_error_comparison_gcn_optimized.png')
        print('Median distance error comparison saved to asset/figures/median_error_comparison_gcn_optimized.png')

if __name__ == '__main__':
    main()
