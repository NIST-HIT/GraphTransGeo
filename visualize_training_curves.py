#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Visualize training curves for the optimized GraphTransGeo model

import os
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse
import pandas as pd
import seaborn as sns

def parse_training_log(log_file):
    """
    Parse training log file to extract training and validation metrics
    
    Args:
        log_file: Path to log file
        
    Returns:
        metrics: Dictionary of metrics
    """
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_mse': [],
        'train_haversine': [],
        'train_adversarial': [],
        'val_loss': [],
        'val_mse': [],
        'val_haversine': []
    }
    
    with open(log_file, 'r') as f:
        content = f.read()
        
        # Extract epochs and metrics
        epoch_pattern = r'Epoch (\d+)/\d+:'
        train_loss_pattern = r'Train Loss: ([\d\.]+)'
        train_mse_pattern = r'Train Loss: [\d\.]+, MSE: ([\d\.]+)'
        train_haversine_pattern = r'Haversine: ([\d\.]+)'
        train_adversarial_pattern = r'Adversarial: ([\d\.]+)'
        val_loss_pattern = r'Val Loss: ([\d\.]+)'
        val_mse_pattern = r'Val Loss: [\d\.]+, MSE: ([\d\.]+)'
        val_haversine_pattern = r'Val Loss: [\d\.]+, MSE: [\d\.]+, Haversine: ([\d\.]+)'
        
        # Find all matches
        epochs = re.findall(epoch_pattern, content)
        train_losses = re.findall(train_loss_pattern, content)
        train_mses = re.findall(train_mse_pattern, content)
        train_haversines = re.findall(train_haversine_pattern, content)
        train_adversarials = re.findall(train_adversarial_pattern, content)
        val_losses = re.findall(val_loss_pattern, content)
        val_mses = re.findall(val_mse_pattern, content)
        val_haversines = re.findall(val_haversine_pattern, content)
        
        # Convert to appropriate types
        metrics['epoch'] = [int(e) for e in epochs]
        metrics['train_loss'] = [float(l) for l in train_losses]
        metrics['train_mse'] = [float(m) for m in train_mses]
        metrics['train_haversine'] = [float(h) for h in train_haversines[:len(epochs)]]  # Only take the first match per epoch
        metrics['train_adversarial'] = [float(a) for a in train_adversarials]
        metrics['val_loss'] = [float(l) for l in val_losses]
        metrics['val_mse'] = [float(m) for m in val_mses]
        metrics['val_haversine'] = [float(h) for h in val_haversines]
    
    return metrics

def plot_training_curves(metrics, output_file, dataset_name, english=True):
    """
    Plot training and validation curves
    
    Args:
        metrics: Dictionary of metrics
        output_file: Path to save the plot
        dataset_name: Name of the dataset
        english: Whether to use English labels (True) or Chinese labels (False)
    """
    plt.figure(figsize=(15, 10))
    
    # Set font for Chinese characters if needed
    if not english:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # For Chinese characters
        plt.rcParams['axes.unicode_minus'] = False  # For minus sign
    
    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], 'b-', label='Train' if english else '训练')
    plt.plot(metrics['epoch'], metrics['val_loss'], 'r-', label='Validation' if english else '验证')
    if english:
        plt.title(f'{dataset_name} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
    else:
        plt.title(f'{dataset_name} 损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot training and validation MSE
    plt.subplot(2, 2, 2)
    plt.plot(metrics['epoch'], metrics['train_mse'], 'b-', label='Train' if english else '训练')
    plt.plot(metrics['epoch'], metrics['val_mse'], 'r-', label='Validation' if english else '验证')
    if english:
        plt.title(f'{dataset_name} MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
    else:
        plt.title(f'{dataset_name} 均方误差')
        plt.xlabel('轮次')
        plt.ylabel('均方误差')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot training and validation Haversine loss
    plt.subplot(2, 2, 3)
    plt.plot(metrics['epoch'], metrics['train_haversine'], 'b-', label='Train' if english else '训练')
    plt.plot(metrics['epoch'], metrics['val_haversine'], 'r-', label='Validation' if english else '验证')
    if english:
        plt.title(f'{dataset_name} Haversine Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Haversine Loss')
    else:
        plt.title(f'{dataset_name} Haversine 损失')
        plt.xlabel('轮次')
        plt.ylabel('Haversine 损失')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot adversarial loss
    plt.subplot(2, 2, 4)
    plt.plot(metrics['epoch'], metrics['train_adversarial'], 'g-')
    if english:
        plt.title(f'{dataset_name} Adversarial Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Adversarial Loss')
    else:
        plt.title(f'{dataset_name} 对抗损失')
        plt.xlabel('轮次')
        plt.ylabel('对抗损失')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Training curves saved to {output_file}')

def plot_error_distribution(log_file, output_file, dataset_name, english=True):
    """
    Plot error distribution from test log file
    
    Args:
        log_file: Path to test log file
        output_file: Path to save the plot
        dataset_name: Name of the dataset
        english: Whether to use English labels (True) or Chinese labels (False)
    """
    # Extract distance errors from log file
    with open(log_file, 'r') as f:
        content = f.read()
        
        # Extract MSE, MAE, and median distance error
        mse_match = re.search(r'Test MSE:\s+([\d\.]+)', content)
        mae_match = re.search(r'Test MAE:\s+([\d\.]+)', content)
        median_match = re.search(r'Median Distance Error:\s+([\d\.]+)', content)
        
        if mse_match and mae_match and median_match:
            mse = float(mse_match.group(1))
            mae = float(mae_match.group(1))
            median = float(median_match.group(1))
        else:
            print(f"Could not extract metrics from {log_file}")
            return
    
    # Set font for Chinese characters if needed
    if not english:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # For Chinese characters
        plt.rcParams['axes.unicode_minus'] = False  # For minus sign
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create bar chart for metrics
    metrics = ['MSE', 'MAE', 'Median (km)']
    values = [mse, mae, median]
    
    if not english:
        metrics = ['均方误差', '平均绝对误差', '中位距离误差 (km)']
    
    # Use different colors for each metric
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Create bar chart
    bars = plt.bar(metrics, values, color=colors, alpha=0.8)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Use log scale for y-axis
    plt.yscale('log')
    
    # Add labels and title
    if english:
        plt.title(f'{dataset_name} Test Metrics')
        plt.ylabel('Value (log scale)')
    else:
        plt.title(f'{dataset_name} 测试指标')
        plt.ylabel('值 (对数刻度)')
    
    plt.grid(True, axis='y', alpha=0.3)
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Error distribution saved to {output_file}')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize training curves for the optimized GCN model')
    parser.add_argument('--dataset', type=str, default='New_York', help='Dataset name')
    parser.add_argument('--log_dir', type=str, default='asset/log', help='Directory containing log files')
    parser.add_argument('--output_dir', type=str, default='asset/figures', help='Directory to save plots')
    parser.add_argument('--chinese', action='store_true', help='Use Chinese labels')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse training log file
    train_log_file = f'{args.log_dir}/{args.dataset}_train_gcn_optimized.log'
    if os.path.exists(train_log_file):
        metrics = parse_training_log(train_log_file)
        
        # Plot training curves
        output_file = f'{args.output_dir}/{args.dataset}_training_curves_detailed.png'
        plot_training_curves(metrics, output_file, args.dataset, english=not args.chinese)
        
        # Plot training curves with Chinese labels if not already done
        if not args.chinese:
            output_file = f'{args.output_dir}/{args.dataset}_training_curves_detailed_cn.png'
            plot_training_curves(metrics, output_file, args.dataset, english=False)
    else:
        print(f"Training log file not found: {train_log_file}")
    
    # Parse test log file
    test_log_file = f'{args.log_dir}/{args.dataset}_test_gcn_optimized.log'
    if os.path.exists(test_log_file):
        # Plot error distribution
        output_file = f'{args.output_dir}/{args.dataset}_error_metrics.png'
        plot_error_distribution(test_log_file, output_file, args.dataset, english=not args.chinese)
        
        # Plot error distribution with Chinese labels if not already done
        if not args.chinese:
            output_file = f'{args.output_dir}/{args.dataset}_error_metrics_cn.png'
            plot_error_distribution(test_log_file, output_file, args.dataset, english=False)
    else:
        print(f"Test log file not found: {test_log_file}")

if __name__ == '__main__':
    main()
