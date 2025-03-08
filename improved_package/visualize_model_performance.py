#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Visualize model performance across different datasets

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import argparse
import seaborn as sns
from matplotlib.ticker import PercentFormatter

def parse_training_log(log_file):
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

def parse_test_log(log_file):
    """
    Parse test log file to extract metrics
    
    Args:
        log_file: Path to log file
        
    Returns:
        metrics: Dictionary of metrics
    """
    metrics = {}
    
    with open(log_file, 'r') as f:
        content = f.read()
        
        # Extract MSE
        mse_match = re.search(r'Test MSE:\s+([\d\.]+)', content)
        if mse_match:
            metrics['mse'] = float(mse_match.group(1))
        
        # Extract MAE
        mae_match = re.search(r'Test MAE:\s+([\d\.]+)', content)
        if mae_match:
            metrics['mae'] = float(mae_match.group(1))
        
        # Extract median distance error
        median_match = re.search(r'Median Distance Error:\s+([\d\.]+)', content)
        if median_match:
            metrics['median'] = float(median_match.group(1))
        
        # Extract distance errors if available
        errors_match = re.search(r'Distance Errors \(km\):\s+\[([\d\.,\s]+)\]', content)
        if errors_match:
            errors_str = errors_match.group(1)
            errors = [float(x) for x in errors_str.split(',')]
            metrics['errors'] = np.array(errors)
    
    return metrics

def plot_training_curves(dataset, output_dir):
    """
    Plot training curves for a dataset
    
    Args:
        dataset: Dataset name
        output_dir: Output directory
    """
    log_file = f'asset/log/{dataset}_training_gcn_optimized.log'
    
    if not os.path.exists(log_file):
        print(f'Training log file not found: {log_file}')
        return
    
    # Parse training log
    epochs, train_losses, train_mse_losses, train_haversine_losses, val_losses, val_mse_losses, val_haversine_losses = parse_training_log(log_file)
    
    # Create figure
    plt.figure(figsize=(15, 15))
    
    # Plot total loss
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    plt.plot(epochs, val_losses, 'r-', label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title(f'训练和验证损失 - {dataset}')
    plt.legend()
    plt.grid(True)
    
    # Plot MSE loss
    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_mse_losses, 'b-', label='训练MSE')
    plt.plot(epochs, val_mse_losses, 'r-', label='验证MSE')
    plt.xlabel('轮次')
    plt.ylabel('MSE')
    plt.title(f'训练和验证MSE - {dataset}')
    plt.legend()
    plt.grid(True)
    
    # Plot Haversine loss
    plt.subplot(3, 1, 3)
    plt.plot(epochs, train_haversine_losses, 'b-', label='训练Haversine损失')
    plt.plot(epochs, val_haversine_losses, 'r-', label='验证Haversine损失')
    plt.xlabel('轮次')
    plt.ylabel('Haversine损失 (km)')
    plt.title(f'训练和验证Haversine损失 - {dataset}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset}_training_curves_cn.png'))
    plt.close()
    
    print(f'Training curves for {dataset} saved to {os.path.join(output_dir, f"{dataset}_training_curves_cn.png")}')
    
    # Create English version
    plt.figure(figsize=(15, 15))
    
    # Plot total loss
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {dataset}')
    plt.legend()
    plt.grid(True)
    
    # Plot MSE loss
    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_mse_losses, 'b-', label='Training MSE')
    plt.plot(epochs, val_mse_losses, 'r-', label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(f'Training and Validation MSE - {dataset}')
    plt.legend()
    plt.grid(True)
    
    # Plot Haversine loss
    plt.subplot(3, 1, 3)
    plt.plot(epochs, train_haversine_losses, 'b-', label='Training Haversine Loss')
    plt.plot(epochs, val_haversine_losses, 'r-', label='Validation Haversine Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Haversine Loss (km)')
    plt.title(f'Training and Validation Haversine Loss - {dataset}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset}_training_curves.png'))
    plt.close()
    
    print(f'Training curves for {dataset} saved to {os.path.join(output_dir, f"{dataset}_training_curves.png")}')

def plot_error_distribution(dataset, output_dir):
    """
    Plot error distribution for a dataset
    
    Args:
        dataset: Dataset name
        output_dir: Output directory
    """
    log_file = f'asset/log/{dataset}_test_gcn_optimized.log'
    
    if not os.path.exists(log_file):
        print(f'Test log file not found: {log_file}')
        return
    
    # Parse test log
    metrics = parse_test_log(log_file)
    
    if 'errors' not in metrics:
        print(f'No error data found in test log for {dataset}')
        return
    
    errors = metrics['errors']
    
    # Plot CDF (Chinese)
    plt.figure(figsize=(10, 6))
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    
    plt.plot(sorted_errors, cdf, 'b-', linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='中位数')
    plt.axhline(y=0.75, color='g', linestyle='--', alpha=0.5, label='75百分位')
    plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90百分位')
    
    plt.xlabel('距离误差 (km)')
    plt.ylabel('累积概率')
    plt.title(f'距离误差累积分布函数 - {dataset}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    
    plt.savefig(os.path.join(output_dir, f'{dataset}_distance_cdf_cn.png'))
    plt.close()
    
    print(f'Error CDF for {dataset} saved to {os.path.join(output_dir, f"{dataset}_distance_cdf_cn.png")}')
    
    # Plot CDF (English)
    plt.figure(figsize=(10, 6))
    
    plt.plot(sorted_errors, cdf, 'b-', linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Median')
    plt.axhline(y=0.75, color='g', linestyle='--', alpha=0.5, label='75th Percentile')
    plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90th Percentile')
    
    plt.xlabel('Distance Error (km)')
    plt.ylabel('Cumulative Probability')
    plt.title(f'CDF of Distance Errors - {dataset}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    
    plt.savefig(os.path.join(output_dir, f'{dataset}_distance_cdf.png'))
    plt.close()
    
    print(f'Error CDF for {dataset} saved to {os.path.join(output_dir, f"{dataset}_distance_cdf.png")}')
    
    # Plot histogram (Chinese)
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    
    median = np.median(errors)
    plt.axvline(x=median, color='r', linestyle='--', linewidth=2, label=f'中位数: {median:.2f} km')
    
    mean = np.mean(errors)
    plt.axvline(x=mean, color='g', linestyle='--', linewidth=2, label=f'平均值: {mean:.2f} km')
    
    plt.xlabel('距离误差 (km)')
    plt.ylabel('频率')
    plt.title(f'距离误差直方图 - {dataset}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, f'{dataset}_distance_histogram_cn.png'))
    plt.close()
    
    print(f'Error histogram for {dataset} saved to {os.path.join(output_dir, f"{dataset}_distance_histogram_cn.png")}')
    
    # Plot histogram (English)
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    
    plt.axvline(x=median, color='r', linestyle='--', linewidth=2, label=f'Median: {median:.2f} km')
    plt.axvline(x=mean, color='g', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f} km')
    
    plt.xlabel('Distance Error (km)')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Distance Errors - {dataset}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, f'{dataset}_distance_histogram.png'))
    plt.close()
    
    print(f'Error histogram for {dataset} saved to {os.path.join(output_dir, f"{dataset}_distance_histogram.png")}')

def plot_model_comparison(datasets, output_dir):
    """
    Plot model comparison for all datasets
    
    Args:
        datasets: List of dataset names
        output_dir: Output directory
    """
    # Parse test logs for GraphTransGeo
    gcn_metrics = {}
    for dataset in datasets:
        log_file = f'asset/log/{dataset}_test_gcn_optimized.log'
        if os.path.exists(log_file):
            gcn_metrics[dataset] = parse_test_log(log_file)
        else:
            gcn_metrics[dataset] = {}
    
    # MLP metrics (from previous results)
    mlp_metrics = {
        'New_York': {'mse': 3.51, 'mae': 1.49, 'median': 224.83},
        'Shanghai': {'mse': 7859.51, 'mae': 76.32, 'median': 12953.86},
        'Los_Angeles': {'mse': 7569.97, 'mae': 76.15, 'median': 12573.21}
    }
    
    # Metrics to plot
    metrics = ['mse', 'mae', 'median']
    metric_names = {
        'mse': 'MSE', 
        'mae': 'MAE', 
        'median': 'Median Distance Error (km)'
    }
    metric_names_cn = {
        'mse': 'MSE', 
        'mae': 'MAE', 
        'median': '中位距离误差 (km)'
    }
    
    # Plot comparison for each metric (Chinese)
    for metric in metrics:
        gcn_values = []
        mlp_values = []
        
        for dataset in datasets:
            if dataset in gcn_metrics and metric in gcn_metrics[dataset]:
                gcn_values.append(gcn_metrics[dataset][metric])
            else:
                gcn_values.append(0)
            
            if dataset in mlp_metrics and metric in mlp_metrics[dataset]:
                mlp_values.append(mlp_metrics[dataset][metric])
            else:
                mlp_values.append(0)
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(datasets))
        width = 0.35
        
        plt.bar(x - width/2, gcn_values, width, label='GCN')
        plt.bar(x + width/2, mlp_values, width, label='MLP')
        
        plt.xlabel('数据集')
        plt.ylabel(metric_names_cn[metric])
        plt.title(f'{metric_names_cn[metric]} 跨数据集比较')
        plt.xticks(x, datasets)
        plt.legend()
        plt.grid(True, axis='y')
        
        # Use log scale if values vary significantly
        if max(gcn_values + mlp_values) / (min(gcn_values + mlp_values) + 1e-10) > 100:
            plt.yscale('log')
            plt.title(f'{metric_names_cn[metric]} 跨数据集比较 (对数尺度)')
        
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison_cn.png'))
        plt.close()
        
        print(f'{metric_names_cn[metric]} comparison saved to {os.path.join(output_dir, f"{metric}_comparison_cn.png")}')
    
    # Plot comparison for each metric (English)
    for metric in metrics:
        gcn_values = []
        mlp_values = []
        
        for dataset in datasets:
            if dataset in gcn_metrics and metric in gcn_metrics[dataset]:
                gcn_values.append(gcn_metrics[dataset][metric])
            else:
                gcn_values.append(0)
            
            if dataset in mlp_metrics and metric in mlp_metrics[dataset]:
                mlp_values.append(mlp_metrics[dataset][metric])
            else:
                mlp_values.append(0)
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(datasets))
        width = 0.35
        
        plt.bar(x - width/2, gcn_values, width, label='GCN')
        plt.bar(x + width/2, mlp_values, width, label='MLP')
        
        plt.xlabel('Datasets')
        plt.ylabel(metric_names[metric])
        plt.title(f'{metric_names[metric]} Comparison Across Datasets')
        plt.xticks(x, datasets)
        plt.legend()
        plt.grid(True, axis='y')
        
        # Use log scale if values vary significantly
        if max(gcn_values + mlp_values) / (min(gcn_values + mlp_values) + 1e-10) > 100:
            plt.yscale('log')
            plt.title(f'{metric_names[metric]} Comparison Across Datasets (Log Scale)')
        
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'))
        plt.close()
        
        print(f'{metric_names[metric]} comparison saved to {os.path.join(output_dir, f"{metric}_comparison.png")}')

def plot_improvement_heatmap(datasets, output_dir):
    """
    Plot heatmap of performance improvement
    
    Args:
        datasets: List of dataset names
        output_dir: Output directory
    """
    # Parse test logs for GraphTransGeo
    gcn_metrics = {}
    for dataset in datasets:
        log_file = f'asset/log/{dataset}_test_gcn_optimized.log'
        if os.path.exists(log_file):
            gcn_metrics[dataset] = parse_test_log(log_file)
        else:
            gcn_metrics[dataset] = {}
    
    # MLP metrics (from previous results)
    mlp_metrics = {
        'New_York': {'mse': 3.51, 'mae': 1.49, 'median': 224.83},
        'Shanghai': {'mse': 7859.51, 'mae': 76.32, 'median': 12953.86},
        'Los_Angeles': {'mse': 7569.97, 'mae': 76.15, 'median': 12573.21}
    }
    
    # Metrics to plot
    metrics = ['mse', 'mae', 'median']
    
    # Prepare data for heatmap
    improvement_data = np.zeros((len(metrics), len(datasets)))
    
    for i, metric in enumerate(metrics):
        for j, dataset in enumerate(datasets):
            if dataset in gcn_metrics and metric in gcn_metrics[dataset] and dataset in mlp_metrics and metric in mlp_metrics[dataset]:
                # Calculate improvement percentage
                gcn_value = gcn_metrics[dataset][metric]
                mlp_value = mlp_metrics[dataset][metric]
                
                if mlp_value != 0:
                    improvement = (mlp_value - gcn_value) / mlp_value * 100
                    improvement_data[i, j] = improvement
    
    # Plot heatmap (Chinese)
    plt.figure(figsize=(10, 6))
    sns.heatmap(improvement_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                xticklabels=datasets, yticklabels=[m.upper() for m in metrics])
    
    plt.title('GraphTransGeo相对于MLP的性能提升 (%)')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'performance_improvement_heatmap_cn.png'))
    plt.close()
    
    print(f'Performance improvement heatmap saved to {os.path.join(output_dir, "performance_improvement_heatmap_cn.png")}')
    
    # Plot heatmap (English)
    plt.figure(figsize=(10, 6))
    sns.heatmap(improvement_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                xticklabels=datasets, yticklabels=[m.upper() for m in metrics])
    
    plt.title('Performance Improvement of GraphTransGeo over MLP (%)')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'performance_improvement_heatmap.png'))
    plt.close()
    
    print(f'Performance improvement heatmap saved to {os.path.join(output_dir, "performance_improvement_heatmap.png")}')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize model performance across different datasets')
    parser.add_argument('--output_dir', type=str, default='asset/figures', help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dataset names
    datasets = ['New_York', 'Shanghai', 'Los_Angeles']
    
    # Plot training curves for each dataset
    for dataset in datasets:
        plot_training_curves(dataset, args.output_dir)
    
    # Plot error distribution for each dataset
    for dataset in datasets:
        plot_error_distribution(dataset, args.output_dir)
    
    # Plot model comparison
    plot_model_comparison(datasets, args.output_dir)
    
    # Plot improvement heatmap
    plot_improvement_heatmap(datasets, args.output_dir)

if __name__ == '__main__':
    main()
