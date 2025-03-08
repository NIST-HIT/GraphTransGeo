#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Generate comprehensive visualizations of model performance

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import argparse
from matplotlib.ticker import PercentFormatter
import seaborn as sns

def parse_log_files(dataset_names):
    """
    Parse training and test log files for multiple datasets
    
    Args:
        dataset_names: List of dataset names
        
    Returns:
        training_data: Dictionary of training data for each dataset
        test_data: Dictionary of test data for each dataset
    """
    training_data = {}
    test_data = {}
    
    for dataset in dataset_names:
        # Parse training log file
        train_log_file = f'asset/log/{dataset}_training_gcn_optimized.log'
        if os.path.exists(train_log_file):
            epochs, train_losses, train_mse_losses, train_haversine_losses, val_losses, val_mse_losses, val_haversine_losses = parse_training_log(train_log_file)
            
            training_data[dataset] = {
                'epochs': epochs,
                'train_losses': train_losses,
                'train_mse_losses': train_mse_losses,
                'train_haversine_losses': train_haversine_losses,
                'val_losses': val_losses,
                'val_mse_losses': val_mse_losses,
                'val_haversine_losses': val_haversine_losses
            }
        
        # Parse test log file
        test_log_file = f'asset/log/{dataset}_test_gcn_optimized.log'
        if os.path.exists(test_log_file):
            metrics = parse_test_log(test_log_file)
            test_data[dataset] = metrics
    
    return training_data, test_data

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

def plot_training_curves(training_data, output_dir):
    """
    Plot training curves for all datasets
    
    Args:
        training_data: Dictionary of training data for each dataset
        output_dir: Output directory
    """
    for dataset, data in training_data.items():
        plt.figure(figsize=(15, 15))
        
        # Plot total loss
        plt.subplot(3, 1, 1)
        plt.plot(data['epochs'], data['train_losses'], 'b-', label='Train Loss')
        plt.plot(data['epochs'], data['val_losses'], 'r-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss - {dataset}')
        plt.legend()
        plt.grid(True)
        
        # Plot MSE loss
        plt.subplot(3, 1, 2)
        plt.plot(data['epochs'], data['train_mse_losses'], 'b-', label='Train MSE')
        plt.plot(data['epochs'], data['val_mse_losses'], 'r-', label='Val MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title(f'Training and Validation MSE - {dataset}')
        plt.legend()
        plt.grid(True)
        
        # Plot Haversine loss
        plt.subplot(3, 1, 3)
        plt.plot(data['epochs'], data['train_haversine_losses'], 'b-', label='Train Haversine')
        plt.plot(data['epochs'], data['val_haversine_losses'], 'r-', label='Val Haversine')
        plt.xlabel('Epoch')
        plt.ylabel('Haversine Loss (km)')
        plt.title(f'Training and Validation Haversine Loss - {dataset}')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset}_training_curves.png'))
        plt.close()
        
        print(f'Training curves for {dataset} saved to {os.path.join(output_dir, f"{dataset}_training_curves.png")}')

def plot_error_distributions(test_data, output_dir):
    """
    Plot error distributions for all datasets
    
    Args:
        test_data: Dictionary of test data for each dataset
        output_dir: Output directory
    """
    for dataset, metrics in test_data.items():
        if 'errors' in metrics:
            errors = metrics['errors']
            
            # Plot CDF
            plt.figure(figsize=(10, 6))
            sorted_errors = np.sort(errors)
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            
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
            
            # Plot histogram
            plt.figure(figsize=(10, 6))
            plt.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
            
            median = np.median(errors)
            plt.axvline(x=median, color='r', linestyle='--', linewidth=2, label=f'Median: {median:.2f} km')
            
            mean = np.mean(errors)
            plt.axvline(x=mean, color='g', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f} km')
            
            plt.xlabel('Distance Error (km)')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of Distance Errors - {dataset}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.savefig(os.path.join(output_dir, f'{dataset}_distance_histogram.png'))
            plt.close()
            
            print(f'Error histogram for {dataset} saved to {os.path.join(output_dir, f"{dataset}_distance_histogram.png")}')

def plot_model_comparison(test_data, mlp_metrics, output_dir):
    """
    Plot model comparison for all datasets
    
    Args:
        test_data: Dictionary of test data for each dataset
        mlp_metrics: Dictionary of MLP metrics for each dataset
        output_dir: Output directory
    """
    # Prepare data for comparison
    datasets = list(mlp_metrics.keys())
    metrics = ['mse', 'mae', 'median']
    metric_names = {'mse': 'MSE', 'mae': 'MAE', 'median': 'Median Distance Error (km)'}
    
    for metric in metrics:
        gcn_values = []
        mlp_values = []
        
        for dataset in datasets:
            if dataset in test_data and metric in test_data[dataset]:
                gcn_values.append(test_data[dataset][metric])
            else:
                gcn_values.append(0)
            
            if dataset in mlp_metrics and metric in mlp_metrics[dataset]:
                mlp_values.append(mlp_metrics[dataset][metric])
            else:
                mlp_values.append(0)
        
        # Plot comparison
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

def create_summary_table(test_data, mlp_metrics, output_file):
    """
    Create summary table of all metrics
    
    Args:
        test_data: Dictionary of test data for each dataset
        mlp_metrics: Dictionary of MLP metrics for each dataset
        output_file: Output file path
    """
    # Prepare data for table
    datasets = list(mlp_metrics.keys())
    metrics = ['mse', 'mae', 'median']
    
    data = []
    for dataset in datasets:
        row = {'Dataset': dataset}
        
        for metric in metrics:
            if dataset in test_data and metric in test_data[dataset]:
                row[f'GCN {metric.upper()}'] = test_data[dataset][metric]
            else:
                row[f'GCN {metric.upper()}'] = 'N/A'
            
            if dataset in mlp_metrics and metric in mlp_metrics[dataset]:
                row[f'MLP {metric.upper()}'] = mlp_metrics[dataset][metric]
            else:
                row[f'MLP {metric.upper()}'] = 'N/A'
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f'Summary table saved to {output_file}')
    
    # Return formatted markdown table
    return df.to_markdown(index=False)

def plot_heatmap(test_data, mlp_metrics, output_dir):
    """
    Plot heatmap of performance improvement
    
    Args:
        test_data: Dictionary of test data for each dataset
        mlp_metrics: Dictionary of MLP metrics for each dataset
        output_dir: Output directory
    """
    # Prepare data for heatmap
    datasets = list(mlp_metrics.keys())
    metrics = ['mse', 'mae', 'median']
    
    improvement_data = np.zeros((len(metrics), len(datasets)))
    
    for i, metric in enumerate(metrics):
        for j, dataset in enumerate(datasets):
            if dataset in test_data and metric in test_data[dataset] and dataset in mlp_metrics and metric in mlp_metrics[dataset]:
                # Calculate improvement percentage
                gcn_value = test_data[dataset][metric]
                mlp_value = mlp_metrics[dataset][metric]
                
                if mlp_value != 0:
                    improvement = (mlp_value - gcn_value) / mlp_value * 100
                    improvement_data[i, j] = improvement
    
    # Plot heatmap
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
    parser = argparse.ArgumentParser(description='Generate comprehensive visualizations of model performance')
    parser.add_argument('--output_dir', type=str, default='asset/figures', help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dataset names
    dataset_names = ['New_York', 'Shanghai', 'Los_Angeles']
    
    # Parse log files
    training_data, test_data = parse_log_files(dataset_names)
    
    # MLP metrics (from previous results)
    mlp_metrics = {
        'New_York': {'mse': 3.51, 'mae': 1.49, 'median': 224.83},
        'Shanghai': {'mse': 7859.51, 'mae': 76.32, 'median': 12953.86},
        'Los_Angeles': {'mse': 7569.97, 'mae': 76.15, 'median': 12573.21}
    }
    
    # Plot training curves
    if training_data:
        plot_training_curves(training_data, args.output_dir)
    
    # Plot error distributions
    if test_data:
        plot_error_distributions(test_data, args.output_dir)
    
    # Plot model comparison
    if test_data:
        plot_model_comparison(test_data, mlp_metrics, args.output_dir)
    
    # Create summary table
    if test_data:
        output_file = os.path.join('reports', 'model_comparison_summary.csv')
        os.makedirs('reports', exist_ok=True)
        markdown_table = create_summary_table(test_data, mlp_metrics, output_file)
        print('\nModel Comparison Summary:')
        print(markdown_table)
    
    # Plot heatmap
    if test_data:
        plot_heatmap(test_data, mlp_metrics, args.output_dir)

if __name__ == '__main__':
    main()
