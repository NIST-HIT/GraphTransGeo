#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Analyze test results of the optimized GraphTransGeo model

import os
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse
import pandas as pd
from matplotlib.ticker import PercentFormatter

def parse_test_log_file(log_file):
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

def plot_error_cdf(errors, output_file, dataset_name):
    """
    Plot cumulative distribution function of distance errors
    
    Args:
        errors: Array of distance errors
        output_file: Path to save the plot
        dataset_name: Name of the dataset
    """
    plt.figure(figsize=(10, 6))
    
    # Sort errors
    sorted_errors = np.sort(errors)
    
    # Calculate CDF
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    
    # Plot CDF
    plt.plot(sorted_errors, cdf, 'b-', linewidth=2)
    
    # Add reference lines
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Median')
    plt.axhline(y=0.75, color='g', linestyle='--', alpha=0.5, label='75th Percentile')
    plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90th Percentile')
    
    # Add labels and title
    plt.xlabel('Distance Error (km)')
    plt.ylabel('Cumulative Probability')
    plt.title(f'CDF of Distance Errors - {dataset_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Error CDF plot saved to {output_file}')

def plot_error_histogram(errors, output_file, dataset_name):
    """
    Plot histogram of distance errors
    
    Args:
        errors: Array of distance errors
        output_file: Path to save the plot
        dataset_name: Name of the dataset
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    plt.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    
    # Add vertical line for median
    median = np.median(errors)
    plt.axvline(x=median, color='r', linestyle='--', linewidth=2, label=f'Median: {median:.2f} km')
    
    # Add vertical line for mean
    mean = np.mean(errors)
    plt.axvline(x=mean, color='g', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f} km')
    
    # Add labels and title
    plt.xlabel('Distance Error (km)')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Distance Errors - {dataset_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Error histogram saved to {output_file}')

def create_comparison_table(datasets, gcn_metrics, mlp_metrics, output_file):
    """
    Create comparison table of GCN and MLP metrics
    
    Args:
        datasets: List of dataset names
        gcn_metrics: List of dictionaries containing GCN metrics
        mlp_metrics: List of dictionaries containing MLP metrics
        output_file: Path to save the table
    """
    # Create DataFrame
    data = []
    
    for i, dataset in enumerate(datasets):
        row = {
            'Dataset': dataset,
            'GCN MSE': gcn_metrics[i].get('mse', 'N/A'),
            'MLP MSE': mlp_metrics[i].get('mse', 'N/A'),
            'GCN MAE': gcn_metrics[i].get('mae', 'N/A'),
            'MLP MAE': mlp_metrics[i].get('mae', 'N/A'),
            'GCN Median': gcn_metrics[i].get('median', 'N/A'),
            'MLP Median': mlp_metrics[i].get('median', 'N/A')
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f'Comparison table saved to {output_file}')
    
    # Return formatted markdown table
    markdown_table = df.to_markdown(index=False)
    return markdown_table

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze test results of the optimized GCN model')
    parser.add_argument('--dataset', type=str, default='New_York', help='Dataset name')
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs('asset/figures', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Parse test log file
    log_file = f'asset/log/{args.dataset}_test_gcn_optimized.log'
    if os.path.exists(log_file):
        metrics = parse_test_log_file(log_file)
        
        # Print metrics
        print(f'Test metrics for {args.dataset}:')
        for key, value in metrics.items():
            if key != 'errors':
                print(f'  {key}: {value}')
        
        # Plot error CDF if errors are available
        if 'errors' in metrics:
            output_file = f'asset/figures/{args.dataset}_distance_cdf_gcn_optimized.png'
            plot_error_cdf(metrics['errors'], output_file, args.dataset)
            
            # Plot error histogram
            output_file = f'asset/figures/{args.dataset}_distance_histogram_gcn_optimized.png'
            plot_error_histogram(metrics['errors'], output_file, args.dataset)
        
        # Create comparison table for all datasets
        datasets = ['New_York', 'Shanghai', 'Los_Angeles']
        gcn_metrics = []
        
        # MLP metrics (from previous results)
        mlp_metrics = [
            {'mse': 3.51, 'mae': 1.49, 'median': 224.83},
            {'mse': 7859.51, 'mae': 76.32, 'median': 12953.86},
            {'mse': 7569.97, 'mae': 76.15, 'median': 12573.21}
        ]
        
        # Parse GraphTransGeo metrics for all datasets
        for dataset in datasets:
            log_file = f'asset/log/{dataset}_test_gcn_optimized.log'
            if os.path.exists(log_file):
                gcn_metrics.append(parse_test_log_file(log_file))
            else:
                gcn_metrics.append({})
        
        # Create comparison table
        output_file = 'reports/model_comparison.csv'
        markdown_table = create_comparison_table(datasets, gcn_metrics, mlp_metrics, output_file)
        
        # Print markdown table
        print('\nModel Comparison:')
        print(markdown_table)
    else:
        print(f'Test log file not found: {log_file}')

if __name__ == '__main__':
    main()
