#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Visualize cross-dataset comparison for GraphTransGeo and MLP models

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_mse_comparison(output_file):
    """
    Plot MSE comparison across datasets for GCN and MLP models
    
    Args:
        output_file: Path to save the plot
    """
    # Data
    datasets = ['New York', 'Shanghai', 'Los Angeles']
    gcn_mse = [3272.84, 15676.42, 15198.00]
    mlp_mse = [3.51, 7859.51, 7569.97]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    x = np.arange(len(datasets))
    width = 0.35
    
    plt.bar(x - width/2, gcn_mse, width, label='GCN')
    plt.bar(x + width/2, mlp_mse, width, label='MLP')
    
    plt.yscale('log')
    plt.xlabel('Datasets')
    plt.ylabel('MSE (log scale)')
    plt.title('MSE Comparison Across Datasets')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(True, axis='y')
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_mae_comparison(output_file):
    """
    Plot MAE comparison across datasets for GCN and MLP models
    
    Args:
        output_file: Path to save the plot
    """
    # Data
    datasets = ['New York', 'Shanghai', 'Los Angeles']
    gcn_mae = [74.90, 152.58, 152.48]
    mlp_mae = [1.49, 76.32, 76.15]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    x = np.arange(len(datasets))
    width = 0.35
    
    plt.bar(x - width/2, gcn_mae, width, label='GCN')
    plt.bar(x + width/2, mlp_mae, width, label='MLP')
    
    plt.yscale('log')
    plt.xlabel('Datasets')
    plt.ylabel('MAE (log scale)')
    plt.title('MAE Comparison Across Datasets')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(True, axis='y')
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_median_error_comparison(output_file):
    """
    Plot median distance error comparison across datasets for GCN and MLP models
    
    Args:
        output_file: Path to save the plot
    """
    # Data
    datasets = ['New York', 'Shanghai', 'Los Angeles']
    gcn_median = [5274.75, 12921.35, 12604.54]
    mlp_median = [224.83, 12953.86, 12573.21]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    x = np.arange(len(datasets))
    width = 0.35
    
    plt.bar(x - width/2, gcn_median, width, label='GCN')
    plt.bar(x + width/2, mlp_median, width, label='MLP')
    
    plt.xlabel('Datasets')
    plt.ylabel('Median Distance Error (km)')
    plt.title('Median Distance Error Comparison Across Datasets')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(True, axis='y')
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create output directory if it doesn't exist
    os.makedirs('reports/figures', exist_ok=True)
    
    # Plot MSE comparison
    plot_mse_comparison('reports/figures/mse_comparison.png')
    
    # Plot MAE comparison
    plot_mae_comparison('reports/figures/mae_comparison.png')
    
    # Plot median distance error comparison
    plot_median_error_comparison('reports/figures/median_error_comparison.png')
    
    print("Cross-dataset comparison visualizations saved to reports/figures/ directory.")

if __name__ == "__main__":
    main()
