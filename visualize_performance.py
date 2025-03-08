#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Visualize performance metrics for GraphTransGeo and MLP models across datasets

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_output_dir():
    """Create output directory for figures"""
    output_dir = 'asset/figures/comparison'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def create_comparison_plot(metric_name, gcn_values, mlp_values, output_file, log_scale=True, cn=False):
    """
    Create comparison bar plot for a specific metric
    
    Args:
        metric_name: Name of the metric
        gcn_values: List of GCN values
        mlp_values: List of MLP values
        output_file: Path to save the plot
        log_scale: Whether to use log scale
        cn: Whether to use Chinese labels
    """
    plt.figure(figsize=(12, 6))
    
    # Data
    datasets = ['New York', 'Los Angeles', 'Shanghai']
    
    # Filter out None values
    valid_indices = [i for i, v in enumerate(gcn_values) if v is not None]
    valid_datasets = [datasets[i] for i in valid_indices]
    valid_gcn = [gcn_values[i] for i in valid_indices]
    valid_mlp = [mlp_values[i] for i in valid_indices]
    
    x = np.arange(len(valid_datasets))
    width = 0.35
    
    plt.bar(x - width/2, valid_gcn, width, label='GraphTransGeo', color='blue', alpha=0.7)
    plt.bar(x + width/2, valid_mlp, width, label='MLP', color='orange', alpha=0.7)
    
    if log_scale and min(valid_gcn + valid_mlp) > 0:
        plt.yscale('log')
    
    if cn:
        plt.xlabel('数据集')
        if metric_name == 'MSE':
            plt.ylabel('均方误差 (MSE)')
            plt.title('GraphTransGeo与MLP模型均方误差对比')
        elif metric_name == 'MAE':
            plt.ylabel('平均绝对误差 (MAE)')
            plt.title('GraphTransGeo与MLP模型平均绝对误差对比')
        else:
            plt.ylabel('中位距离误差 (km)')
            plt.title('GraphTransGeo与MLP模型中位距离误差对比')
        plt.legend(['GraphTransGeo模型', 'MLP模型'])
    else:
        plt.xlabel('Datasets')
        plt.ylabel(f'{metric_name}')
        plt.title(f'{metric_name} Comparison: GraphTransGeo vs MLP')
        plt.legend()
    
    plt.xticks(x, valid_datasets)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(valid_gcn):
        plt.text(i - width/2, v * 1.1, f'{v:.2f}', ha='center', va='bottom', fontsize=9, rotation=0)
    
    for i, v in enumerate(valid_mlp):
        plt.text(i + width/2, v * 1.1, f'{v:.2f}', ha='center', va='bottom', fontsize=9, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'{metric_name} comparison saved to {output_file}')

def plot_training_curves(log_file, output_file, cn=False):
    """
    Plot training and validation curves
    
    Args:
        log_file: Path to training log file
        output_file: Path to save the plot
        cn: Whether to use Chinese labels
    """
    # Parse log file
    epochs = []
    train_loss = []
    val_loss = []
    train_mse = []
    val_mse = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if line.startswith('Epoch'):
                parts = line.strip().split(' - ')
                epoch_part = parts[0].split('/')[0].replace('Epoch ', '')
                epochs.append(int(epoch_part))
                
                metrics = parts[1].split(', ')
                train_loss.append(float(metrics[0].split(': ')[1]))
                train_mse.append(float(metrics[1].split(': ')[1]))
                val_loss.append(float(metrics[3].split(': ')[1]))
                val_mse.append(float(metrics[4].split(': ')[1]))
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss, 'b-', label='Train Loss' if not cn else '训练损失')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss' if not cn else '验证损失')
    if cn:
        plt.title('训练和验证损失曲线')
        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
    else:
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot MSE
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_mse, 'b-', label='Train MSE' if not cn else '训练MSE')
    plt.plot(epochs, val_mse, 'r-', label='Validation MSE' if not cn else '验证MSE')
    if cn:
        plt.title('训练和验证MSE曲线')
        plt.xlabel('训练轮次')
        plt.ylabel('均方误差 (MSE)')
    else:
        plt.title('Training and Validation MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved to {output_file}')

def plot_cross_dataset_performance(output_file, cn=False):
    """
    Plot cross-dataset performance comparison
    
    Args:
        output_file: Path to save the plot
        cn: Whether to use Chinese labels
    """
    plt.figure(figsize=(10, 6))
    
    # Data
    metrics = ['MSE', 'MAE', 'Median Error (km)']
    ny_performance = [1.0, 1.0, 1.0]  # Normalized to 1.0 for New York
    la_performance = [
        2352.65/452.49 if 452.49 > 0 else 0,  # MSE ratio
        35.35/19.44 if 19.44 > 0 else 0,      # MAE ratio
        6280.91/2826.96 if 2826.96 > 0 else 0 # Median ratio
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, ny_performance, width, label='New York', color='blue', alpha=0.7)
    plt.bar(x + width/2, la_performance, width, label='Los Angeles', color='green', alpha=0.7)
    
    if cn:
        plt.xlabel('评估指标')
        plt.ylabel('相对性能 (相对于纽约)')
        plt.title('GraphTransGeo模型跨数据集性能对比')
        plt.legend(['纽约', '洛杉矶'])
    else:
        plt.xlabel('Metrics')
        plt.ylabel('Relative Performance (normalized to New York)')
        plt.title('GraphTransGeo Cross-Dataset Performance')
        plt.legend()
    
    plt.xticks(x, metrics)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(ny_performance):
        plt.text(i - width/2, v * 1.1, f'{v:.2f}x', ha='center', va='bottom', fontsize=9)
    
    for i, v in enumerate(la_performance):
        plt.text(i + width/2, v * 1.1, f'{v:.2f}x', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Cross-dataset performance visualization saved to {output_file}')

def create_performance_improvement_heatmap(output_file, cn=False):
    """
    Create heatmap showing performance improvement of GCN over MLP
    
    Args:
        output_file: Path to save the plot
        cn: Whether to use Chinese labels
    """
    # Data
    datasets = ['New York', 'Los Angeles']
    metrics = ['MSE', 'MAE', 'Median Error']
    
    # Performance improvement (negative means GraphTransGeo is worse)
    improvement = np.array([
        [-452.49/3.51 + 1, -19.44/1.49 + 1, -2826.96/224.83 + 1],  # New York (GraphTransGeo is worse)
        [1 - 2352.65/7569.97, 1 - 35.35/76.15, 1 - 6280.91/12573.21]  # Los Angeles (GraphTransGeo is better)
    ]) * 100  # Convert to percentage
    
    # Create DataFrame
    df = pd.DataFrame(improvement, index=datasets, columns=metrics)
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap='RdYlGn', center=0, fmt='.1f', 
                cbar_kws={'label': 'Performance Improvement (%)' if not cn else '相对于MLP的性能提升 (%)'},
                linewidths=0.5)
    
    if cn:
        plt.title('GraphTransGeo相对于MLP的性能提升')
    else:
        plt.title('Performance Improvement of GraphTransGeo Relative to MLP (%)')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Performance improvement heatmap saved to {output_file}')

def main():
    # Create output directory
    output_dir = create_output_dir()
    
    # Data for comparison
    datasets = ['New York', 'Los Angeles', 'Shanghai']
    gcn_mse = [452.49, 2352.65, None]
    mlp_mse = [3.51, 7569.97, 7859.51]
    gcn_mae = [19.44, 35.35, None]
    mlp_mae = [1.49, 76.15, 76.32]
    gcn_median = [2826.96, 6280.91, None]
    mlp_median = [224.83, 12573.21, 12953.86]
    
    # Create comparison plots
    create_comparison_plot('MSE', gcn_mse, mlp_mse, os.path.join(output_dir, 'mse_comparison.png'))
    create_comparison_plot('MAE', gcn_mae, mlp_mae, os.path.join(output_dir, 'mae_comparison.png'))
    create_comparison_plot('Median Distance Error (km)', gcn_median, mlp_median, os.path.join(output_dir, 'median_comparison.png'))
    
    # Create Chinese versions
    create_comparison_plot('MSE', gcn_mse, mlp_mse, os.path.join(output_dir, 'mse_comparison_cn.png'), cn=True)
    create_comparison_plot('MAE', gcn_mae, mlp_mae, os.path.join(output_dir, 'mae_comparison_cn.png'), cn=True)
    create_comparison_plot('Median Distance Error (km)', gcn_median, mlp_median, os.path.join(output_dir, 'median_comparison_cn.png'), cn=True)
    
    # Plot training curves if log file exists
    if os.path.exists('asset/log/New_York_training_gcn_optimized.log'):
        plot_training_curves('asset/log/New_York_training_gcn_optimized.log', 
                            os.path.join(output_dir, 'training_curves.png'))
        plot_training_curves('asset/log/New_York_training_gcn_optimized.log', 
                            os.path.join(output_dir, 'training_curves_cn.png'), cn=True)
    
    # Create cross-dataset performance visualizations
    plot_cross_dataset_performance(os.path.join(output_dir, 'cross_dataset_performance.png'))
    plot_cross_dataset_performance(os.path.join(output_dir, 'cross_dataset_performance_cn.png'), cn=True)
    
    # Create performance improvement heatmap
    create_performance_improvement_heatmap(os.path.join(output_dir, 'performance_improvement_heatmap.png'))
    create_performance_improvement_heatmap(os.path.join(output_dir, 'performance_improvement_heatmap_cn.png'), cn=True)

if __name__ == '__main__':
    main()
