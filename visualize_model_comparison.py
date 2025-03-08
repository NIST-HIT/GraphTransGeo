#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Visualize model comparison between GraphTransGeo and MLP models

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse

def load_comparison_data(csv_file):
    """
    Load comparison data from CSV file
    
    Args:
        csv_file: Path to CSV file
        
    Returns:
        df: DataFrame with comparison data
    """
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        return df
    else:
        print(f"Comparison data file not found: {csv_file}")
        return None

def create_performance_comparison_plots(df, output_dir, english=True):
    """
    Create performance comparison plots between GCN and MLP models
    
    Args:
        df: DataFrame with comparison data
        output_dir: Directory to save plots
        english: Whether to use English labels (True) or Chinese labels (False)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set font for Chinese characters if needed
    if not english:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # For Chinese characters
        plt.rcParams['axes.unicode_minus'] = False  # For minus sign
    
    # Extract data
    datasets = df['Dataset'].tolist()
    
    # Translate dataset names if using Chinese
    if not english:
        dataset_names_cn = {'New_York': '纽约', 'Shanghai': '上海', 'Los_Angeles': '洛杉矶'}
        datasets_cn = [dataset_names_cn.get(d, d) for d in datasets]
        datasets_display = datasets_cn
    else:
        datasets_display = datasets
    
    # Extract metrics
    gcn_mse = df['GCN MSE'].tolist()
    mlp_mse = df['MLP MSE'].tolist()
    gcn_mae = df['GCN MAE'].tolist()
    mlp_mae = df['MLP MAE'].tolist()
    gcn_median = df['GCN Median'].tolist()
    mlp_median = df['MLP Median'].tolist()
    
    # Set up figure size and style
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Create MSE comparison plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, gcn_mse, width, label='GCN', color='#1f77b4', alpha=0.8)
    bars2 = plt.bar(x + width/2, mlp_mse, width, label='MLP', color='#ff7f0e', alpha=0.8)
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.yscale('log')
    if english:
        plt.xlabel('Datasets')
        plt.ylabel('MSE (log scale)')
        plt.title('MSE Comparison Across Datasets')
    else:
        plt.xlabel('数据集')
        plt.ylabel('均方误差 (MSE, 对数刻度)')
        plt.title('不同数据集上的均方误差对比')
    
    plt.xticks(x, datasets_display)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    # Save MSE plot
    if english:
        plt.savefig(f"{output_dir}/mse_comparison.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"{output_dir}/mse_comparison_cn.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create MAE comparison plot
    plt.figure(figsize=(12, 6))
    
    bars1 = plt.bar(x - width/2, gcn_mae, width, label='GCN', color='#1f77b4', alpha=0.8)
    bars2 = plt.bar(x + width/2, mlp_mae, width, label='MLP', color='#ff7f0e', alpha=0.8)
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.yscale('log')
    if english:
        plt.xlabel('Datasets')
        plt.ylabel('MAE (log scale)')
        plt.title('MAE Comparison Across Datasets')
    else:
        plt.xlabel('数据集')
        plt.ylabel('平均绝对误差 (MAE, 对数刻度)')
        plt.title('不同数据集上的平均绝对误差对比')
    
    plt.xticks(x, datasets_display)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    # Save MAE plot
    if english:
        plt.savefig(f"{output_dir}/mae_comparison.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"{output_dir}/mae_comparison_cn.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create Median Distance Error comparison plot
    plt.figure(figsize=(12, 6))
    
    bars1 = plt.bar(x - width/2, gcn_median, width, label='GCN', color='#1f77b4', alpha=0.8)
    bars2 = plt.bar(x + width/2, mlp_median, width, label='MLP', color='#ff7f0e', alpha=0.8)
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.yscale('log')
    if english:
        plt.xlabel('Datasets')
        plt.ylabel('Median Distance Error (km, log scale)')
        plt.title('Median Distance Error Comparison Across Datasets')
    else:
        plt.xlabel('数据集')
        plt.ylabel('中位距离误差 (km, 对数刻度)')
        plt.title('不同数据集上的中位距离误差对比')
    
    plt.xticks(x, datasets_display)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    # Save Median plot
    if english:
        plt.savefig(f"{output_dir}/median_comparison.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"{output_dir}/median_comparison_cn.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create improvement percentage plot
    plt.figure(figsize=(12, 6))
    
    # Calculate improvement percentages
    mse_improvement = [(mlp - gcn) / mlp * 100 for mlp, gcn in zip(mlp_mse, gcn_mse)]
    mae_improvement = [(mlp - gcn) / mlp * 100 for mlp, gcn in zip(mlp_mae, gcn_mae)]
    median_improvement = [(mlp - gcn) / mlp * 100 for mlp, gcn in zip(mlp_median, gcn_median)]
    
    # Create a DataFrame for the improvement data
    improvement_data = {
        'Dataset': datasets_display,
        'MSE Improvement (%)': mse_improvement,
        'MAE Improvement (%)': mae_improvement,
        'Median Improvement (%)': median_improvement
    }
    improvement_df = pd.DataFrame(improvement_data)
    
    # Melt the DataFrame for easier plotting
    melted_df = pd.melt(improvement_df, id_vars=['Dataset'], 
                        value_vars=['MSE Improvement (%)', 'MAE Improvement (%)', 'Median Improvement (%)'],
                        var_name='Metric', value_name='Improvement (%)')
    
    # Create the grouped bar chart
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Dataset', y='Improvement (%)', hue='Metric', data=melted_df)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom',
                    xytext = (0, 5), textcoords = 'offset points')
    
    if english:
        plt.title('Performance Improvement of GraphTransGeo over MLP (%)')
        plt.xlabel('Datasets')
        plt.ylabel('Improvement (%)')
    else:
        plt.title('GraphTransGeo相比MLP的性能提升 (%)')
        plt.xlabel('数据集')
        plt.ylabel('提升百分比 (%)')
    
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save improvement plot
    if english:
        plt.savefig(f"{output_dir}/improvement_comparison.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"{output_dir}/improvement_comparison_cn.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance comparison plots saved to {output_dir}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize model comparison between GCN and MLP models')
    parser.add_argument('--csv_file', type=str, default='reports/model_comparison.csv', help='Path to CSV file with comparison data')
    parser.add_argument('--output_dir', type=str, default='asset/figures', help='Directory to save plots')
    parser.add_argument('--chinese', action='store_true', help='Use Chinese labels')
    args = parser.parse_args()
    
    # Load comparison data
    df = load_comparison_data(args.csv_file)
    
    if df is not None:
        # Create performance comparison plots
        create_performance_comparison_plots(df, args.output_dir, english=not args.chinese)
        
        # Create Chinese version if not already created
        if not args.chinese:
            create_performance_comparison_plots(df, args.output_dir, english=False)

if __name__ == '__main__':
    main()
