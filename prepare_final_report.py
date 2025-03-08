#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Prepare final report for the optimized GraphTransGeo model

import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

def update_final_report(dataset_names, output_dir):
    """
    Update final report with actual values
    
    Args:
        dataset_names: List of dataset names
        output_dir: Output directory
    """
    # Parse test logs for GraphTransGeo
    gcn_metrics = {}
    for dataset in dataset_names:
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
    
    # Parse training log for New York
    training_data = {}
    log_file = f'asset/log/New_York_training_gcn_optimized.log'
    if os.path.exists(log_file):
        epochs, train_losses, train_mse_losses, train_haversine_losses, val_losses, val_mse_losses, val_haversine_losses = parse_training_log(log_file)
        
        training_data = {
            'epochs': epochs,
            'train_losses': train_losses,
            'train_mse_losses': train_mse_losses,
            'train_haversine_losses': train_haversine_losses,
            'val_losses': val_losses,
            'val_mse_losses': val_mse_losses,
            'val_haversine_losses': val_haversine_losses
        }
    
    # Update English report
    update_english_report(gcn_metrics, mlp_metrics, training_data, output_dir)
    
    # Update Chinese report
    update_chinese_report(gcn_metrics, mlp_metrics, training_data, output_dir)

def update_english_report(gcn_metrics, mlp_metrics, training_data, output_dir):
    """
    Update English report with actual values
    
    Args:
        gcn_metrics: Dictionary of GCN metrics
        mlp_metrics: Dictionary of MLP metrics
        training_data: Dictionary of training data
        output_dir: Output directory
    """
    report_file = os.path.join(output_dir, 'reports', 'final_gcn_report.md')
    
    if not os.path.exists(report_file):
        print(f'Report file not found: {report_file}')
        return
    
    with open(report_file, 'r') as f:
        content = f.read()
    
    # Update training epochs
    if training_data and 'epochs' in training_data and len(training_data['epochs']) > 0:
        final_epoch = training_data['epochs'][-1]
        content = re.sub(r'The model converged after X epochs', f'The model converged after {final_epoch} epochs', content)
        
        # Update validation loss
        if 'val_losses' in training_data and len(training_data['val_losses']) > 0:
            final_val_loss = training_data['val_losses'][-1]
            content = re.sub(r'final validation loss of Y', f'final validation loss of {final_val_loss:.2f}', content)
    
    # Update performance metrics table
    for dataset in gcn_metrics:
        if 'mse' in gcn_metrics[dataset]:
            content = re.sub(f'\\| {dataset} \\| X ', f'| {dataset} | {gcn_metrics[dataset]["mse"]:.2f} ', content)
        
        if 'mae' in gcn_metrics[dataset]:
            content = re.sub(f'\\| {dataset} \\|.*?\\| Y ', f'| {dataset} | {gcn_metrics[dataset]["mse"]:.2f} | {gcn_metrics[dataset]["mae"]:.2f} ', content)
        
        if 'median' in gcn_metrics[dataset]:
            content = re.sub(f'\\| {dataset} \\|.*?\\| Z \\|', f'| {dataset} | {gcn_metrics[dataset]["mse"]:.2f} | {gcn_metrics[dataset]["mae"]:.2f} | {gcn_metrics[dataset]["median"]:.2f} |', content)
    
    # Update comparison tables
    for dataset in gcn_metrics:
        if 'mse' in gcn_metrics[dataset]:
            content = re.sub(f'\\| GCN \\(Optimized\\) \\| X ', f'| GCN (Optimized) | {gcn_metrics[dataset]["mse"]:.2f} ', content)
        
        if 'mae' in gcn_metrics[dataset]:
            content = re.sub(f'\\| GCN \\(Optimized\\) \\|.*?\\| Y ', f'| GCN (Optimized) | {gcn_metrics[dataset]["mse"]:.2f} | {gcn_metrics[dataset]["mae"]:.2f} ', content)
        
        if 'median' in gcn_metrics[dataset]:
            content = re.sub(f'\\| GCN \\(Optimized\\) \\|.*?\\| Z \\|', f'| GCN (Optimized) | {gcn_metrics[dataset]["mse"]:.2f} | {gcn_metrics[dataset]["mae"]:.2f} | {gcn_metrics[dataset]["median"]:.2f} |', content)
    
    # Update error CDF
    if 'New_York' in gcn_metrics and 'median' in gcn_metrics['New_York']:
        median = gcn_metrics['New_York']['median']
        content = re.sub(r'X% of predictions have an error less than Y km', f'50% of predictions have an error less than {median:.2f} km', content)
    
    # Write updated content
    with open(report_file, 'w') as f:
        f.write(content)
    
    print(f'Updated English report: {report_file}')

def update_chinese_report(gcn_metrics, mlp_metrics, training_data, output_dir):
    """
    Update Chinese report with actual values
    
    Args:
        gcn_metrics: Dictionary of GCN metrics
        mlp_metrics: Dictionary of MLP metrics
        training_data: Dictionary of training data
        output_dir: Output directory
    """
    report_file = os.path.join(output_dir, 'reports', 'final_report_cn.md')
    
    if not os.path.exists(report_file):
        print(f'Report file not found: {report_file}')
        return
    
    with open(report_file, 'r') as f:
        content = f.read()
    
    # Update training epochs
    if training_data and 'epochs' in training_data and len(training_data['epochs']) > 0:
        final_epoch = training_data['epochs'][-1]
        content = re.sub(r'模型在X轮后收敛', f'模型在{final_epoch}轮后收敛', content)
        
        # Update validation loss
        if 'val_losses' in training_data and len(training_data['val_losses']) > 0:
            final_val_loss = training_data['val_losses'][-1]
            content = re.sub(r'最终验证损失为Y', f'最终验证损失为{final_val_loss:.2f}', content)
    
    # Update performance metrics table
    dataset_names_cn = {
        'New_York': '纽约',
        'Shanghai': '上海',
        'Los_Angeles': '洛杉矶'
    }
    
    for dataset, dataset_cn in dataset_names_cn.items():
        if dataset in gcn_metrics and 'mse' in gcn_metrics[dataset]:
            content = re.sub(f'\\| {dataset_cn} \\| X ', f'| {dataset_cn} | {gcn_metrics[dataset]["mse"]:.2f} ', content)
        
        if dataset in gcn_metrics and 'mae' in gcn_metrics[dataset]:
            content = re.sub(f'\\| {dataset_cn} \\|.*?\\| Y ', f'| {dataset_cn} | {gcn_metrics[dataset]["mse"]:.2f} | {gcn_metrics[dataset]["mae"]:.2f} ', content)
        
        if dataset in gcn_metrics and 'median' in gcn_metrics[dataset]:
            content = re.sub(f'\\| {dataset_cn} \\|.*?\\| Z \\|', f'| {dataset_cn} | {gcn_metrics[dataset]["mse"]:.2f} | {gcn_metrics[dataset]["mae"]:.2f} | {gcn_metrics[dataset]["median"]:.2f} |', content)
    
    # Update comparison tables
    for dataset, dataset_cn in dataset_names_cn.items():
        if dataset in gcn_metrics and 'mse' in gcn_metrics[dataset]:
            content = re.sub(f'\\| GCN \\(优化\\) \\| X ', f'| GCN (优化) | {gcn_metrics[dataset]["mse"]:.2f} ', content)
        
        if dataset in gcn_metrics and 'mae' in gcn_metrics[dataset]:
            content = re.sub(f'\\| GCN \\(优化\\) \\|.*?\\| Y ', f'| GCN (优化) | {gcn_metrics[dataset]["mse"]:.2f} | {gcn_metrics[dataset]["mae"]:.2f} ', content)
        
        if dataset in gcn_metrics and 'median' in gcn_metrics[dataset]:
            content = re.sub(f'\\| GCN \\(优化\\) \\|.*?\\| Z \\|', f'| GCN (优化) | {gcn_metrics[dataset]["mse"]:.2f} | {gcn_metrics[dataset]["mae"]:.2f} | {gcn_metrics[dataset]["median"]:.2f} |', content)
    
    # Update error CDF
    if 'New_York' in gcn_metrics and 'median' in gcn_metrics['New_York']:
        median = gcn_metrics['New_York']['median']
        content = re.sub(r'X%的预测误差小于Y公里', f'50%的预测误差小于{median:.2f}公里', content)
    
    # Write updated content
    with open(report_file, 'w') as f:
        f.write(content)
    
    print(f'Updated Chinese report: {report_file}')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prepare final report for the optimized GCN model')
    parser.add_argument('--output_dir', type=str, default='./', help='Output directory')
    args = parser.parse_args()
    
    # Dataset names
    dataset_names = ['New_York', 'Shanghai', 'Los_Angeles']
    
    # Update final report
    update_final_report(dataset_names, args.output_dir)

if __name__ == '__main__':
    main()
