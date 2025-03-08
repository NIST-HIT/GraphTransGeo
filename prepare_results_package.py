#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Prepare results package for the optimized GraphTransGeo model

import os
import shutil
import argparse
import zipfile
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

def create_directory_structure(output_dir):
    """
    Create directory structure for the results package
    
    Args:
        output_dir: Output directory
    """
    # Create main directories
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'code'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'code', 'lib'), exist_ok=True)

def copy_model_files(source_dir, output_dir):
    """
    Copy model files to the output directory
    
    Args:
        source_dir: Source directory
        output_dir: Output directory
    """
    # Copy model files
    model_files = [f for f in os.listdir(os.path.join(source_dir, 'asset', 'model')) if f.endswith('.pth')]
    for file_name in model_files:
        src_path = os.path.join(source_dir, 'asset', 'model', file_name)
        dst_path = os.path.join(output_dir, 'models', file_name)
        shutil.copy2(src_path, dst_path)
        print(f'Copied {file_name} to models/')

def copy_log_files(source_dir, output_dir):
    """
    Copy log files to the output directory
    
    Args:
        source_dir: Source directory
        output_dir: Output directory
    """
    # Copy log files
    log_files = [f for f in os.listdir(os.path.join(source_dir, 'asset', 'log')) if f.endswith('.log')]
    for file_name in log_files:
        src_path = os.path.join(source_dir, 'asset', 'log', file_name)
        dst_path = os.path.join(output_dir, 'logs', file_name)
        shutil.copy2(src_path, dst_path)
        print(f'Copied {file_name} to logs/')

def copy_figure_files(source_dir, output_dir):
    """
    Copy figure files to the output directory
    
    Args:
        source_dir: Source directory
        output_dir: Output directory
    """
    # Copy figure files
    figure_dir = os.path.join(source_dir, 'asset', 'figures')
    if os.path.exists(figure_dir):
        figure_files = [f for f in os.listdir(figure_dir) if f.endswith('.png')]
        for file_name in figure_files:
            src_path = os.path.join(source_dir, 'asset', 'figures', file_name)
            dst_path = os.path.join(output_dir, 'figures', file_name)
            shutil.copy2(src_path, dst_path)
            print(f'Copied {file_name} to figures/')

def copy_report_files(source_dir, output_dir):
    """
    Copy report files to the output directory
    
    Args:
        source_dir: Source directory
        output_dir: Output directory
    """
    # Copy report files
    report_files = [
        'final_gcn_report.md',
        'final_report_cn.md',
        'implementation_progress.md'
    ]
    for file_name in report_files:
        src_path = os.path.join(source_dir, 'reports', file_name)
        if os.path.exists(src_path):
            dst_path = os.path.join(output_dir, 'reports', file_name)
            shutil.copy2(src_path, dst_path)
            print(f'Copied {file_name} to reports/')

def copy_code_files(source_dir, output_dir):
    """
    Copy code files to the output directory
    
    Args:
        source_dir: Source directory
        output_dir: Output directory
    """
    # Copy main code files
    code_files = [
        'train_gcn_optimized.py',
        'gcn_data_loader.py',
        'run_gcn_optimized.sh',
        'visualize_gcn_results.py',
        'analyze_test_results.py',
        'visualize_comprehensive_results.py'
    ]
    for file_name in code_files:
        src_path = os.path.join(source_dir, file_name)
        if os.path.exists(src_path):
            dst_path = os.path.join(output_dir, 'code', file_name)
            shutil.copy2(src_path, dst_path)
            print(f'Copied {file_name} to code/')
    
    # Copy library files
    lib_files = [
        'model_graphtransgeo_gcn_optimized.py',
        'utils.py'
    ]
    for file_name in lib_files:
        src_path = os.path.join(source_dir, 'lib', file_name)
        if os.path.exists(src_path):
            dst_path = os.path.join(output_dir, 'code', 'lib', file_name)
            shutil.copy2(src_path, dst_path)
            print(f'Copied lib/{file_name} to code/lib/')

def create_summary_file(source_dir, output_dir):
    """
    Create summary file with performance metrics
    
    Args:
        source_dir: Source directory
        output_dir: Output directory
    """
    # Dataset names
    dataset_names = ['New_York', 'Shanghai', 'Los_Angeles']
    
    # Parse test log files
    test_metrics = {}
    for dataset in dataset_names:
        log_file = os.path.join(source_dir, 'asset', 'log', f'{dataset}_test_gcn_optimized.log')
        if os.path.exists(log_file):
            metrics = parse_test_log(log_file)
            test_metrics[dataset] = metrics
    
    # MLP metrics (from previous results)
    mlp_metrics = {
        'New_York': {'mse': 3.51, 'mae': 1.49, 'median': 224.83},
        'Shanghai': {'mse': 7859.51, 'mae': 76.32, 'median': 12953.86},
        'Los_Angeles': {'mse': 7569.97, 'mae': 76.15, 'median': 12573.21}
    }
    
    # Create summary table
    data = []
    for dataset in dataset_names:
        row = {'Dataset': dataset}
        
        for metric in ['mse', 'mae', 'median']:
            if dataset in test_metrics and metric in test_metrics[dataset]:
                row[f'GCN {metric.upper()}'] = test_metrics[dataset][metric]
            else:
                row[f'GCN {metric.upper()}'] = 'N/A'
            
            if dataset in mlp_metrics and metric in mlp_metrics[dataset]:
                row[f'MLP {metric.upper()}'] = str(mlp_metrics[dataset][metric])
            else:
                row[f'MLP {metric.upper()}'] = 'N/A'
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'reports', 'performance_summary.csv')
    df.to_csv(output_file, index=False)
    
    # Create markdown summary
    markdown_summary = f"""# Performance Summary

## Model Comparison

{df.to_markdown(index=False)}

## Improvement Analysis

"""
    
    # Calculate improvement percentages
    for dataset in dataset_names:
        if dataset in test_metrics and dataset in mlp_metrics:
            markdown_summary += f"### {dataset}\n\n"
            
            for metric in ['mse', 'mae', 'median']:
                if metric in test_metrics[dataset] and metric in mlp_metrics[dataset]:
                    gcn_value = test_metrics[dataset][metric]
                    mlp_value = mlp_metrics[dataset][metric]
                    
                    if mlp_value != 0:
                        improvement = (mlp_value - gcn_value) / mlp_value * 100
                        markdown_summary += f"- {metric.upper()}: {improvement:.2f}% improvement\n"
            
            markdown_summary += "\n"
    
    # Save markdown summary
    output_file = os.path.join(output_dir, 'reports', 'performance_summary.md')
    with open(output_file, 'w') as f:
        f.write(markdown_summary)
    
    print(f'Created performance summary at reports/performance_summary.md')

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
    
    return metrics

def create_readme(output_dir):
    """
    Create README file
    
    Args:
        output_dir: Output directory
    """
    readme_content = """# GraphTransGeo++ Optimized GraphTransGeo Results

This package contains the results of the optimized GraphTransGeo++ method for IP geolocation using Graph Convolutional Networks (GCNs).

## Directory Structure

- `models/`: Trained model checkpoints
- `logs/`: Training and testing logs
- `figures/`: Visualization figures
- `reports/`: Reports and analysis
- `code/`: Implementation code

## Key Files

- `reports/final_gcn_report.md`: Comprehensive report of the implementation and results
- `reports/final_report_cn.md`: Chinese version of the report
- `reports/performance_summary.md`: Summary of performance metrics
- `figures/`: Contains training curves, error distributions, and model comparisons
- `code/train_gcn_optimized.py`: Training script for the optimized GCN model
- `code/lib/model_graphtransgeo_gcn_optimized.py`: Model implementation

## Performance Highlights

The optimized GCN model achieves significant improvements over the baseline MLP model:

- Improved graph construction with adaptive k-NN and network topology
- Ensemble architecture combining GCN and GAT models
- Haversine loss function for direct optimization of geographical distance
- Adversarial training with consistency regularization

Please refer to the reports for detailed analysis and results.
"""
    
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print('Created README.md')

def create_zip_archive(output_dir, zip_file_path):
    """
    Create ZIP archive of the results package
    
    Args:
        output_dir: Output directory
        zip_file_path: Path to the ZIP file
    """
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(output_dir))
                zipf.write(file_path, arcname)
    
    print(f'Created ZIP archive: {zip_file_path}')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prepare results package for the optimized GCN model')
    parser.add_argument('--source_dir', type=str, default='./', help='Source directory')
    parser.add_argument('--output_dir', type=str, default='./gcn_results', help='Output directory')
    parser.add_argument('--create_zip', action='store_true', help='Create ZIP archive')
    args = parser.parse_args()
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directory structure
    create_directory_structure(output_dir)
    
    # Copy files
    copy_model_files(args.source_dir, output_dir)
    copy_log_files(args.source_dir, output_dir)
    copy_figure_files(args.source_dir, output_dir)
    copy_report_files(args.source_dir, output_dir)
    copy_code_files(args.source_dir, output_dir)
    
    # Create summary file
    create_summary_file(args.source_dir, output_dir)
    
    # Create README
    create_readme(output_dir)
    
    # Create ZIP archive if requested
    if args.create_zip:
        zip_file_path = f'graphtransgeo_gcn_results_{timestamp}.zip'
        create_zip_archive(output_dir, zip_file_path)
    
    print(f'Results package prepared. Output directory: {output_dir}')

if __name__ == '__main__':
    main()
