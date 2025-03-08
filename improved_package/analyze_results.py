#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Analyze and compare GraphTransGeo and MLP model results

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def parse_training_log(log_file):
    """
    Parse training log file to extract epoch, train loss, and validation loss
    
    Args:
        log_file: Path to training log file
        
    Returns:
        epochs: List of epochs
        train_loss: List of training losses
        val_loss: List of validation losses
    """
    epochs = []
    train_loss = []
    train_mse = []
    val_loss = []
    val_mse = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Match the pattern: Epoch X/Y - Train Loss: A, Train MSE: B, Val Loss: C, Val MSE: D
            match = re.match(r'Epoch (\d+)/\d+ - Train Loss: ([\d\.]+), Train MSE: ([\d\.]+), Val Loss: ([\d\.]+), Val MSE: ([\d\.]+)', line)
            if match:
                epoch = int(match.group(1))
                t_loss = float(match.group(2))
                t_mse = float(match.group(3))
                v_loss = float(match.group(4))
                v_mse = float(match.group(5))
                
                epochs.append(epoch)
                train_loss.append(t_loss)
                train_mse.append(t_mse)
                val_loss.append(v_loss)
                val_mse.append(v_mse)
    
    return epochs, train_loss, train_mse, val_loss, val_mse

def parse_test_results(log_file):
    """
    Parse test results log file to extract MSE, MAE, and median distance error
    
    Args:
        log_file: Path to test results log file
        
    Returns:
        mse: Mean squared error
        mae: Mean absolute error
        median_error: Median distance error in km
    """
    mse = None
    mae = None
    median_error = None
    
    with open(log_file, 'r') as f:
        content = f.read()
        
        # Match MSE pattern
        mse_match = re.search(r'MSE: ([\d\.]+)', content)
        if mse_match:
            mse = float(mse_match.group(1))
        else:
            # Try alternative pattern for MLP logs
            mse_match = re.search(r'Loss: ([\d\.]+)', content)
            if mse_match:
                mse = float(mse_match.group(1))
        
        # Match MAE pattern
        mae_match = re.search(r'MAE: ([\d\.]+)', content)
        if mae_match:
            mae = float(mae_match.group(1))
        
        # Match median distance error pattern
        median_match = re.search(r'Median Distance Error: ([\d\.]+) km', content)
        if median_match:
            median_error = float(median_match.group(1))
        else:
            # Try alternative pattern for MLP logs
            median_match = re.search(r'Median Distance Error: ([\d\.]+)', content)
            if median_match:
                median_error = float(median_match.group(1))
    
    return mse, mae, median_error

def plot_training_curves(gcn_epochs, gcn_train_loss, gcn_val_loss, 
                         mlp_epochs, mlp_train_loss, mlp_val_loss,
                         output_file):
    """
    Plot training and validation loss curves for GCN and MLP models
    
    Args:
        gcn_epochs: List of epochs for GCN model
        gcn_train_loss: List of training losses for GCN model
        gcn_val_loss: List of validation losses for GCN model
        mlp_epochs: List of epochs for MLP model
        mlp_train_loss: List of training losses for MLP model
        mlp_val_loss: List of validation losses for MLP model
        output_file: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot GraphTransGeo curves
    plt.plot(gcn_epochs, gcn_train_loss, 'b-', label='GCN Train Loss')
    plt.plot(gcn_epochs, gcn_val_loss, 'b--', label='GCN Val Loss')
    
    # Plot MLP curves
    plt.plot(mlp_epochs, mlp_train_loss, 'r-', label='MLP Train Loss')
    plt.plot(mlp_epochs, mlp_val_loss, 'r--', label='MLP Val Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(output_file)
    plt.close()

def create_comparison_report(gcn_metrics, mlp_metrics, output_file):
    """
    Create a comparison report between GCN and MLP models
    
    Args:
        gcn_metrics: Dictionary containing GCN metrics
        mlp_metrics: Dictionary containing MLP metrics
        output_file: Path to save the report
    """
    # Check if metrics are available
    if gcn_metrics['mse'] is None or mlp_metrics['mse'] is None:
        mse_improvement = 0
    else:
        mse_improvement = (mlp_metrics['mse'] - gcn_metrics['mse']) / mlp_metrics['mse'] * 100 if mlp_metrics['mse'] != 0 else 0
    
    if gcn_metrics['mae'] is None or mlp_metrics['mae'] is None:
        mae_improvement = 0
    else:
        mae_improvement = (mlp_metrics['mae'] - gcn_metrics['mae']) / mlp_metrics['mae'] * 100 if mlp_metrics['mae'] != 0 else 0
    
    if gcn_metrics['median_error'] is None or mlp_metrics['median_error'] is None:
        median_improvement = 0
    else:
        median_improvement = (mlp_metrics['median_error'] - gcn_metrics['median_error']) / mlp_metrics['median_error'] * 100 if mlp_metrics['median_error'] != 0 else 0
    
    # Create report
    report = f"""# GraphTransGeo vs MLP Model Comparison Report

## Overview
This report compares the performance of the Graph Convolutional Network (GCN) model with the Multi-Layer Perceptron (MLP) model for IP geolocation using the GraphTransGeo++ approach.

## Model Architectures
- **GCN Model**: Graph Convolutional Network with adversarial training
- **MLP Model**: Multi-Layer Perceptron with adversarial training

## Training Parameters
- Learning Rate: 0.001
- Epsilon: 0.01
- Alpha: 0.01
- Beta: 0.5
- Hidden Dimension: 256
- Number of Layers (GCN): 2

## Performance Metrics

| Metric | GCN | MLP | Improvement |
|--------|-----|-----|-------------|
| MSE | {gcn_metrics['mse']:.2f} | {mlp_metrics['mse']:.2f} | {mse_improvement:.2f}% |
| MAE | {gcn_metrics['mae']:.2f} | {mlp_metrics['mae']:.2f} | {mae_improvement:.2f}% |
| Median Distance Error | {gcn_metrics['median_error']:.2f} km | {mlp_metrics['median_error']:.2f} km | {median_improvement:.2f}% |

## Analysis

### MSE Comparison
The GCN model achieved an MSE of {gcn_metrics['mse']:.2f}, compared to the MLP model's MSE of {mlp_metrics['mse']:.2f}. This represents a {'improvement' if mse_improvement > 0 else 'decrease'} of {abs(mse_improvement):.2f}% in MSE.

### MAE Comparison
The GCN model achieved an MAE of {gcn_metrics['mae']:.2f}, compared to the MLP model's MAE of {mlp_metrics['mae']:.2f}. This represents a {'improvement' if mae_improvement > 0 else 'decrease'} of {abs(mae_improvement):.2f}% in MAE.

### Median Distance Error Comparison
The GCN model achieved a median distance error of {gcn_metrics['median_error']:.2f} km, compared to the MLP model's median distance error of {mlp_metrics['median_error']:.2f} km. This represents a {'improvement' if median_improvement > 0 else 'decrease'} of {abs(median_improvement):.2f}% in median distance error.

## Convergence Analysis
The GCN model converged {'faster' if len(gcn_metrics['epochs']) < len(mlp_metrics['epochs']) else 'slower'} than the MLP model. The GCN model trained for {len(gcn_metrics['epochs'])} epochs, while the MLP model trained for {len(mlp_metrics['epochs'])} epochs.

## Conclusion
{'The GCN model outperformed the MLP model in terms of MSE, MAE, and median distance error.' if mse_improvement > 0 and mae_improvement > 0 and median_improvement > 0 else 'The results show mixed performance between the GCN and MLP models.'}

{'The GCN model\'s ability to capture the graph structure of the IP network topology provides a significant advantage over the MLP model.' if mse_improvement > 0 and mae_improvement > 0 and median_improvement > 0 else 'Further tuning and analysis may be needed to improve the GCN model performance.'}

## Generated on
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Write report to file
    with open(output_file, 'w') as f:
        f.write(report)

def main():
    # Define file paths
    gcn_training_log = '/home/ubuntu/gcn_project/asset/log/New_York_training_gcn.log'
    gcn_test_log = '/home/ubuntu/gcn_project/asset/log/New_York_test_results_gcn.log'
    mlp_training_log = '/home/ubuntu/graphsage_project/graphsage#U6a21#U578b#U5b9e#U9a8c/asset/log/New_York_training_mlp_final.log'
    mlp_test_log = '/home/ubuntu/graphsage_project/graphsage#U6a21#U578b#U5b9e#U9a8c/asset/log/New_York_results_mlp_final.log'
    
    # Create output directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    # Parse GraphTransGeo logs
    gcn_epochs, gcn_train_loss, gcn_train_mse, gcn_val_loss, gcn_val_mse = parse_training_log(gcn_training_log)
    gcn_mse, gcn_mae, gcn_median_error = parse_test_results(gcn_test_log)
    
    # Parse MLP logs
    mlp_epochs, mlp_train_loss, mlp_train_mse, mlp_val_loss, mlp_val_mse = parse_training_log(mlp_training_log)
    mlp_mse, mlp_mae, mlp_median_error = parse_test_results(mlp_test_log)
    
    # Create metrics dictionaries
    gcn_metrics = {
        'epochs': gcn_epochs,
        'train_loss': gcn_train_loss,
        'train_mse': gcn_train_mse,
        'val_loss': gcn_val_loss,
        'val_mse': gcn_val_mse,
        'mse': gcn_mse,
        'mae': gcn_mae,
        'median_error': gcn_median_error
    }
    
    mlp_metrics = {
        'epochs': mlp_epochs,
        'train_loss': mlp_train_loss,
        'train_mse': mlp_train_mse,
        'val_loss': mlp_val_loss,
        'val_mse': mlp_val_mse,
        'mse': mlp_mse,
        'mae': mlp_mae,
        'median_error': mlp_median_error
    }
    
    # Plot training curves
    plot_training_curves(gcn_epochs, gcn_train_loss, gcn_val_loss,
                         mlp_epochs, mlp_train_loss, mlp_val_loss,
                         'reports/model_comparison_curves.png')
    
    # Create comparison report
    create_comparison_report(gcn_metrics, mlp_metrics, 'reports/model_comparison_report.md')
    
    # Print summary
    print("GCN vs MLP Performance Comparison:")
    
    if gcn_mse is not None and mlp_mse is not None:
        print(f"MSE: GCN={gcn_mse:.2f}, MLP={mlp_mse:.2f}")
    else:
        print("MSE: Not available for comparison")
    
    if gcn_mae is not None and mlp_mae is not None:
        print(f"MAE: GCN={gcn_mae:.2f}, MLP={mlp_mae:.2f}")
    else:
        print("MAE: Not available for comparison")
    
    if gcn_median_error is not None and mlp_median_error is not None:
        print(f"Median Distance Error: GCN={gcn_median_error:.2f}km, MLP={mlp_median_error:.2f}km")
    else:
        print("Median Distance Error: Not available for comparison")
    
    # Calculate improvement percentages
    if gcn_mse is not None and mlp_mse is not None and mlp_mse != 0:
        mse_improvement = (mlp_mse - gcn_mse) / mlp_mse * 100
        print(f"MSE Improvement: {mse_improvement:.2f}%")
    
    if gcn_mae is not None and mlp_mae is not None and mlp_mae != 0:
        mae_improvement = (mlp_mae - gcn_mae) / mlp_mae * 100
        print(f"MAE Improvement: {mae_improvement:.2f}%")
    
    if gcn_median_error is not None and mlp_median_error is not None and mlp_median_error != 0:
        median_improvement = (mlp_median_error - gcn_median_error) / mlp_median_error * 100
        print(f"Median Distance Error Improvement: {median_improvement:.2f}%")

if __name__ == "__main__":
    main()
