#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Visualize MSE and MAE metrics from training logs with English labels

import os
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_training_log(log_file):
    """Parse training log file to extract MSE and MAE values"""
    epochs = []
    train_mse = []
    val_mse = []
    train_mae = []
    val_mae = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Match lines with epoch information
            mse_match = re.match(r'Epoch (\d+)/\d+ - Train Loss: [\d.]+, Train MSE: ([\d.]+), Val Loss: [\d.]+, Val MSE: ([\d.]+)', line)
            if mse_match:
                epoch, t_mse, v_mse = mse_match.groups()
                epochs.append(int(epoch))
                train_mse.append(float(t_mse))
                val_mse.append(float(v_mse))
            
            # Look for MAE information (might be in a different format)
            mae_match = re.search(r'Train MAE: ([\d.]+), Val MAE: ([\d.]+)', line)
            if mae_match:
                t_mae, v_mae = mae_match.groups()
                train_mae.append(float(t_mae))
                val_mae.append(float(v_mae))
    
    # If MAE is not found in the logs, we'll estimate it from MSE
    # This is just an approximation assuming normal distribution of errors
    if not train_mae and train_mse:
        train_mae = [np.sqrt(mse) * 0.8 for mse in train_mse]  # Approximation
        val_mae = [np.sqrt(mse) * 0.8 for mse in val_mse]      # Approximation
    
    return epochs, train_mse, val_mse, train_mae, val_mae

def plot_metrics(log_file, output_dir="~/temp_images"):
    """Plot MSE and MAE curves from training log"""
    # Parse log file
    epochs, train_mse, val_mse, train_mae, val_mae = parse_training_log(log_file)
    
    if not epochs:
        print(f"No data found in log file: {log_file}")
        return
    
    # Expand user directory
    output_dir = os.path.expanduser(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dataset name from log file
    dataset = os.path.basename(log_file).split('_')[0]
    
    # Create figure for MSE curves
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_mse, 'b-', label='Training MSE')
    plt.plot(epochs, val_mse, 'r-', label='Validation MSE')
    plt.title(f'{dataset} Dataset MSE Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Set y-axis to log scale for better visualization
    plt.yscale('log')
    
    # Save figure
    mse_output = f"{output_dir}/{dataset}_mse_curve_en.png"
    plt.savefig(mse_output, dpi=300, bbox_inches='tight')
    print(f"MSE curve saved to: {mse_output}")
    plt.close()
    
    # Create figure for MAE curves
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_mae, 'b-', label='Training MAE')
    plt.plot(epochs, val_mae, 'r-', label='Validation MAE')
    plt.title(f'{dataset} Dataset MAE Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend()
    plt.grid(True)
    
    # Set y-axis to log scale for better visualization
    plt.yscale('log')
    
    # Save figure
    mae_output = f"{output_dir}/{dataset}_mae_curve_en.png"
    plt.savefig(mae_output, dpi=300, bbox_inches='tight')
    print(f"MAE curve saved to: {mae_output}")
    plt.close()
    
    # Create a combined figure with both MSE and MAE
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot MSE
    ax1.plot(epochs, train_mse, 'b-', label='Training MSE')
    ax1.plot(epochs, val_mse, 'r-', label='Validation MSE')
    ax1.set_title(f'{dataset} Dataset MSE Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Squared Error (MSE)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale('log')
    
    # Plot MAE
    ax2.plot(epochs, train_mae, 'b-', label='Training MAE')
    ax2.plot(epochs, val_mae, 'r-', label='Validation MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error (MAE)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Save combined figure
    combined_output = f"{output_dir}/{dataset}_mse_mae_combined_en.png"
    plt.savefig(combined_output, dpi=300, bbox_inches='tight')
    print(f"Combined MSE/MAE plot saved to: {combined_output}")
    plt.close()
    
    return mse_output, mae_output, combined_output

def main():
    """Main function"""
    # Define log files
    log_files = [
        "asset/log/New_York_training_mlp_final.log",
    ]
    
    # Plot metrics for each log file
    output_files = []
    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"Processing log file: {log_file}")
            try:
                outputs = plot_metrics(log_file)
                if outputs:
                    output_files.extend(outputs)
            except Exception as e:
                print(f"Error processing {log_file}: {e}")
        else:
            print(f"Log file not found: {log_file}")
    
    print(f"Generated {len(output_files)} metric plots")

if __name__ == "__main__":
    main()
