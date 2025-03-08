#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Visualize GraphTransGeo model performance metrics

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
        train_mse: List of training MSE values
        val_loss: List of validation losses
        val_mse: List of validation MSE values
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

def plot_loss_curves(epochs, train_loss, val_loss, output_file, title="Training and Validation Loss Curves"):
    """
    Plot training and validation loss curves
    
    Args:
        epochs: List of epochs
        train_loss: List of training losses
        val_loss: List of validation losses
        output_file: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Val Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_curves_log_scale(epochs, train_loss, val_loss, output_file, title="Training and Validation Loss Curves (Log Scale)"):
    """
    Plot training and validation loss curves with log scale
    
    Args:
        epochs: List of epochs
        train_loss: List of training losses
        val_loss: List of validation losses
        output_file: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Val Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Use log scale for y-axis
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_mse_curves(epochs, train_mse, val_mse, output_file, title="Training and Validation MSE Curves"):
    """
    Plot training and validation MSE curves
    
    Args:
        epochs: List of epochs
        train_mse: List of training MSE values
        val_mse: List of validation MSE values
        output_file: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    plt.plot(epochs, train_mse, 'g-', label='Train MSE')
    plt.plot(epochs, val_mse, 'm-', label='Val MSE')
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_mse_curves_log_scale(epochs, train_mse, val_mse, output_file, title="Training and Validation MSE Curves (Log Scale)"):
    """
    Plot training and validation MSE curves with log scale
    
    Args:
        epochs: List of epochs
        train_mse: List of training MSE values
        val_mse: List of validation MSE values
        output_file: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    plt.plot(epochs, train_mse, 'g-', label='Train MSE')
    plt.plot(epochs, val_mse, 'm-', label='Val MSE')
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE (log scale)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Use log scale for y-axis
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_metrics(epochs, train_loss, val_loss, train_mse, val_mse, output_file, title="Training and Validation Metrics"):
    """
    Plot combined training and validation metrics (loss and MSE)
    
    Args:
        epochs: List of epochs
        train_loss: List of training losses
        val_loss: List of validation losses
        train_mse: List of training MSE values
        val_mse: List of validation MSE values
        output_file: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Plot loss curves
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot MSE curves
    ax2.plot(epochs, train_mse, 'g-', label='Train MSE')
    ax2.plot(epochs, val_mse, 'm-', label='Val MSE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.set_title('Training and Validation MSE')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_metrics_log_scale(epochs, train_loss, val_loss, train_mse, val_mse, output_file, title="Training and Validation Metrics (Log Scale)"):
    """
    Plot combined training and validation metrics (loss and MSE) with log scale
    
    Args:
        epochs: List of epochs
        train_loss: List of training losses
        val_loss: List of validation losses
        train_mse: List of training MSE values
        val_mse: List of validation MSE values
        output_file: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Plot loss curves with log scale
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Training and Validation Loss (Log Scale)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale('log')  # Use log scale for y-axis
    
    # Plot MSE curves with log scale
    ax2.plot(epochs, train_mse, 'g-', label='Train MSE')
    ax2.plot(epochs, val_mse, 'm-', label='Val MSE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE (log scale)')
    ax2.set_title('Training and Validation MSE (Log Scale)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale('log')  # Use log scale for y-axis
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Define file paths
    gcn_training_log = '/home/ubuntu/gcn_project/asset/log/New_York_training_gcn.log'
    
    # Create output directory if it doesn't exist
    os.makedirs('reports/figures/performance', exist_ok=True)
    
    # Parse GraphTransGeo logs
    epochs, train_loss, train_mse, val_loss, val_mse = parse_training_log(gcn_training_log)
    
    # Plot loss curves
    plot_loss_curves(epochs, train_loss, val_loss, 'reports/figures/performance/gcn_loss_curves.png')
    
    # Plot loss curves with log scale
    plot_loss_curves_log_scale(epochs, train_loss, val_loss, 'reports/figures/performance/gcn_loss_curves_log_scale.png')
    
    # Plot MSE curves
    plot_mse_curves(epochs, train_mse, val_mse, 'reports/figures/performance/gcn_mse_curves.png')
    
    # Plot MSE curves with log scale
    plot_mse_curves_log_scale(epochs, train_mse, val_mse, 'reports/figures/performance/gcn_mse_curves_log_scale.png')
    
    # Plot combined metrics
    plot_combined_metrics(epochs, train_loss, val_loss, train_mse, val_mse, 'reports/figures/performance/gcn_combined_metrics.png')
    
    # Plot combined metrics with log scale
    plot_combined_metrics_log_scale(epochs, train_loss, val_loss, train_mse, val_mse, 'reports/figures/performance/gcn_combined_metrics_log_scale.png')
    
    # Create Chinese versions of the plots
    plot_loss_curves(epochs, train_loss, val_loss, 'reports/figures/performance/gcn_loss_curves_cn.png', title="训练和验证损失曲线")
    plot_loss_curves_log_scale(epochs, train_loss, val_loss, 'reports/figures/performance/gcn_loss_curves_log_scale_cn.png', title="训练和验证损失曲线 (对数尺度)")
    plot_mse_curves(epochs, train_mse, val_mse, 'reports/figures/performance/gcn_mse_curves_cn.png', title="训练和验证MSE曲线")
    plot_mse_curves_log_scale(epochs, train_mse, val_mse, 'reports/figures/performance/gcn_mse_curves_log_scale_cn.png', title="训练和验证MSE曲线 (对数尺度)")
    
    print("GCN performance visualizations saved to reports/figures/performance/ directory.")
    
    # Create a zip file of all the visualization images
    os.system('cd reports/figures && zip -r gcn_performance_visualizations.zip performance/*.png')
    print("Zip file created at reports/figures/gcn_performance_visualizations.zip")

if __name__ == "__main__":
    main()
