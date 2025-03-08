#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Training script for GraphTransGeo-based GraphTransGeo++

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from lib.model_graphtransgeo_gcn import GraphTransGeoPlusPlus
from gcn_data_loader import load_ip_graph_data
from lib.utils import haversine_distance

def train(model, train_loader, val_loader, optimizer, scheduler, device, args):
    """
    Train the model
    
    Args:
        model: GraphTransGeoPlusPlus model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use
        args: Command line arguments
        
    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    # Create log directory
    os.makedirs('asset/log', exist_ok=True)
    os.makedirs('asset/model', exist_ok=True)
    
    # Initialize lists to store losses
    train_losses = []
    train_mse_losses = []
    val_losses = []
    val_mse_losses = []
    
    # MSE loss function
    mse_loss = nn.MSELoss()
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        
        for batch in pbar:
            # Move batch to device
            batch = batch.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with adversarial training
            clean_output, adv_loss = model.adversarial_loss(
                batch.x, batch.edge_index, batch.batch,
                alpha=args.alpha, beta=args.beta
            )
            
            # Calculate MSE loss
            # Handle the case where batch.y is a single coordinate for the entire graph
            if batch.y.size(0) == 1:
                # Repeat the target for each node in the batch
                target = batch.y.repeat(clean_output.size(0), 1)
                mse = mse_loss(clean_output, target)
            else:
                mse = mse_loss(clean_output, batch.y)
            
            # Total loss
            loss = mse + adv_loss
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            epoch_mse += mse.item()
            pbar.set_postfix({'loss': loss.item(), 'mse': mse.item()})
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_mse = epoch_mse / len(train_loader)
        train_losses.append(avg_train_loss)
        train_mse_losses.append(avg_train_mse)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = batch.to(device)
                
                # Forward pass
                output = model(batch.x, batch.edge_index, batch.batch)
                
                # Calculate MSE loss
                # Handle the case where batch.y is a single coordinate for the entire graph
                if batch.y.size(0) == 1:
                    # Repeat the target for each node in the batch
                    target = batch.y.repeat(output.size(0), 1)
                    mse = mse_loss(output, target)
                else:
                    mse = mse_loss(output, batch.y)
                
                # Update validation loss
                val_loss += mse.item()
                val_mse += mse.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mse = val_mse / len(val_loader)
        val_losses.append(avg_val_loss)
        val_mse_losses.append(avg_val_mse)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print epoch results
        print(f'Epoch {epoch}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Train MSE: {avg_train_mse:.4f}, Val Loss: {avg_val_loss:.4f}, Val MSE: {avg_val_mse:.4f}')
        
        # Save model checkpoint
        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), f'asset/model/{args.dataset}_gcn_{epoch}.pth')
        
        # Log results
        with open(f'asset/log/{args.dataset}_training_gcn.log', 'a') as f:
            f.write(f'Epoch {epoch}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Train MSE: {avg_train_mse:.4f}, Val Loss: {avg_val_loss:.4f}, Val MSE: {avg_val_mse:.4f}\n')
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), train_mse_losses, label='Train MSE')
    plt.plot(range(1, args.epochs + 1), val_mse_losses, label='Val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training and Validation MSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'asset/log/{args.dataset}_training_curves_gcn.png')
    
    return model, train_losses, val_losses

def test(model, test_loader, device, args):
    """
    Test the model
    
    Args:
        model: GraphTransGeoPlusPlus model
        test_loader: DataLoader for test data
        device: Device to use
        args: Command line arguments
        
    Returns:
        mse: Mean squared error
        mae: Mean absolute error
        median_error: Median distance error (km)
    """
    model.eval()
    
    # Initialize lists to store predictions and targets
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch.x, batch.edge_index, batch.batch)
            
            # Handle the case where batch.y is a single coordinate for the entire graph
            if batch.y.size(0) == 1:
                # Use the same target for all nodes in this batch
                target = batch.y.repeat(output.size(0), 1)
                # Store predictions and targets
                all_preds.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
            else:
                # Store predictions and targets
                all_preds.append(output.cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())
    
    # Concatenate predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate MSE
    mse = np.mean(np.sum((all_preds - all_targets) ** 2, axis=1))
    
    # Calculate MAE
    mae = np.mean(np.sum(np.abs(all_preds - all_targets), axis=1))
    
    # Calculate distance errors (in km)
    distance_errors = []
    for i in range(len(all_preds)):
        pred_lat, pred_lon = all_preds[i]
        true_lat, true_lon = all_targets[i]
        
        # Calculate haversine distance
        distance = haversine_distance(pred_lat, pred_lon, true_lat, true_lon)
        distance_errors.append(distance)
    
    # Calculate median distance error
    median_error = np.median(distance_errors)
    
    # Print results
    print(f'Test Results:')
    print(f'MSE: {mse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'Median Distance Error: {median_error:.2f} km')
    
    # Log results
    with open(f'asset/log/{args.dataset}_test_results_gcn.log', 'w') as f:
        f.write(f'Test Results:\n')
        f.write(f'MSE: {mse:.2f}\n')
        f.write(f'MAE: {mae:.2f}\n')
        f.write(f'Median Distance Error: {median_error:.2f} km\n')
    
    return mse, mae, median_error

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train GraphTransGeo++ with GCN')
    parser.add_argument('--dataset', type=str, default='New_York', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Perturbation magnitude')
    parser.add_argument('--alpha', type=float, default=0.01, help='Weight for consistency loss')
    parser.add_argument('--beta', type=float, default=0.5, help='Weight for adversarial loss')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GCN layers')
    parser.add_argument('--test_only', action='store_true', help='Test only')
    parser.add_argument('--load_epoch', type=int, default=100, help='Epoch to load for testing')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    train_loader, val_loader, test_loader = load_ip_graph_data(
        args.dataset, batch_size=args.batch_size
    )
    
    # Get input dimension from data
    sample_data = next(iter(train_loader))
    input_dim = sample_data.x.size(1)
    
    # Create model
    model = GraphTransGeoPlusPlus(
        input_dim=input_dim,
        hidden_dim=args.hidden,
        output_dim=2,  # Latitude and longitude
        dropout=args.dropout,
        num_layers=args.num_layers,
        epsilon=args.epsilon
    ).to(device)
    
    # Print model summary
    print(model)
    
    if args.test_only:
        # Load model for testing
        model_path = f'asset/model/{args.dataset}_gcn_{args.load_epoch}.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f'Loaded model from {model_path}')
        else:
            print(f'Model file {model_path} not found. Testing with random weights.')
        
        # Test model
        test(model, test_loader, device, args)
    else:
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Create scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Train model
        model, train_losses, val_losses = train(model, train_loader, val_loader, optimizer, scheduler, device, args)
        
        # Test model
        test(model, test_loader, device, args)

if __name__ == '__main__':
    main()
