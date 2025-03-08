#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Training script for optimized GraphTransGeo model (simplified version)

import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import math
from gcn_data_loader import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.3, epsilon=0.01):
        super(SimpleGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epsilon = epsilon
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        # Input layer
        x = self.input_layer(x)
        x = torch.relu(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.relu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        pred = self.output_layer(x)
        
        return pred
    
    def adversarial_loss(self, x, edge_index=None, edge_attr=None, batch=None, alpha=0.01, beta=0.5, epsilon=None):
        # Use default epsilon if not provided
        if epsilon is None:
            epsilon = self.epsilon
        
        # Forward pass with clean inputs
        pred_clean = self.forward(x, edge_index, edge_attr, batch)
        
        # Generate perturbation
        delta = torch.zeros_like(x, requires_grad=True)
        delta.data.uniform_(-epsilon, epsilon)
        
        # Forward pass with perturbed inputs
        pred_adv = self.forward(x + delta, edge_index, edge_attr, batch)
        
        # Calculate consistency loss
        consistency_loss = nn.MSELoss()(pred_adv, pred_clean.detach())
        
        return pred_clean, consistency_loss

class HaversineLoss(nn.Module):
    def __init__(self):
        super(HaversineLoss, self).__init__()
    
    def forward(self, pred, target):
        # Convert to radians
        pred_lat = pred[:, 0] * math.pi / 180
        pred_lon = pred[:, 1] * math.pi / 180
        target_lat = target[:, 0] * math.pi / 180
        target_lon = target[:, 1] * math.pi / 180
        
        # Calculate Haversine distance
        dlon = target_lon - pred_lon
        dlat = target_lat - pred_lat
        
        a = torch.sin(dlat/2)**2 + torch.cos(pred_lat) * torch.cos(target_lat) * torch.sin(dlon/2)**2
        c = 2 * torch.asin(torch.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        # Calculate loss
        loss = c * r
        
        return loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, hav_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.haversine_loss = HaversineLoss()
        self.hav_weight = hav_weight
    
    def forward(self, pred, target):
        # Calculate MSE loss
        mse = self.mse_loss(pred, target)
        
        # Calculate Haversine loss
        hav = self.haversine_loss(pred, target)
        
        # Calculate combined loss
        loss = (1 - self.hav_weight) * mse + self.hav_weight * hav
        
        return loss, mse, hav

class FeatureAdapter(nn.Module):
    def __init__(self, input_dim, target_dim, strategy='projection'):
        super(FeatureAdapter, self).__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.strategy = strategy
        
        if strategy == 'projection':
            # Linear projection
            self.projection = nn.Linear(input_dim, target_dim)
        elif strategy == 'padding':
            # No parameters needed for padding
            pass
        elif strategy == 'truncation':
            # No parameters needed for truncation
            pass
        else:
            raise ValueError(f'Unknown strategy: {strategy}')
    
    def forward(self, x):
        if self.input_dim == self.target_dim:
            # No adaptation needed
            return x
        
        if self.strategy == 'projection':
            # Linear projection
            return self.projection(x)
        elif self.strategy == 'padding':
            # Padding
            if self.input_dim < self.target_dim:
                padding = torch.zeros(x.size(0), self.target_dim - self.input_dim, device=x.device)
                return torch.cat([x, padding], dim=1)
            else:
                return x[:, :self.target_dim]
        elif self.strategy == 'truncation':
            # Truncation
            if self.input_dim > self.target_dim:
                return x[:, :self.target_dim]
            else:
                padding = torch.zeros(x.size(0), self.target_dim - self.input_dim, device=x.device)
                return torch.cat([x, padding], dim=1)

class ModelWithAdapter(nn.Module):
    def __init__(self, model, adapter):
        super(ModelWithAdapter, self).__init__()
        self.model = model
        self.adapter = adapter
    
    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        x = self.adapter(x)
        return self.model(x, edge_index, edge_attr, batch)
    
    def adversarial_loss(self, x, edge_index=None, edge_attr=None, batch=None, alpha=0.01, beta=0.5, epsilon=0.01):
        x = self.adapter(x)
        return self.model.adversarial_loss(x, edge_index, edge_attr, batch, alpha, beta, epsilon)

def train(model, train_loader, val_loader, optimizer, criterion, device, args):
    # Initialize losses
    train_losses = []
    val_losses = []
    
    # Initialize best validation loss
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Initialize early stopping counter
    early_stop_counter = 0
    
    # Train model
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_mse = 0
        train_hav = 0
        train_adv = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        
        for data in pbar:
            # Move data to device
            data = data.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with adversarial training
            pred, adv_loss = model.adversarial_loss(
                data.x, data.edge_index, None, data.batch,
                alpha=args.alpha, beta=args.beta, epsilon=args.epsilon
            )
            
            # Calculate loss
            loss, mse, hav = criterion(pred, data.y)
            
            # Add adversarial loss
            loss = loss + args.beta * adv_loss
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            # Update weights
            optimizer.step()
            
            # Update loss
            train_loss += loss.item()
            train_mse += mse.item()
            train_hav += hav.item()
            train_adv += adv_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'mse': mse.item(),
                'hav': hav.item(),
                'adv': adv_loss.item()
            })
        
        # Calculate average loss
        train_loss /= len(train_loader)
        train_mse /= len(train_loader)
        train_hav /= len(train_loader)
        train_adv /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        val_mse = 0
        val_hav = 0
        
        with torch.no_grad():
            for data in val_loader:
                # Move data to device
                data = data.to(device)
                
                # Forward pass
                pred = model(data.x, data.edge_index, None, data.batch)
                
                # Calculate loss
                loss, mse, hav = criterion(pred, data.y)
                
                # Update loss
                val_loss += loss.item()
                val_mse += mse.item()
                val_hav += hav.item()
        
        # Calculate average loss
        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        val_hav /= len(val_loader)
        
        # Log losses
        logging.info(f'Epoch {epoch}/{args.epochs}:')
        logging.info(f'  Train Loss: {train_loss:.6f}, MSE: {train_mse:.6f}, Haversine: {train_hav:.6f}, Adversarial: {train_adv:.6f}')
        logging.info(f'  Val Loss: {val_loss:.6f}, MSE: {val_mse:.6f}, Haversine: {val_hav:.6f}')
        
        # Save losses
        train_losses.append({
            'loss': train_loss,
            'mse': train_mse,
            'hav': train_hav,
            'adv': train_adv
        })
        
        val_losses.append({
            'loss': val_loss,
            'mse': val_mse,
            'hav': val_hav
        })
        
        # Save model
        os.makedirs(args.model_dir, exist_ok=True)
        
        if epoch % args.epochs == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), os.path.join(args.model_dir, f'{args.dataset}_gcn_optimized_epoch_{epoch}.pth'))
        
        # Check if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            # Save best model
            torch.save(model.state_dict(), os.path.join(args.model_dir, f'{args.dataset}_gcn_optimized_best.pth'))
            logging.info(f'  Model saved to {os.path.join(args.model_dir, f"{args.dataset}_gcn_optimized_best.pth")}')
            
            # Reset early stopping counter
            early_stop_counter = 0
        else:
            # Increment early stopping counter
            early_stop_counter += 1
            
            # Check if early stopping
            if early_stop_counter >= args.patience and epoch >= args.min_epochs:
                logging.info(f'Early stopping at epoch {epoch}')
                break
    
    # Log best epoch
    logging.info(f'Best epoch: {best_epoch} with validation loss: {best_val_loss:.6f}')
    
    return train_losses, val_losses

def test(model, test_loader, criterion, device):
    # Test model
    model.eval()
    test_loss = 0
    test_mse = 0
    test_hav = 0
    
    # Initialize predictions and targets
    preds = []
    targets = []
    
    with torch.no_grad():
        for data in test_loader:
            # Move data to device
            data = data.to(device)
            
            # Forward pass
            pred = model(data.x, data.edge_index, None, data.batch)
            
            # Calculate loss
            loss, mse, hav = criterion(pred, data.y)
            
            # Update loss
            test_loss += loss.item()
            test_mse += mse.item()
            test_hav += hav.item()
            
            # Save predictions and targets
            preds.append(pred.cpu().numpy())
            targets.append(data.y.cpu().numpy())
    
    # Calculate average loss
    test_loss /= len(test_loader)
    test_mse /= len(test_loader)
    test_hav /= len(test_loader)
    
    # Concatenate predictions and targets
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Calculate MSE and MAE
    test_mse = np.mean((preds - targets) ** 2)
    test_mae = np.mean(np.abs(preds - targets))
    
    # Calculate distance errors
    distance_errors = []
    
    for i in range(len(preds)):
        # Calculate Haversine distance
        pred_lat = preds[i, 0] * math.pi / 180
        pred_lon = preds[i, 1] * math.pi / 180
        target_lat = targets[i, 0] * math.pi / 180
        target_lon = targets[i, 1] * math.pi / 180
        
        dlon = target_lon - pred_lon
        dlat = target_lat - pred_lat
        
        a = math.sin(dlat/2)**2 + math.cos(pred_lat) * math.cos(target_lat) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        distance_errors.append(c * r)
    
    # Calculate median distance error
    test_median = np.median(distance_errors)
    
    return test_loss, test_mse, test_mae, test_median, distance_errors

def plot_training_curves(train_losses, val_losses, args):
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    axs[0, 0].plot([x['loss'] for x in train_losses], label='Train')
    axs[0, 0].plot([x['loss'] for x in val_losses], label='Val')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot MSE
    axs[0, 1].plot([x['mse'] for x in train_losses], label='Train')
    axs[0, 1].plot([x['mse'] for x in val_losses], label='Val')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('MSE')
    axs[0, 1].set_title('MSE')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot Haversine
    axs[1, 0].plot([x['hav'] for x in train_losses], label='Train')
    axs[1, 0].plot([x['hav'] for x in val_losses], label='Val')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Haversine')
    axs[1, 0].set_title('Haversine')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot Adversarial
    axs[1, 1].plot([x['adv'] for x in train_losses], label='Train')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Adversarial')
    axs[1, 1].set_title('Adversarial')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(args.fig_dir, exist_ok=True)
    plt.savefig(os.path.join(args.fig_dir, f'{args.dataset}_training_curves_gcn_optimized.png'))
    
    # Close figure
    plt.close()
    
    logging.info(f'Training curves saved to {os.path.join(args.fig_dir, f"{args.dataset}_training_curves_gcn_optimized.png")}')

def plot_error_distribution(distance_errors, args):
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot histogram
    axs[0].hist(distance_errors, bins=50)
    axs[0].set_xlabel('Distance Error (km)')
    axs[0].set_ylabel('Count')
    axs[0].set_title('Distance Error Histogram')
    axs[0].grid(True)
    
    # Plot CDF
    axs[1].hist(distance_errors, bins=50, cumulative=True, density=True)
    axs[1].set_xlabel('Distance Error (km)')
    axs[1].set_ylabel('CDF')
    axs[1].set_title('Distance Error CDF')
    axs[1].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(args.fig_dir, exist_ok=True)
    plt.savefig(os.path.join(args.fig_dir, f'{args.dataset}_error_distribution_gcn_optimized.png'))
    
    # Close figure
    plt.close()
    
    logging.info(f'Error distribution saved to {os.path.join(args.fig_dir, f"{args.dataset}_error_distribution_gcn_optimized.png")}')

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train optimized GCN model')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='New_York', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='asset/data', help='Data directory')
    
    # Model arguments
    parser.add_argument('--hidden', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--ensemble_size', type=int, default=3, help='Ensemble size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--min_epochs', type=int, default=30, help='Minimum epochs before early stopping')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--hav_weight', type=float, default=0.3, help='Weight for Haversine loss')
    
    # Adversarial training arguments
    parser.add_argument('--epsilon', type=float, default=0.01, help='Perturbation magnitude')
    parser.add_argument('--alpha', type=float, default=0.01, help='Weight for consistency loss')
    parser.add_argument('--beta', type=float, default=0.5, help='Weight for adversarial loss')
    
    # Feature adapter arguments
    parser.add_argument('--use_feature_adapter', action='store_true', help='Use feature adapter')
    parser.add_argument('--adapter_strategy', type=str, default='projection', help='Feature adapter strategy')
    parser.add_argument('--target_dim', type=int, default=30, help='Target dimension for feature adapter')
    
    # Test arguments
    parser.add_argument('--test_only', action='store_true', help='Test only')
    parser.add_argument('--load_epoch', type=int, default=0, help='Epoch to load')
    parser.add_argument('--load_model', type=str, default=None, help='Model to load')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_dir', type=str, default='asset/log', help='Log directory')
    parser.add_argument('--model_dir', type=str, default='asset/model', help='Model directory')
    parser.add_argument('--fig_dir', type=str, default='asset/figures', help='Figure directory')
    
    args = parser.parse_args()
    
    # Log arguments
    logging.info(f'Arguments: {args}')
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Load dataset
    train_dataset, val_dataset, test_dataset, input_dim = load_dataset(
        args.dataset, args.data_dir, seed=args.seed
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create feature adapter if needed
    if args.use_feature_adapter and input_dim != args.target_dim:
        logging.info(f'Using feature adapter: {args.adapter_strategy} ({input_dim} -> {args.target_dim})')
        feature_adapter = FeatureAdapter(input_dim, args.target_dim, args.adapter_strategy)
        
        # Create model with target dimension
        model = SimpleGCN(args.target_dim, args.hidden, 2, args.num_layers, args.dropout, args.epsilon)
        
        # Wrap model with feature adapter
        model = ModelWithAdapter(model, feature_adapter)
    else:
        logging.info(f'No feature adapter needed: input_dim = {input_dim}')
        
        # Create model with input dimension
        model = SimpleGCN(input_dim, args.hidden, 2, args.num_layers, args.dropout, args.epsilon)
    
    # Move model to device
    model = model.to(device)
    
    # Create loss function
    criterion = CombinedLoss(hav_weight=args.hav_weight)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Test only or train and test
    if args.test_only:
        # Load model
        if args.load_model is not None:
            model_path = os.path.join(args.model_dir, args.load_model)
        elif args.load_epoch > 0:
            model_path = os.path.join(args.model_dir, f'{args.dataset}_gcn_optimized_epoch_{args.load_epoch}.pth')
        else:
            model_path = os.path.join(args.model_dir, f'{args.dataset}_gcn_optimized_best.pth')
        
        # Load model
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f'Model loaded from {model_path}')
        
        # Test model
        test_loss, test_mse, test_mae, test_median, distance_errors = test(model, test_loader, criterion, device)
        
        # Log test results
        logging.info(f'Test Loss: {test_loss:.6f}, MSE: {test_mse:.6f}, Haversine: {test_loss - test_mse:.6f}')
        logging.info(f'Test MSE: {test_mse:.6f}')
        logging.info(f'Test MAE: {test_mae:.6f}')
        logging.info(f'Median Distance Error: {test_median:.6f}')
        
        # Plot error distribution
        plot_error_distribution(distance_errors, args)
        
        # Save test results
        os.makedirs(args.log_dir, exist_ok=True)
        
        with open(os.path.join(args.log_dir, f'{args.dataset}_test_gcn_optimized.log'), 'w') as f:
            f.write(f'Test Loss: {test_loss:.6f}, MSE: {test_mse:.6f}, Haversine: {test_loss - test_mse:.6f}\n')
            f.write(f'Test MSE: {test_mse:.6f}\n')
            f.write(f'Test MAE: {test_mae:.6f}\n')
            f.write(f'Median Distance Error: {test_median:.6f}\n')
    else:
        # Create directories
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.model_dir, exist_ok=True)
        os.makedirs(args.fig_dir, exist_ok=True)
        
        # Train model
        start_time = time.time()
        train_losses, val_losses = train(model, train_loader, val_loader, optimizer, criterion, device, args)
        end_time = time.time()
        
        # Log training time
        logging.info(f'Training time: {end_time - start_time:.2f} seconds')
        
        # Plot training curves
        plot_training_curves(train_losses, val_losses, args)
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(args.model_dir, f'{args.dataset}_gcn_optimized_best.pth'), map_location=device))
        logging.info(f'Best model loaded from {os.path.join(args.model_dir, f"{args.dataset}_gcn_optimized_best.pth")}')
        
        # Test model
        test_loss, test_mse, test_mae, test_median, distance_errors = test(model, test_loader, criterion, device)
        
        # Log test results
        logging.info(f'Test Loss: {test_loss:.6f}, MSE: {test_mse:.6f}, Haversine: {test_loss - test_mse:.6f}')
        logging.info(f'Test MSE: {test_mse:.6f}')
        logging.info(f'Test MAE: {test_mae:.6f}')
        logging.info(f'Median Distance Error: {test_median:.6f}')
        
        # Plot error distribution
        plot_error_distribution(distance_errors, args)
        
        # Save test results
        with open(os.path.join(args.log_dir, f'{args.dataset}_test_gcn_optimized.log'), 'w') as f:
            f.write(f'Test Loss: {test_loss:.6f}, MSE: {test_mse:.6f}, Haversine: {test_loss - test_mse:.6f}\n')
            f.write(f'Test MSE: {test_mse:.6f}\n')
            f.write(f'Test MAE: {test_mae:.6f}\n')
            f.write(f'Median Distance Error: {test_median:.6f}\n')

if __name__ == '__main__':
    main()
