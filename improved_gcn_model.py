#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Improved GraphTransGeo model with tricks for better performance

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

# Import custom modules
from gcn_data_loader import load_ip_graph_data

class ImprovedGraphTransGeoGCN(nn.Module):
    """
    Improved GraphTransGeo++ model with GCN layers and various tricks for better performance
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=2, num_layers=2, dropout=0.3, 
                 residual=True, layer_norm=True, batch_norm=True, gnn_type='gcn'):
        super(ImprovedGraphTransGeoGCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.gnn_type = gnn_type
        
        # Input normalization
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_dropout = nn.Dropout(dropout)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        # First layer
        if gnn_type == 'gcn':
            self.gnn_layers.append(GCNConv(input_dim, hidden_dim))
        elif gnn_type == 'sage':
            self.gnn_layers.append(SAGEConv(input_dim, hidden_dim))
        elif gnn_type == 'gat':
            self.gnn_layers.append(GATConv(input_dim, hidden_dim))
        
        # Additional layers
        for i in range(1, num_layers):
            if gnn_type == 'gcn':
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'sage':
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim))
        
        # Batch normalization layers
        if batch_norm:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
        # Layer normalization layers
        if layer_norm:
            self.ln_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Dropout layers
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Feature scaling parameters (learnable)
        self.feature_scale = nn.Parameter(torch.ones(input_dim))
        self.feature_bias = nn.Parameter(torch.zeros(input_dim))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, batch=None):
        # Apply feature scaling
        x = x * self.feature_scale + self.feature_bias
        
        # Input normalization
        x = self.input_bn(x)
        x = self.input_dropout(x)
        
        # Apply GNN layers
        prev_x = x  # For residual connections
        
        for i in range(self.num_layers):
            # GNN layer
            x = self.gnn_layers[i](x, edge_index)
            
            # Apply activation
            x = F.relu(x)
            
            # Apply batch normalization if enabled
            if self.batch_norm:
                x = self.bn_layers[i](x)
            
            # Apply layer normalization if enabled
            if self.layer_norm:
                x = self.ln_layers[i](x)
            
            # Apply dropout
            x = self.dropout_layers[i](x)
            
            # Add residual connection if enabled and dimensions match
            if self.residual and i > 0 and prev_x.shape == x.shape:
                x = x + prev_x
            
            prev_x = x
        
        # Output layer
        x = self.output_layer(x)
        
        return x

class ImprovedGraphTransGeoPlusPlus(nn.Module):
    """
    Improved GraphTransGeo++ model with adversarial training and ensemble
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=2, num_layers=2, dropout=0.3, 
                 epsilon=0.01, ensemble_size=3, gnn_types=['gcn', 'sage', 'gat']):
        super(ImprovedGraphTransGeoPlusPlus, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.ensemble_size = min(ensemble_size, len(gnn_types))
        
        # Create ensemble of models
        self.models = nn.ModuleList()
        for i in range(self.ensemble_size):
            gnn_type = gnn_types[i % len(gnn_types)]
            model = ImprovedGraphTransGeoGCN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout,
                gnn_type=gnn_type
            )
            self.models.append(model)
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(self.ensemble_size) / self.ensemble_size)
    
    def forward(self, x, edge_index, batch=None):
        # Get predictions from each model in the ensemble
        outputs = []
        for model in self.models:
            outputs.append(model(x, edge_index, batch))
        
        # Apply softmax to ensemble weights
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # Weighted average of predictions
        ensemble_output = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            ensemble_output += weights[i] * output
        
        return ensemble_output
    
    def generate_adversarial_perturbation(self, x, edge_index, batch, y):
        """
        Generate adversarial perturbation for input features
        
        Args:
            x: Input features
            edge_index: Edge indices
            batch: Batch indices
            y: Target labels
            
        Returns:
            perturbed_x: Perturbed input features
        """
        # Enable gradient calculation for input
        x.requires_grad = True
        
        # Forward pass
        output = self.forward(x, edge_index, batch)
        
        # Calculate loss
        loss = F.mse_loss(output, y)
        
        # Backward pass to get gradients
        loss.backward()
        
        # Get sign of gradients
        grad_sign = torch.sign(x.grad)
        
        # Generate perturbation
        perturbation = self.epsilon * grad_sign
        
        # Apply perturbation
        perturbed_x = x + perturbation
        
        # Detach from computation graph
        perturbed_x = perturbed_x.detach()
        x.requires_grad = False
        
        return perturbed_x

def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=50, 
          epsilon=0.01, alpha=0.01, beta=0.5, log_file=None):
    """
    Train the model
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use for training
        epochs: Number of epochs
        epsilon: Perturbation magnitude for adversarial training
        alpha: Weight for adversarial loss
        beta: Weight for consistency loss
        log_file: Path to log file
        
    Returns:
        model: Trained model
    """
    # Loss function
    mse_loss = nn.MSELoss()
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass with clean input
            clean_output = model(batch.x, batch.edge_index, batch.batch)
            
            # Generate adversarial perturbation
            perturbed_x = model.generate_adversarial_perturbation(
                batch.x.clone(), batch.edge_index, batch.batch, batch.y
            )
            
            # Forward pass with perturbed input
            adv_output = model(
                perturbed_x, batch.edge_index, batch.batch
            )
            
            # Calculate MSE loss
            # Handle the case where batch.y is a single coordinate for the entire graph
            if batch.y.size(0) == 1:
                # Repeat the target for each node in the batch
                target = batch.y.repeat(clean_output.size(0), 1)
                mse = mse_loss(clean_output, target)
            else:
                mse = mse_loss(clean_output, batch.y)
            
            # Calculate adversarial loss (KL divergence between clean and adversarial outputs)
            adv_loss = alpha * F.mse_loss(clean_output, adv_output)
            
            # Calculate consistency loss (L2 distance between clean and adversarial outputs)
            consistency_loss = beta * F.mse_loss(clean_output, adv_output)
            
            # Total loss
            loss = mse + adv_loss + consistency_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update training loss
            train_loss += loss.item()
            train_mse += mse.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        train_mse /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
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
        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        log_message = f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.2f}, Train MSE: {train_mse:.2f}, Val Loss: {val_loss:.2f}, Val MSE: {val_mse:.2f}"
        print(log_message)
        
        # Log to file
        if log_file:
            with open(log_file, 'a') as f:
                f.write(log_message + '\n')
        
        # Save model checkpoint
        if epoch % 10 == 0 or epoch == epochs:
            model_dir = os.path.dirname(log_file) if log_file else 'asset/model'
            os.makedirs(model_dir, exist_ok=True)
            # Extract dataset name from log file or use a default
            if log_file:
                dataset_name = os.path.basename(log_file).split('_')[0]
            else:
                dataset_name = "default"
            torch.save(model.state_dict(), f"{model_dir}/{dataset_name}_improved_gcn_{epoch}.pth")
    
    return model

def test(model, test_loader, device, log_file=None):
    """
    Test the model
    
    Args:
        model: Model to test
        test_loader: Test data loader
        device: Device to use for testing
        log_file: Path to log file
        
    Returns:
        mse: Mean squared error
        mae: Mean absolute error
        median_error: Median distance error in km
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
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
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    
    # Calculate median distance error in km
    distances = []
    for i in range(len(all_preds)):
        # Calculate Haversine distance
        lat1, lon1 = all_preds[i]
        lat2, lon2 = all_targets[i]
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        
        distance = c * r
        distances.append(distance)
    
    median_error = np.median(distances)
    
    # Print results
    log_message = f"Test Results:\nMSE: {mse:.2f}\nMAE: {mae:.2f}\nMedian Distance Error: {median_error:.2f} km"
    print(log_message)
    
    # Log to file
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_message + '\n')
    
    return mse, mae, median_error

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train and test improved GraphTransGeo++ model')
    parser.add_argument('--dataset', type=str, default='New_York', help='Dataset name')
    parser.add_argument('--root', type=str, default='datasets', help='Root directory for datasets')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Perturbation magnitude')
    parser.add_argument('--alpha', type=float, default=0.01, help='Adversarial loss weight')
    parser.add_argument('--beta', type=float, default=0.5, help='Consistency loss weight')
    parser.add_argument('--ensemble_size', type=int, default=3, help='Number of models in ensemble')
    parser.add_argument('--test_only', action='store_true', help='Test only')
    parser.add_argument('--load_epoch', type=int, default=100, help='Epoch to load for testing')
    parser.add_argument('--dim_in', type=int, default=None, help='Input dimension (for testing on different datasets)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create log directory
    os.makedirs('asset/log', exist_ok=True)
    os.makedirs('asset/model', exist_ok=True)
    
    # Set log files
    training_log = f"asset/log/{args.dataset}_training_improved_gcn.log"
    test_log = f"asset/log/{args.dataset}_test_results_improved_gcn.log"
    
    # Load data
    data_loaders = load_ip_graph_data(
        root=args.root,
        dataset_name=args.dataset,
        batch_size=args.batch_size
    )
    # Unpack data loaders
    if isinstance(data_loaders, tuple) and len(data_loaders) >= 3:
        train_loader = data_loaders[0]
        val_loader = data_loaders[1]
        test_loader = data_loaders[2]
        # Get input dimension from the first batch in train loader
        for batch in train_loader:
            input_dim = batch.x.size(1)
            break
    else:
        raise ValueError("Invalid data loaders returned from load_ip_graph_data")
    
    # Override input dimension if specified
    if args.dim_in is not None:
        input_dim = args.dim_in
    
    # Create model
    model = ImprovedGraphTransGeoPlusPlus(
        input_dim=input_dim,
        hidden_dim=args.hidden,
        output_dim=2,  # Latitude and longitude
        num_layers=args.num_layers,
        dropout=args.dropout,
        epsilon=args.epsilon,
        ensemble_size=args.ensemble_size
    )
    model = model.to(device)
    print(model)
    
    # Test only
    if args.test_only:
        # Load model
        model_path = f"asset/model/{args.dataset}_improved_gcn_{args.load_epoch}.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model file {model_path} not found. Testing with random weights.")
        
        # Test model
        test(model, test_loader, device, test_log)
        return
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Train model
    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        epsilon=args.epsilon,
        alpha=args.alpha,
        beta=args.beta,
        log_file=training_log
    )
    
    # Test model
    test(model, test_loader, device, test_log)

if __name__ == "__main__":
    main()
