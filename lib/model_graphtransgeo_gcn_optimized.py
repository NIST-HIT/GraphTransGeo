#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Optimized GraphTransGeo model for GraphTransGeo++

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GraphSAGE, GATConv

class OptimizedGraphTransGeoGCN(nn.Module):
    """
    Optimized GraphTransGeo GCN model with residual connections and attention
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=2, dropout=0.3, num_layers=3):
        super(OptimizedGraphTransGeoGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input normalization
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_dropout = nn.Dropout(dropout)
        
        # GraphTransGeo layers with residual connections
        self.gcn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # First layer
        self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
        self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
        self.dropout_layers.append(nn.Dropout(dropout))
        
        # Additional layers with residual connections
        for i in range(1, num_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout))
        
        # Attention layer for node weighting
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass with residual connections and attention
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes]
            
        Returns:
            output: Predicted coordinates [batch_size, output_dim]
        """
        # Input normalization
        x = self.input_bn(x)
        x = self.input_dropout(x)
        
        # First GraphTransGeo layer
        x1 = self.gcn_layers[0](x, edge_index)
        x1 = F.relu(x1)
        x1 = self.bn_layers[0](x1)
        x1 = self.dropout_layers[0](x1)
        
        # Additional GraphTransGeo layers with residual connections
        x_prev = x1
        for i in range(1, self.num_layers):
            x_new = self.gcn_layers[i](x_prev, edge_index)
            x_new = F.relu(x_new)
            x_new = self.bn_layers[i](x_new)
            x_new = self.dropout_layers[i](x_new)
            
            # Residual connection
            x_prev = x_new + x_prev
        
        # Final representation
        x = x_prev
        
        # Apply attention for node weighting if in batch mode
        if batch is not None:
            # Calculate attention scores
            attention_scores = self.attention(x)
            attention_weights = torch.softmax(attention_scores, dim=0)
            
            # Apply attention weights
            x = x * attention_weights
            
            # Global mean pooling
            x = global_mean_pool(x, batch)
        
        # Output layer
        output = self.output_layer(x)
        
        return output


class OptimizedGraphTransGeoGAT(nn.Module):
    """
    Optimized GraphTransGeo GAT model with multi-head attention
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=2, dropout=0.3, num_layers=3, heads=4):
        super(OptimizedGraphTransGeoGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        
        # Input normalization
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_dropout = nn.Dropout(dropout)
        
        # GAT layers with residual connections
        self.gat_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(GATConv(input_dim, hidden_dim // heads, heads=heads))
        self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
        self.dropout_layers.append(nn.Dropout(dropout))
        
        # Additional layers with residual connections
        for i in range(1, num_layers):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass with multi-head attention and residual connections
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes]
            
        Returns:
            output: Predicted coordinates [batch_size, output_dim]
        """
        # Input normalization
        x = self.input_bn(x)
        x = self.input_dropout(x)
        
        # First GAT layer
        x1 = self.gat_layers[0](x, edge_index)
        x1 = F.relu(x1)
        x1 = self.bn_layers[0](x1)
        x1 = self.dropout_layers[0](x1)
        
        # Additional GAT layers with residual connections
        x_prev = x1
        for i in range(1, self.num_layers):
            x_new = self.gat_layers[i](x_prev, edge_index)
            x_new = F.relu(x_new)
            x_new = self.bn_layers[i](x_new)
            x_new = self.dropout_layers[i](x_new)
            
            # Residual connection
            x_prev = x_new + x_prev
        
        # Final representation
        x = x_prev
        
        # Global mean pooling if in batch mode
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Output layer
        output = self.output_layer(x)
        
        return output


class EnsembleGraphTransGeoPlusPlus(nn.Module):
    """
    Ensemble model combining multiple GCN variants
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=2, dropout=0.3, 
                 num_layers=3, epsilon=0.01, ensemble_size=3):
        super(EnsembleGraphTransGeoPlusPlus, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.ensemble_size = ensemble_size
        
        # Create ensemble of models with different architectures
        self.models = nn.ModuleList()
        
        # Standard GraphTransGeo
        self.models.append(OptimizedGraphTransGeoGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            num_layers=num_layers
        ))
        
        # GraphTransGeo with more layers
        if ensemble_size > 1:
            self.models.append(OptimizedGraphTransGeoGCN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
                num_layers=num_layers + 1
            ))
        
        # GAT model
        if ensemble_size > 2:
            self.models.append(OptimizedGraphTransGeoGAT(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
                num_layers=num_layers,
                heads=4
            ))
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(len(self.models)) / len(self.models))
    
    def forward(self, x, edge_index, batch=None, return_perturbed=False):
        """
        Forward pass with ensemble
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes]
            return_perturbed: Whether to return perturbed predictions
            
        Returns:
            output: Predicted coordinates [batch_size, output_dim]
            perturbed_output: Perturbed predictions (optional)
        """
        # Get predictions from each model
        predictions = []
        for model in self.models:
            pred = model(x, edge_index, batch)
            predictions.append(pred)
        
        # Normalize ensemble weights
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # Weighted average of predictions
        output = torch.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            output += weights[i] * pred
        
        if return_perturbed:
            # Generate adversarial perturbation
            perturbed_x = self.get_adversarial_perturbation(x, edge_index, batch, self.epsilon)
            
            # Forward pass with perturbed input
            perturbed_predictions = []
            for model in self.models:
                perturbed_pred = model(perturbed_x, edge_index, batch)
                perturbed_predictions.append(perturbed_pred)
            
            # Weighted average of perturbed predictions
            perturbed_output = torch.zeros_like(perturbed_predictions[0])
            for i, pred in enumerate(perturbed_predictions):
                perturbed_output += weights[i] * pred
            
            return output, perturbed_output
        
        return output
    
    def get_adversarial_perturbation(self, x, edge_index, batch, epsilon=0.01):
        """
        Generate adversarial perturbation for input features
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes]
            epsilon: Perturbation magnitude
            
        Returns:
            perturbed_x: Perturbed features [num_nodes, input_dim]
        """
        # Create a copy of x for gradient calculation
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = self.forward(x_adv, edge_index, batch)
        
        # Calculate loss (using MSE as a proxy)
        target = torch.zeros_like(output)  # Dummy target
        loss = F.mse_loss(output, target)
        
        # Backward pass to get gradients
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        
        # Generate perturbation using sign of gradients (FGSM-like approach)
        perturbation = epsilon * torch.sign(grad)
        
        # Create perturbed input
        perturbed_x = x.detach() + perturbation
        
        return perturbed_x
    
    def adversarial_loss(self, x, edge_index, batch=None, alpha=0.01, beta=0.5):
        """
        Calculate adversarial loss with improved consistency regularization
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes]
            alpha: Weight for consistency loss
            beta: Weight for adversarial loss
            
        Returns:
            clean_output: Predictions on clean input
            adv_loss: Adversarial loss component (already weighted with beta)
        """
        # Standard forward pass
        clean_output = self.forward(x, edge_index, batch)
        
        # Generate adversarial perturbation
        perturbed_x = self.get_adversarial_perturbation(x, edge_index, batch, self.epsilon)
        
        # Get predictions on perturbed input
        perturbed_output = self.forward(perturbed_x, edge_index, batch)
        
        # Improved consistency loss with KL divergence
        # For regression, we use MSE as a proxy
        consistency_loss = F.mse_loss(clean_output, perturbed_output)
        
        # Total adversarial loss (apply beta here so it's not applied again in training loop)
        adv_loss = beta * alpha * consistency_loss
        
        return clean_output, adv_loss
