#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Improved architecture for GraphTransGeo++

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class ResidualGCNLayer(nn.Module):
    """
    Residual GCN layer with layer normalization
    """
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super(ResidualGCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, num_features]
            
        Returns:
            x: Updated node features [num_nodes, out_channels]
        """
        # Residual connection
        identity = self.residual(x)
        
        # GCN layer
        x = self.conv(x, edge_index)
        
        # Add residual connection
        x = x + identity
        
        # Layer normalization
        x = self.norm(x)
        
        # Activation
        x = F.relu(x)
        
        # Dropout
        x = self.dropout(x)
        
        return x

class MultiHeadGATLayer(nn.Module):
    """
    Multi-head GAT layer with layer normalization
    """
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.3):
        super(MultiHeadGATLayer, self).__init__()
        self.conv = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(out_channels * heads)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Linear(in_channels, out_channels * heads) if in_channels != out_channels * heads else nn.Identity()
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, num_features]
            
        Returns:
            x: Updated node features [num_nodes, out_channels * heads]
        """
        # Residual connection
        identity = self.residual(x)
        
        # GAT layer
        x = self.conv(x, edge_index)
        
        # Add residual connection
        x = x + identity
        
        # Layer normalization
        x = self.norm(x)
        
        # Activation
        x = F.relu(x)
        
        # Dropout
        x = self.dropout(x)
        
        return x

class GraphTransGeoPlusPlus(nn.Module):
    """
    GraphTransGeo++ model for IP geolocation
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, heads=4, dropout=0.3, epsilon=0.01):
        super(GraphTransGeoPlusPlus, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.epsilon = epsilon
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gcn_layers.append(ResidualGCNLayer(hidden_dim, hidden_dim, dropout))
            else:
                self.gcn_layers.append(ResidualGCNLayer(hidden_dim, hidden_dim, dropout))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, num_features]
            batch: Batch indices [num_nodes]
            
        Returns:
            pred: Predictions [batch_size, output_dim]
        """
        # Input layer
        x = self.input_layer(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GCN layers
        for layer in self.gcn_layers:
            x = layer(x, edge_index, edge_attr)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Output layer
        pred = self.output_layer(x)
        
        return pred
    
    def adversarial_loss(self, x, edge_index, edge_attr=None, batch=None, alpha=0.01, beta=0.5, epsilon=None):
        """
        Calculate adversarial loss
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, num_features]
            batch: Batch indices [num_nodes]
            alpha: Weight for consistency loss
            beta: Weight for adversarial loss
            epsilon: Perturbation magnitude (if None, use self.epsilon)
            
        Returns:
            pred: Predictions [batch_size, output_dim]
            adv_loss: Adversarial loss
        """
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

class GATGraphTransGeoPlusPlus(nn.Module):
    """
    GAT-based GraphTransGeo++ model for IP geolocation
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, heads=4, dropout=0.3, epsilon=0.01):
        super(GATGraphTransGeoPlusPlus, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.epsilon = epsilon
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(MultiHeadGATLayer(hidden_dim, hidden_dim // heads, heads, dropout))
            else:
                self.gat_layers.append(MultiHeadGATLayer(hidden_dim, hidden_dim // heads, heads, dropout))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, num_features]
            batch: Batch indices [num_nodes]
            
        Returns:
            pred: Predictions [batch_size, output_dim]
        """
        # Input layer
        x = self.input_layer(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT layers
        for layer in self.gat_layers:
            x = layer(x, edge_index, edge_attr)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Output layer
        pred = self.output_layer(x)
        
        return pred
    
    def adversarial_loss(self, x, edge_index, edge_attr=None, batch=None, alpha=0.01, beta=0.5, epsilon=None):
        """
        Calculate adversarial loss
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, num_features]
            batch: Batch indices [num_nodes]
            alpha: Weight for consistency loss
            beta: Weight for adversarial loss
            epsilon: Perturbation magnitude (if None, use self.epsilon)
            
        Returns:
            pred: Predictions [batch_size, output_dim]
            adv_loss: Adversarial loss
        """
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

class EnhancedEnsembleGraphTransGeoPlusPlus(nn.Module):
    """
    Enhanced ensemble model for GraphTransGeo++
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, heads=4, dropout=0.3, ensemble_size=3, epsilon=0.01):
        super(EnhancedEnsembleGraphTransGeoPlusPlus, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.ensemble_size = ensemble_size
        self.epsilon = epsilon
        
        # Create ensemble models
        self.models = nn.ModuleList()
        
        # Add GraphTransGeo models
        for _ in range(ensemble_size // 2 + ensemble_size % 2):
            self.models.append(GraphTransGeoPlusPlus(input_dim, hidden_dim, output_dim, num_layers, heads, dropout, epsilon))
        
        # Add GAT models
        for _ in range(ensemble_size // 2):
            self.models.append(GATGraphTransGeoPlusPlus(input_dim, hidden_dim, output_dim, num_layers, heads, dropout, epsilon))
        
        # Learnable weights for ensemble
        self.ensemble_weights = nn.Parameter(torch.ones(len(self.models)) / len(self.models))
        
        # Temperature for softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, num_features]
            batch: Batch indices [num_nodes]
            
        Returns:
            pred: Predictions [batch_size, output_dim]
        """
        # Get predictions from all models
        preds = []
        for model in self.models:
            pred = model(x, edge_index, edge_attr, batch)
            preds.append(pred)
        
        # Stack predictions
        preds = torch.stack(preds, dim=0)
        
        # Apply temperature scaling to ensemble weights
        weights = F.softmax(self.ensemble_weights / self.temperature, dim=0)
        
        # Weighted sum of predictions
        pred = torch.sum(weights.view(-1, 1, 1) * preds, dim=0)
        
        return pred
    
    def adversarial_loss(self, x, edge_index, edge_attr=None, batch=None, alpha=0.01, beta=0.5, epsilon=None):
        """
        Calculate adversarial loss
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, num_features]
            batch: Batch indices [num_nodes]
            alpha: Weight for consistency loss
            beta: Weight for adversarial loss
            epsilon: Perturbation magnitude (if None, use self.epsilon)
            
        Returns:
            pred: Predictions [batch_size, output_dim]
            adv_loss: Adversarial loss
        """
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
