#!/usr/bin/env python
# -*- coding: utf-8 -*-
# GraphTransGeo++ model with GraphTransGeo implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data
import numpy as np

class GraphTransGeoGCN(nn.Module):
    """
    GraphTransGeo++ model with Graph Convolutional Networks
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=2, dropout=0.3, num_layers=2):
        super(GraphTransGeoGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input layer
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_dropout = nn.Dropout(dropout)
        
        # GraphTransGeo layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
        
        # Batch normalization and dropout for GraphTransGeo layers
        self.bn_layers = nn.ModuleList()
        self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
        self.dropout_layers = nn.ModuleList()
        self.dropout_layers.append(nn.Dropout(dropout))
        
        # Additional GCN layers
        for i in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
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
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes]
            
        Returns:
            output: Predicted coordinates [batch_size, output_dim]
        """
        # Input normalization and dropout
        x = self.input_bn(x)
        x = self.input_dropout(x)
        
        # GraphTransGeo layers
        for i in range(self.num_layers):
            x = self.gcn_layers[i](x, edge_index)
            x = F.relu(x)
            x = self.bn_layers[i](x)
            x = self.dropout_layers[i](x)
        
        # Global mean pooling if batch is provided
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch)
        
        # Output layer
        output = self.output_layer(x)
        
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
        # Enable gradient calculation for input
        x.requires_grad = True
        
        # Forward pass
        output = self.forward(x, edge_index, batch)
        
        # Calculate loss (using MSE as a proxy)
        target = torch.zeros_like(output)  # Dummy target
        loss = F.mse_loss(output, target)
        
        # Backward pass to get gradients
        loss.backward()
        
        # Generate perturbation using sign of gradients (FGSM-like approach)
        perturbation = epsilon * torch.sign(x.grad)
        
        # Create perturbed input
        perturbed_x = x.detach() + perturbation
        
        # Reset requires_grad
        x.requires_grad = False
        
        return perturbed_x
    
    def consistency_loss(self, x, perturbed_x, edge_index, batch=None):
        """
        Calculate consistency loss between predictions on clean and perturbed inputs
        
        Args:
            x: Original features [num_nodes, input_dim]
            perturbed_x: Perturbed features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes]
            
        Returns:
            loss: Consistency loss (KL divergence)
        """
        # Get predictions
        pred_clean = self.forward(x, edge_index, batch)
        pred_perturbed = self.forward(perturbed_x, edge_index, batch)
        
        # Calculate KL divergence loss
        # Since we're dealing with regression, we'll use MSE as a proxy for consistency
        loss = F.mse_loss(pred_clean, pred_perturbed)
        
        return loss


class GraphTransGeoPlusPlus(nn.Module):
    """
    Complete GraphTransGeo++ model with adversarial training
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=2, dropout=0.3, num_layers=2, epsilon=0.01):
        super(GraphTransGeoPlusPlus, self).__init__()
        self.base_model = GraphTransGeoGCN(input_dim, hidden_dim, output_dim, dropout, num_layers)
        self.epsilon = epsilon
    
    def forward(self, x, edge_index, batch=None, return_perturbed=False):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes]
            return_perturbed: Whether to return perturbed predictions
            
        Returns:
            output: Predicted coordinates [batch_size, output_dim]
            perturbed_output: Perturbed predictions (optional)
        """
        # Standard forward pass
        output = self.base_model(x, edge_index, batch)
        
        if return_perturbed:
            # Generate adversarial perturbation
            perturbed_x = self.base_model.get_adversarial_perturbation(x, edge_index, batch, self.epsilon)
            
            # Forward pass with perturbed input
            perturbed_output = self.base_model(perturbed_x, edge_index, batch)
            
            return output, perturbed_output
        
        return output
    
    def adversarial_loss(self, x, edge_index, batch=None, alpha=0.01, beta=0.5):
        """
        Calculate adversarial loss
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes]
            alpha: Weight for consistency loss
            beta: Weight for adversarial loss
            
        Returns:
            clean_output: Predictions on clean input
            adv_loss: Adversarial loss component
        """
        # Standard forward pass
        clean_output = self.base_model(x, edge_index, batch)
        
        # Generate adversarial perturbation
        perturbed_x = self.base_model.get_adversarial_perturbation(x, edge_index, batch, self.epsilon)
        
        # Consistency loss
        consistency_loss = self.base_model.consistency_loss(x, perturbed_x, edge_index, batch)
        
        # Total adversarial loss
        adv_loss = alpha * consistency_loss
        
        return clean_output, adv_loss
