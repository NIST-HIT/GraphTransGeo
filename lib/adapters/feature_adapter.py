#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Feature adapter for cross-dataset compatibility in GraphTransGeo++

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeatureAdapter(nn.Module):
    """
    Feature adapter module for handling different input dimensions across datasets
    
    This module can transform features from one dimension to another, supporting both
    dimension reduction and expansion. It uses different strategies based on the
    input and target dimensions.
    """
    def __init__(self, input_dim, target_dim, strategy='linear'):
        super(FeatureAdapter, self).__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.strategy = strategy
        
        # Create appropriate adapter based on dimensions and strategy
        if input_dim == target_dim:
            # No adaptation needed
            self.adapter = nn.Identity()
        elif input_dim > target_dim:
            # Dimension reduction
            if strategy == 'linear':
                # Simple linear projection
                self.adapter = nn.Linear(input_dim, target_dim)
            elif strategy == 'mlp':
                # MLP with intermediate layer
                hidden_dim = (input_dim + target_dim) // 2
                self.adapter = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, target_dim)
                )
            elif strategy == 'selection':
                # Feature selection (select most important features)
                # This is implemented in the forward pass
                self.adapter = None
                # Initialize feature importance (learnable)
                self.feature_importance = nn.Parameter(torch.ones(input_dim))
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        else:
            # Dimension expansion
            if strategy == 'linear':
                # Simple linear projection
                self.adapter = nn.Linear(input_dim, target_dim)
            elif strategy == 'mlp':
                # MLP with intermediate layer
                hidden_dim = (input_dim + target_dim) // 2
                self.adapter = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, target_dim)
                )
            elif strategy == 'padding':
                # Padding with zeros or learned values
                # This is implemented in the forward pass
                self.adapter = None
                # Initialize padding values (learnable)
                self.padding_values = nn.Parameter(torch.zeros(target_dim - input_dim))
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Transform features from input dimension to target dimension
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            transformed_x: Transformed features [batch_size, target_dim]
        """
        if self.input_dim == self.target_dim:
            # No adaptation needed
            return x
        elif self.input_dim > self.target_dim:
            # Dimension reduction
            if self.strategy == 'selection':
                # Select most important features based on learned importance
                importance = F.softmax(self.feature_importance, dim=0)
                _, indices = torch.topk(importance, self.target_dim)
                indices, _ = torch.sort(indices)  # Sort indices to maintain order
                return x[:, indices]
            else:
                # Use the adapter
                return self.adapter(x)
        else:
            # Dimension expansion
            if self.strategy == 'padding':
                # Pad with learned values
                batch_size = x.size(0)
                padding = self.padding_values.unsqueeze(0).expand(batch_size, -1)
                return torch.cat([x, padding], dim=1)
            else:
                # Use the adapter
                return self.adapter(x)


class AdaptiveGraphTransGeoGCN(nn.Module):
    """
    Adaptive GraphTransGeo GCN model with feature adapter
    
    This model wraps a GCN model with a feature adapter to handle
    different input dimensions across datasets.
    """
    def __init__(self, base_model, input_dim, target_dim, adapter_strategy='linear'):
        super(AdaptiveGraphTransGeoGCN, self).__init__()
        self.feature_adapter = FeatureAdapter(input_dim, target_dim, strategy=adapter_strategy)
        self.base_model = base_model
    
    def forward(self, x, edge_index, batch=None, return_perturbed=False):
        """
        Forward pass with feature adaptation
        
        Args:
            x: Input features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes]
            return_perturbed: Whether to return perturbed predictions
            
        Returns:
            output: Predicted coordinates [batch_size, output_dim]
            perturbed_output: Perturbed predictions (optional)
        """
        # Adapt features to target dimension
        x_adapted = self.feature_adapter(x)
        
        # Forward through base model
        if return_perturbed:
            output, perturbed_output = self.base_model(x_adapted, edge_index, batch, return_perturbed)
            return output, perturbed_output
        else:
            output = self.base_model(x_adapted, edge_index, batch)
            return output
    
    def adversarial_loss(self, x, edge_index, batch=None, alpha=0.01, beta=0.5):
        """
        Calculate adversarial loss with feature adaptation
        
        Args:
            x: Input features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes]
            alpha: Weight for consistency loss
            beta: Weight for adversarial loss
            
        Returns:
            clean_output: Predictions on clean input
            adv_loss: Adversarial loss component
        """
        # Adapt features to target dimension
        x_adapted = self.feature_adapter(x)
        
        # Calculate adversarial loss using base model
        return self.base_model.adversarial_loss(x_adapted, edge_index, batch, alpha, beta)


def create_adaptive_model(model_path, source_dim, target_dim, model_class, adapter_strategy='linear', **model_kwargs):
    """
    Create an adaptive model that can handle different input dimensions
    
    Args:
        model_path: Path to the pre-trained model
        source_dim: Original input dimension
        target_dim: Target input dimension
        model_class: Model class to instantiate
        adapter_strategy: Strategy for feature adaptation
        **model_kwargs: Additional arguments for model instantiation
        
    Returns:
        adaptive_model: Model with feature adapter
    """
    # Create base model with target dimension
    base_model = model_class(input_dim=target_dim, **model_kwargs)
    
    # Load pre-trained weights if model_path is provided
    if model_path and os.path.exists(model_path):
        base_model.load_state_dict(torch.load(model_path))
    
    # Create adaptive model
    adaptive_model = AdaptiveGraphTransGeoGCN(
        base_model=base_model,
        input_dim=source_dim,
        target_dim=target_dim,
        adapter_strategy=adapter_strategy
    )
    
    return adaptive_model
