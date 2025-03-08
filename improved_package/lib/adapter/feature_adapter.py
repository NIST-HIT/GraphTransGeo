#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Feature adapter for cross-dataset compatibility

import torch
import torch.nn as nn
import numpy as np

class FeatureAdapter(nn.Module):
    """
    Feature adapter for handling different input dimensions across datasets
    """
    def __init__(self, input_dim, target_dim=30, strategy='projection'):
        super(FeatureAdapter, self).__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.strategy = strategy
        
        # Create adapter based on strategy
        if strategy == 'projection':
            # Linear projection to target dimension
            self.adapter = nn.Linear(input_dim, target_dim)
        elif strategy == 'padding':
            # No adapter needed, padding is done in forward pass
            self.adapter = nn.Identity()
        elif strategy == 'truncation':
            # No adapter needed, truncation is done in forward pass
            self.adapter = nn.Identity()
        else:
            raise ValueError(f"Unknown adaptation strategy: {strategy}")
    
    def forward(self, x):
        """
        Adapt input features to target dimension
        
        Args:
            x: Input features [num_nodes, input_dim]
            
        Returns:
            adapted_x: Adapted features [num_nodes, target_dim]
        """
        if self.strategy == 'projection':
            # Linear projection
            return self.adapter(x)
        elif self.strategy == 'padding':
            # Padding with zeros
            if self.input_dim >= self.target_dim:
                # Truncate if input dimension is larger
                return x[:, :self.target_dim]
            else:
                # Pad with zeros if input dimension is smaller
                padding = torch.zeros(x.size(0), self.target_dim - self.input_dim, device=x.device)
                return torch.cat([x, padding], dim=1)
        elif self.strategy == 'truncation':
            # Truncation to target dimension
            return x[:, :min(self.input_dim, self.target_dim)]
        else:
            raise ValueError(f"Unknown adaptation strategy: {strategy}")

class AdaptiveFeatureProcessor(nn.Module):
    """
    Adaptive feature processor for handling different datasets
    """
    def __init__(self, dataset_dims={'New_York': 30, 'Shanghai': 51, 'Los_Angeles': 30}, 
                 target_dim=30, strategy='projection'):
        super(AdaptiveFeatureProcessor, self).__init__()
        self.dataset_dims = dataset_dims
        self.target_dim = target_dim
        self.strategy = strategy
        
        # Create adapters for each dataset
        self.adapters = nn.ModuleDict()
        for dataset, dim in dataset_dims.items():
            if dim != target_dim:
                self.adapters[dataset] = FeatureAdapter(dim, target_dim, strategy)
    
    def forward(self, x, dataset):
        """
        Process features based on dataset
        
        Args:
            x: Input features [num_nodes, input_dim]
            dataset: Dataset name
            
        Returns:
            processed_x: Processed features [num_nodes, target_dim]
        """
        if dataset in self.adapters:
            return self.adapters[dataset](x)
        else:
            # No adaptation needed
            return x
