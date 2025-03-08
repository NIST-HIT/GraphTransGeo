#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Test feature adapter with real data

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
import argparse

from lib.adapters.feature_adapter import FeatureAdapter, AdaptiveGraphTransGeoGCN, create_adaptive_model
from lib.model_graphtransgeo_gcn_optimized import EnsembleGraphTransGeoPlusPlus
from gcn_data_loader import IPGraphDataset, load_ip_graph_data

def test_adapter_with_datasets():
    """Test the feature adapter with real datasets"""
    print("Testing Feature Adapter with Real Datasets...")
    
    # Load New York dataset (30 features)
    print("\n1. Loading New York dataset...")
    ny_train_loader, ny_val_loader, ny_test_loader = load_ip_graph_data('New_York', batch_size=32)
    
    # Get a batch from New York dataset
    ny_batch = next(iter(ny_test_loader))
    ny_x = ny_batch.x
    ny_edge_index = ny_batch.edge_index
    ny_batch_idx = ny_batch.batch
    ny_y = ny_batch.y
    
    print(f"  New York features shape: {ny_x.shape}")
    print(f"  New York edge_index shape: {ny_edge_index.shape}")
    print(f"  New York batch shape: {ny_batch_idx.shape}")
    print(f"  New York labels shape: {ny_y.shape}")
    
    # Try to load Shanghai dataset (51 features)
    print("\n2. Loading Shanghai dataset...")
    try:
        sh_train_loader, sh_val_loader, sh_test_loader = load_ip_graph_data('Shanghai', batch_size=32)
        
        # Get a batch from Shanghai dataset
        sh_batch = next(iter(sh_test_loader))
        sh_x = sh_batch.x
        sh_edge_index = sh_batch.edge_index
        sh_batch_idx = sh_batch.batch
        sh_y = sh_batch.y
        
        print(f"  Shanghai features shape: {sh_x.shape}")
        print(f"  Shanghai edge_index shape: {sh_edge_index.shape}")
        print(f"  Shanghai batch shape: {sh_batch_idx.shape}")
        print(f"  Shanghai labels shape: {sh_y.shape}")
        
        # Create feature adapter for Shanghai -> New York (51 -> 30)
        print("\n3. Testing Feature Adapter (Shanghai -> New York)...")
        adapter = FeatureAdapter(sh_x.shape[1], ny_x.shape[1], strategy='linear')
        
        # Adapt Shanghai features to New York dimensions
        adapted_sh_x = adapter(sh_x)
        print(f"  Original Shanghai features shape: {sh_x.shape}")
        print(f"  Adapted Shanghai features shape: {adapted_sh_x.shape}")
        print(f"  Adaptation successful: {adapted_sh_x.shape[1] == ny_x.shape[1]}")
        
        # Create a base GraphTransGeo model for New York
        print("\n4. Testing Adaptive Model with Shanghai data...")
        base_model = EnsembleGraphTransGeoPlusPlus(
            input_dim=ny_x.shape[1],
            hidden_dim=256,
            output_dim=2,
            ensemble_size=2
        )
        
        # Create an adaptive model for Shanghai
        adaptive_model = AdaptiveGraphTransGeoGCN(
            base_model=base_model,
            input_dim=sh_x.shape[1],
            target_dim=ny_x.shape[1],
            adapter_strategy='linear'
        )
        
        # Test forward pass with Shanghai data
        output = adaptive_model(sh_x, sh_edge_index, sh_batch_idx)
        print(f"  Input shape: {sh_x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output mean: {output.mean().item():.4f}")
        print(f"  Output std: {output.std().item():.4f}")
        
        # Calculate loss
        loss = F.mse_loss(output, sh_y)
        print(f"  MSE Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"  Error loading Shanghai dataset: {e}")
        print("  Simulating Shanghai dataset with random data...")
        
        # Create random data with 51 features
        batch_size = 32
        num_nodes = 100
        sh_x = torch.rand(num_nodes, 51)
        sh_edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        sh_batch_idx = torch.zeros(num_nodes, dtype=torch.long)
        sh_y = torch.rand(batch_size, 2)
        
        print(f"  Simulated Shanghai features shape: {sh_x.shape}")
        
        # Create feature adapter for Shanghai -> New York (51 -> 30)
        print("\n3. Testing Feature Adapter (Shanghai -> New York)...")
        adapter = FeatureAdapter(sh_x.shape[1], ny_x.shape[1], strategy='linear')
        
        # Adapt Shanghai features to New York dimensions
        adapted_sh_x = adapter(sh_x)
        print(f"  Original Shanghai features shape: {sh_x.shape}")
        print(f"  Adapted Shanghai features shape: {adapted_sh_x.shape}")
        print(f"  Adaptation successful: {adapted_sh_x.shape[1] == ny_x.shape[1]}")
        
        # Create a base GraphTransGeo model for New York
        print("\n4. Testing Adaptive Model with simulated Shanghai data...")
        base_model = EnsembleGraphTransGeoPlusPlus(
            input_dim=ny_x.shape[1],
            hidden_dim=256,
            output_dim=2,
            ensemble_size=2
        )
        
        # Create an adaptive model for Shanghai
        adaptive_model = AdaptiveGraphTransGeoGCN(
            base_model=base_model,
            input_dim=sh_x.shape[1],
            target_dim=ny_x.shape[1],
            adapter_strategy='linear'
        )
        
        # Test forward pass with simulated Shanghai data
        output = adaptive_model(sh_x, sh_edge_index, sh_batch_idx)
        print(f"  Input shape: {sh_x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output mean: {output.mean().item():.4f}")
        print(f"  Output std: {output.std().item():.4f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test adapter with real datasets
    test_adapter_with_datasets()
