#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Test script for feature adapter

import os
import torch
import numpy as np
import random
from lib.adapter.feature_adapter import FeatureAdapter, AdaptiveFeatureProcessor

def test_feature_adapter_projection():
    """Test the feature adapter with projection strategy"""
    print("Testing Feature Adapter (Projection)...")
    
    # Create random features
    num_nodes = 20
    input_dim = 51  # Shanghai dataset
    target_dim = 30  # New York dataset
    x = torch.rand(num_nodes, input_dim)
    
    # Create feature adapter
    adapter = FeatureAdapter(input_dim, target_dim, strategy='projection')
    
    # Adapt features
    adapted_x = adapter(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Adapted shape: {adapted_x.shape}")
    print(f"  Adapted mean: {adapted_x.mean().item():.4f}")
    print(f"  Adapted std: {adapted_x.std().item():.4f}")

def test_feature_adapter_padding():
    """Test the feature adapter with padding strategy"""
    print("\nTesting Feature Adapter (Padding)...")
    
    # Create random features
    num_nodes = 20
    input_dim = 20  # Smaller than target
    target_dim = 30  # New York dataset
    x = torch.rand(num_nodes, input_dim)
    
    # Create feature adapter
    adapter = FeatureAdapter(input_dim, target_dim, strategy='padding')
    
    # Adapt features
    adapted_x = adapter(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Adapted shape: {adapted_x.shape}")
    print(f"  Adapted mean: {adapted_x.mean().item():.4f}")
    print(f"  Adapted std: {adapted_x.std().item():.4f}")
    
    # Check if padding is correct
    padding_correct = torch.all(adapted_x[:, input_dim:] == 0).item()
    print(f"  Padding correct: {padding_correct}")

def test_feature_adapter_truncation():
    """Test the feature adapter with truncation strategy"""
    print("\nTesting Feature Adapter (Truncation)...")
    
    # Create random features
    num_nodes = 20
    input_dim = 51  # Shanghai dataset
    target_dim = 30  # New York dataset
    x = torch.rand(num_nodes, input_dim)
    
    # Create feature adapter
    adapter = FeatureAdapter(input_dim, target_dim, strategy='truncation')
    
    # Adapt features
    adapted_x = adapter(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Adapted shape: {adapted_x.shape}")
    print(f"  Adapted mean: {adapted_x.mean().item():.4f}")
    print(f"  Adapted std: {adapted_x.std().item():.4f}")
    
    # Check if truncation is correct
    truncation_correct = torch.all(adapted_x == x[:, :target_dim]).item()
    print(f"  Truncation correct: {truncation_correct}")

def test_adaptive_feature_processor():
    """Test the adaptive feature processor"""
    print("\nTesting Adaptive Feature Processor...")
    
    # Create random features
    num_nodes = 20
    dataset_dims = {'New_York': 30, 'Shanghai': 51, 'Los_Angeles': 30}
    target_dim = 30
    
    # Create adaptive feature processor
    processor = AdaptiveFeatureProcessor(dataset_dims, target_dim, strategy='projection')
    
    # Test for each dataset
    for dataset, dim in dataset_dims.items():
        x = torch.rand(num_nodes, dim)
        processed_x = processor(x, dataset)
        
        print(f"  Dataset: {dataset}")
        print(f"  Input shape: {x.shape}")
        print(f"  Processed shape: {processed_x.shape}")
        print(f"  Processed mean: {processed_x.mean().item():.4f}")
        print(f"  Processed std: {processed_x.std().item():.4f}")

def main():
    """Main function to run all tests"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Run tests
    test_feature_adapter_projection()
    test_feature_adapter_padding()
    test_feature_adapter_truncation()
    test_adaptive_feature_processor()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()
