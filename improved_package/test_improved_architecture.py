#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Test script for improved model architecture

import os
import torch
import numpy as np
import random
from lib.model.improved_architecture import (
    ResidualGCNLayer,
    MultiHeadGATLayer,
    ImprovedGraphTransGeoGCN,
    ImprovedGraphTransGeoGAT,
    HybridGraphTransGeo,
    EnhancedEnsembleGraphTransGeoPlusPlus
)

def test_residual_gcn_layer():
    """Test the residual GCN layer"""
    print("Testing Residual GCN Layer...")
    
    # Create random node features and edge index
    num_nodes = 20
    in_channels = 32
    out_channels = 64
    x = torch.rand(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    edge_weight = torch.rand(edge_index.size(1))
    
    # Create residual GraphTransGeo layer
    layer = ResidualGCNLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        use_layer_norm=True,
        dropout=0.3
    )
    
    # Test forward pass
    output = layer(x, edge_index, edge_weight)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.4f}")
    print(f"  Output std: {output.std().item():.4f}")

def test_multi_head_gat_layer():
    """Test the multi-head GAT layer"""
    print("\nTesting Multi-Head GAT Layer...")
    
    # Create random node features and edge index
    num_nodes = 20
    in_channels = 32
    out_channels = 64
    x = torch.rand(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    
    # Create multi-head GAT layer
    layer = MultiHeadGATLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        heads=4,
        use_layer_norm=True,
        dropout=0.3
    )
    
    # Test forward pass
    output = layer(x, edge_index)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.4f}")
    print(f"  Output std: {output.std().item():.4f}")

def test_improved_gcn_model():
    """Test the improved GCN model"""
    print("\nTesting Improved GCN Model...")
    
    # Create random node features and edge index
    num_nodes = 20
    input_dim = 30
    hidden_dim = 64
    output_dim = 2
    x = torch.rand(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    edge_weight = torch.rand(edge_index.size(1))
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    # Create improved GraphTransGeo model
    model = ImprovedGraphTransGeoGCN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=0.3,
        num_layers=3,
        heads=4,
        use_layer_norm=True
    )
    
    # Test forward pass
    output = model(x, edge_index, edge_weight, batch)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.4f}")
    print(f"  Output std: {output.std().item():.4f}")
    
    # Test model parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Number of parameters: {num_params}")

def test_improved_gat_model():
    """Test the improved GAT model"""
    print("\nTesting Improved GAT Model...")
    
    # Create random node features and edge index
    num_nodes = 20
    input_dim = 30
    hidden_dim = 64
    output_dim = 2
    x = torch.rand(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    # Create improved GAT model
    model = ImprovedGraphTransGeoGAT(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=0.3,
        num_layers=3,
        heads=4,
        use_layer_norm=True
    )
    
    # Test forward pass
    output = model(x, edge_index, batch)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.4f}")
    print(f"  Output std: {output.std().item():.4f}")
    
    # Test model parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Number of parameters: {num_params}")

def test_hybrid_model():
    """Test the hybrid model"""
    print("\nTesting Hybrid Model...")
    
    # Create random node features and edge index
    num_nodes = 20
    input_dim = 30
    hidden_dim = 64
    output_dim = 2
    x = torch.rand(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    edge_weight = torch.rand(edge_index.size(1))
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    # Create hybrid model
    model = HybridGraphTransGeo(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=0.3,
        num_layers=3,
        heads=4,
        use_layer_norm=True
    )
    
    # Test forward pass
    output = model(x, edge_index, edge_weight, batch)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.4f}")
    print(f"  Output std: {output.std().item():.4f}")
    
    # Test model parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Number of parameters: {num_params}")
    
    # Test alpha parameter
    alpha = torch.sigmoid(model.alpha)
    print(f"  Alpha parameter: {alpha.item():.4f}")

def test_enhanced_ensemble_model():
    """Test the enhanced ensemble model"""
    print("\nTesting Enhanced Ensemble Model...")
    
    # Create random node features and edge index
    num_nodes = 20
    input_dim = 30
    hidden_dim = 64
    output_dim = 2
    x = torch.rand(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    edge_weight = torch.rand(edge_index.size(1))
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    # Create enhanced ensemble model
    model = EnhancedEnsembleGraphTransGeoPlusPlus(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=0.3,
        num_layers=3,
        ensemble_size=3,
        epsilon=0.01
    )
    
    # Test forward pass
    output = model(x, edge_index, edge_weight, batch)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.4f}")
    print(f"  Output std: {output.std().item():.4f}")
    
    # Test model parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Number of parameters: {num_params}")
    
    # Test ensemble weights
    weights = torch.softmax(model.ensemble_weights / model.temperature, dim=0)
    print(f"  Ensemble weights: {weights.detach().numpy()}")
    
    # Test adversarial perturbation
    perturbed_x = model.get_adversarial_perturbation(x, edge_index, edge_weight, batch, model.epsilon)
    print(f"  Perturbed input shape: {perturbed_x.shape}")
    print(f"  Perturbation magnitude: {torch.norm(perturbed_x - x).item():.4f}")
    
    # Test adversarial loss
    clean_output, adv_loss = model.adversarial_loss(x, edge_index, edge_weight, batch, alpha=0.01, beta=0.5)
    print(f"  Adversarial loss: {adv_loss.item():.4f}")

def main():
    """Main function to run all tests"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Run tests
    test_residual_gcn_layer()
    test_multi_head_gat_layer()
    test_improved_gcn_model()
    test_improved_gat_model()
    test_hybrid_model()
    test_enhanced_ensemble_model()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()
