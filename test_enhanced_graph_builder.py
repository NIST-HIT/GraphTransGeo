#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Test script for enhanced graph builder

import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from lib.graph_construction.enhanced_graph_builder import EnhancedGraphBuilder, DynamicGraphBuilder

def test_enhanced_graph_builder():
    """Test the enhanced graph builder with synthetic data"""
    print("Testing Enhanced Graph Builder...")
    
    # Create random node features
    num_nodes = 20
    feature_dim = 30
    x = torch.rand(num_nodes, feature_dim)
    
    # Create random geographic coordinates
    coords = torch.rand(num_nodes, 2)
    coords[:, 0] = coords[:, 0] * 360 - 180  # Longitude: -180 to 180
    coords[:, 1] = coords[:, 1] * 180 - 90   # Latitude: -90 to 90
    
    # Create a synthetic data item
    item = {
        'lm_X': x.numpy(),
        'lm_Y': coords.numpy(),
        'router': np.zeros((num_nodes, num_nodes))  # Dummy router data
    }
    
    # Create enhanced graph builder
    builder = EnhancedGraphBuilder(
        use_geographic=True,
        use_multi_scale=True,
        use_edge_weights=True
    )
    
    # Test with different configurations
    print("\n1. Testing with all features enabled:")
    result = builder.build_graph(x, item, k=5)
    
    if isinstance(result, tuple):
        edge_index, edge_attr = result
        print(f"  Edge index shape: {edge_index.shape}")
        print(f"  Edge attributes shape: {edge_attr.shape}")
        print(f"  Number of edges: {edge_index.shape[1]}")
        print(f"  Edge weight range: [{edge_attr.min().item():.4f}, {edge_attr.max().item():.4f}]")
    else:
        edge_index = result
        print(f"  Edge index shape: {edge_index.shape}")
        print(f"  Number of edges: {edge_index.shape[1]}")
    
    # Test with geographic features disabled
    print("\n2. Testing with geographic features disabled:")
    builder = EnhancedGraphBuilder(
        use_geographic=False,
        use_multi_scale=True,
        use_edge_weights=False
    )
    
    edge_index = builder.build_graph(x, item, k=5)
    print(f"  Edge index shape: {edge_index.shape}")
    print(f"  Number of edges: {edge_index.shape[1]}")
    
    # Test with multi-scale features disabled
    print("\n3. Testing with multi-scale features disabled:")
    builder = EnhancedGraphBuilder(
        use_geographic=True,
        use_multi_scale=False,
        use_edge_weights=True
    )
    
    result = builder.build_graph(x, item, k=5)
    
    if isinstance(result, tuple):
        edge_index, edge_attr = result
        print(f"  Edge index shape: {edge_index.shape}")
        print(f"  Edge attributes shape: {edge_attr.shape}")
        print(f"  Number of edges: {edge_index.shape[1]}")
    else:
        edge_index = result
        print(f"  Edge index shape: {edge_index.shape}")
        print(f"  Number of edges: {edge_index.shape[1]}")
    
    # Visualize the graph
    print("\n4. Visualizing the graph:")
    visualize_graph(coords, edge_index, "enhanced_graph.png")
    print(f"  Graph visualization saved to enhanced_graph.png")

def test_dynamic_graph_builder():
    """Test the dynamic graph builder with synthetic data"""
    print("\nTesting Dynamic Graph Builder...")
    
    # Create random node features
    num_nodes = 20
    feature_dim = 30
    x1 = torch.rand(num_nodes, feature_dim)
    x2 = torch.rand(num_nodes, feature_dim)  # Different features for update
    
    # Create random geographic coordinates
    coords = torch.rand(num_nodes, 2)
    coords[:, 0] = coords[:, 0] * 360 - 180  # Longitude: -180 to 180
    coords[:, 1] = coords[:, 1] * 180 - 90   # Latitude: -90 to 90
    
    # Create a synthetic data item
    item = {
        'lm_X': x1.numpy(),
        'lm_Y': coords.numpy(),
        'router': np.zeros((num_nodes, num_nodes))  # Dummy router data
    }
    
    # Create enhanced graph builder
    initial_builder = EnhancedGraphBuilder(
        use_geographic=True,
        use_multi_scale=True,
        use_edge_weights=True
    )
    
    # Create dynamic graph builder
    dynamic_builder = DynamicGraphBuilder(
        initial_builder=initial_builder,
        update_interval=2
    )
    
    # Test initial graph creation
    print("\n1. Testing initial graph creation:")
    result = dynamic_builder(x1, item)
    
    if isinstance(result, tuple):
        edge_index, edge_attr = result
        print(f"  Edge index shape: {edge_index.shape}")
        print(f"  Edge attributes shape: {edge_attr.shape}")
        print(f"  Number of edges: {edge_index.shape[1]}")
    else:
        edge_index = result
        print(f"  Edge index shape: {edge_index.shape}")
        print(f"  Number of edges: {edge_index.shape[1]}")
    
    # Test graph update
    print("\n2. Testing graph update:")
    result = dynamic_builder(x2, item, edge_index)
    
    if isinstance(result, tuple):
        updated_edge_index, updated_edge_attr = result
        print(f"  Updated edge index shape: {updated_edge_index.shape}")
        print(f"  Updated edge attributes shape: {updated_edge_attr.shape}")
        print(f"  Number of edges after update: {updated_edge_index.shape[1]}")
    else:
        updated_edge_index = result
        print(f"  Updated edge index shape: {updated_edge_index.shape}")
        print(f"  Number of edges after update: {updated_edge_index.shape[1]}")
    
    # Test another update (should create new graph due to update interval)
    print("\n3. Testing another update (should create new graph):")
    result = dynamic_builder(x2, item, updated_edge_index)
    
    if isinstance(result, tuple):
        new_edge_index, new_edge_attr = result
        print(f"  New edge index shape: {new_edge_index.shape}")
        print(f"  New edge attributes shape: {new_edge_attr.shape}")
        print(f"  Number of edges in new graph: {new_edge_index.shape[1]}")
    else:
        new_edge_index = result
        print(f"  New edge index shape: {new_edge_index.shape}")
        print(f"  Number of edges in new graph: {new_edge_index.shape[1]}")

def visualize_graph(coords, edge_index, filename):
    """Visualize the graph structure"""
    plt.figure(figsize=(10, 8))
    
    # Plot nodes
    plt.scatter(coords[:, 0], coords[:, 1], c='blue', s=50, alpha=0.8)
    
    # Plot edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src != dst:  # Skip self-loops
            plt.plot([coords[src, 0], coords[dst, 0]], 
                     [coords[src, 1], coords[dst, 1]], 
                     'k-', alpha=0.2)
    
    # Add labels
    for i in range(coords.shape[0]):
        plt.text(coords[i, 0], coords[i, 1], str(i), fontsize=8)
    
    plt.title('Graph Visualization')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(alpha=0.3)
    
    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Test enhanced graph builder
    test_enhanced_graph_builder()
    
    # Test dynamic graph builder
    test_dynamic_graph_builder()
