#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Data loader for GraphTransGeo model

import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import logging
import math

class GraphDataset(InMemoryDataset):
    """
    Graph dataset for GCN model
    """
    def __init__(self, features, labels, transform=None, pre_transform=None, k=10):
        """
        Initialize dataset
        
        Args:
            features: Node features [num_nodes, num_features]
            labels: Node labels [num_nodes, num_labels]
            transform: Transform to apply to data
            pre_transform: Transform to apply to data before saving
            k: Number of neighbors for k-NN
        """
        super(GraphDataset, self).__init__(None, transform, pre_transform)
        
        # Create data
        data_list = []
        
        # Process each sample as a separate graph
        for i in range(len(features)):
            # Create feature and label tensors for this sample
            x = torch.tensor(features[i:i+1], dtype=torch.float)
            y = torch.tensor(labels[i:i+1], dtype=torch.float)
            
            # Create a simple self-loop edge index for each node
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            
            # Create data object
            data = Data(x=x, y=y, edge_index=edge_index)
            data_list.append(data)
        
        self.data, self.slices = self.collate(data_list)
    
    def _download(self):
        pass
    
    def _process(self):
        pass

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate Haversine distance between two points
    
    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point
        lat2: Latitude of second point
        lon2: Longitude of second point
        
    Returns:
        distance: Haversine distance in kilometers
    """
    # Convert to radians
    lat1 = lat1 * math.pi / 180
    lon1 = lon1 * math.pi / 180
    lat2 = lat2 * math.pi / 180
    lon2 = lon2 * math.pi / 180
    
    # Calculate Haversine distance
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    
    return c * r

def load_dataset(dataset_name, data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Load dataset
    
    Args:
        dataset_name: Dataset name
        data_dir: Data directory
        train_ratio: Training ratio
        val_ratio: Validation ratio
        test_ratio: Test ratio
        seed: Random seed
        
    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        input_dim: Input dimension
    """
    # Set random seed
    np.random.seed(seed)
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Load dataset
    try:
        # Try to load dataset from file
        data_path = os.path.join(data_dir, f'{dataset_name}.npz')
        
        if os.path.exists(data_path):
            # Load dataset from file
            data = np.load(data_path, allow_pickle=True)
            
            # Get features and labels
            features = data['features']
            labels = data['labels']
            
            # Get input dimension
            input_dim = features.shape[1]
            
            logging.info(f'Dataset loaded from {data_path}')
            logging.info(f'Features shape: {features.shape}')
            logging.info(f'Labels shape: {labels.shape}')
        else:
            # Create dummy dataset for testing
            logging.warning(f'Dataset not found: {data_path}')
            logging.warning('Creating dummy dataset for testing')
            
            # Set input dimension based on dataset
            if dataset_name == 'New_York':
                input_dim = 30
            elif dataset_name == 'Shanghai':
                input_dim = 51
            elif dataset_name == 'Los_Angeles':
                input_dim = 30
            else:
                input_dim = 30
            
            # Create dummy features and labels
            num_samples = 1000
            features = np.random.randn(num_samples, input_dim)
            labels = np.random.uniform(-90, 90, (num_samples, 2))  # Latitude and longitude
            
            # Save dataset
            np.savez(data_path, features=features, labels=labels)
            
            logging.info(f'Dummy dataset saved to {data_path}')
            logging.info(f'Features shape: {features.shape}')
            logging.info(f'Labels shape: {labels.shape}')
    
    except Exception as e:
        logging.error(f'Error loading dataset: {e}')
        
        # Create dummy dataset for testing
        logging.warning('Creating dummy dataset for testing')
        
        # Set input dimension based on dataset
        if dataset_name == 'New_York':
            input_dim = 30
        elif dataset_name == 'Shanghai':
            input_dim = 51
        elif dataset_name == 'Los_Angeles':
            input_dim = 30
        else:
            input_dim = 30
        
        # Create dummy features and labels
        num_samples = 1000
        features = np.random.randn(num_samples, input_dim)
        labels = np.random.uniform(-90, 90, (num_samples, 2))  # Latitude and longitude
    
    # Split dataset
    num_samples = len(features)
    indices = np.random.permutation(num_samples)
    
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Get train, validation, and test sets
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    
    val_features = features[val_indices]
    val_labels = labels[val_indices]
    
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    
    # Create datasets
    train_dataset = GraphDataset(train_features, train_labels)
    val_dataset = GraphDataset(val_features, val_labels)
    test_dataset = GraphDataset(test_features, test_labels)
    
    return train_dataset, val_dataset, test_dataset, input_dim
