#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Create dummy data for testing

import os
import numpy as np
import logging

def create_dummy_dataset(dataset_name, num_samples=1000, seed=42):
    """
    Create dummy dataset for testing
    
    Args:
        dataset_name: Dataset name
        num_samples: Number of samples
        seed: Random seed
        
    Returns:
        features: Node features [num_samples, num_features]
        labels: Node labels [num_samples, num_labels]
    """
    # Set random seed
    np.random.seed(seed)
    
    # Set input dimension based on dataset
    if dataset_name == 'New_York':
        input_dim = 30
    elif dataset_name == 'Shanghai':
        input_dim = 51
    elif dataset_name == 'Los_Angeles':
        input_dim = 30
    else:
        input_dim = 30
    
    # Create dummy features
    features = np.random.randn(num_samples, input_dim)
    
    # Create dummy labels (latitude and longitude)
    if dataset_name == 'New_York':
        # New York coordinates: 40.7128° N, 74.0060° W
        lat_mean, lon_mean = 40.7128, -74.0060
    elif dataset_name == 'Shanghai':
        # Shanghai coordinates: 31.2304° N, 121.4737° E
        lat_mean, lon_mean = 31.2304, 121.4737
    elif dataset_name == 'Los_Angeles':
        # Los Angeles coordinates: 34.0522° N, 118.2437° W
        lat_mean, lon_mean = 34.0522, -118.2437
    else:
        lat_mean, lon_mean = 0.0, 0.0
    
    # Add random noise to coordinates
    lat_std, lon_std = 0.5, 0.5
    
    # Generate labels
    labels = np.zeros((num_samples, 2))
    labels[:, 0] = np.random.normal(lat_mean, lat_std, num_samples)
    labels[:, 1] = np.random.normal(lon_mean, lon_std, num_samples)
    
    return features, labels

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Create data directory
    data_dir = 'asset/data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Create datasets
    datasets = ['New_York', 'Shanghai', 'Los_Angeles']
    
    for dataset_name in datasets:
        # Create dummy dataset
        features, labels = create_dummy_dataset(dataset_name)
        
        # Save dataset
        data_path = os.path.join(data_dir, f'{dataset_name}.npz')
        np.savez(data_path, features=features, labels=labels)
        
        logging.info(f'Dummy dataset saved to {data_path}')
        logging.info(f'Features shape: {features.shape}')
        logging.info(f'Labels shape: {labels.shape}')

if __name__ == '__main__':
    main()
