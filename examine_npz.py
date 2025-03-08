#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Examine NPZ file structure

import os
import numpy as np
import sys

def examine_npz_file(file_path):
    """
    Examine the structure of an NPZ file
    
    Args:
        file_path: Path to the NPZ file
    """
    print(f"Examining NPZ file: {file_path}")
    
    # Load NPZ file
    data = np.load(file_path, allow_pickle=True)
    
    # Print available keys
    print("\nAvailable keys:")
    for key in data.keys():
        print(f"- {key}")
    
    # Print data structure
    print("\nData structure:")
    for key in data.keys():
        if hasattr(data[key], 'shape'):
            print(f"{key} shape: {data[key].shape}")
        else:
            print(f"{key} shape: No shape attribute")
        
        print(f"{key} type: {type(data[key])}")
        
        # Print sample data
        if hasattr(data[key], '__getitem__'):
            try:
                sample = data[key][:2] if len(data[key]) > 1 else data[key]
                print(f"{key} sample: {sample}")
            except:
                print(f"{key} sample: Cannot get sample")
        else:
            print(f"{key} sample: {data[key]}")
        
        print()

if __name__ == "__main__":
    # Check if file path is provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default file path
        file_path = "datasets/New_York/Clustering_s1234_lm70_train.npz"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    # Examine NPZ file
    examine_npz_file(file_path)
