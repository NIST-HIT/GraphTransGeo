#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Examine data structure in NPZ file in detail

import os
import numpy as np
import sys

def examine_data_structure(file_path):
    """
    Examine the structure of data in an NPZ file in detail
    
    Args:
        file_path: Path to the NPZ file
    """
    print(f"Examining data structure in NPZ file: {file_path}")
    
    # Load NPZ file
    data = np.load(file_path, allow_pickle=True)
    
    # Print available keys
    print("\nAvailable keys in NPZ file:")
    for key in data.keys():
        print(f"- {key}")
    
    # Get the data array
    data_array = data['data']
    print(f"\nData array shape: {data_array.shape}")
    print(f"Data array type: {type(data_array)}")
    
    # Examine the first few items
    print("\nExamining first few items:")
    for i in range(min(5, len(data_array))):
        print(f"\nItem {i}:")
        item = data_array[i]
        print(f"Item type: {type(item)}")
        
        # If item is a dictionary, examine its keys
        if isinstance(item, dict):
            print(f"Item keys: {list(item.keys())}")
            
            # Examine each key in the dictionary
            for key in item.keys():
                print(f"\n  Key: {key}")
                value = item[key]
                print(f"  Value type: {type(value)}")
                
                # If value is an array, print its shape and sample
                if hasattr(value, 'shape'):
                    print(f"  Value shape: {value.shape}")
                    print(f"  Value sample: {value[:2] if len(value) > 1 else value}")
                else:
                    print(f"  Value: {value}")
    
    # Check if there are any common keys across all items
    if len(data_array) > 0 and isinstance(data_array[0], dict):
        common_keys = set(data_array[0].keys())
        for i in range(1, len(data_array)):
            if isinstance(data_array[i], dict):
                common_keys = common_keys.intersection(set(data_array[i].keys()))
        
        print(f"\nCommon keys across all items: {common_keys}")
        
        # Examine the structure of common keys
        for key in common_keys:
            print(f"\nExamining common key: {key}")
            
            # Check if values for this key have consistent types
            value_types = set()
            for i in range(len(data_array)):
                if isinstance(data_array[i], dict) and key in data_array[i]:
                    value_types.add(type(data_array[i][key]))
            
            print(f"Value types for key '{key}': {value_types}")
            
            # If values are arrays, check if they have consistent shapes
            if np.ndarray in value_types:
                shapes = set()
                for i in range(len(data_array)):
                    if isinstance(data_array[i], dict) and key in data_array[i] and isinstance(data_array[i][key], np.ndarray):
                        shapes.add(data_array[i][key].shape)
                
                print(f"Array shapes for key '{key}': {shapes}")

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
    
    # Examine data structure
    examine_data_structure(file_path)
