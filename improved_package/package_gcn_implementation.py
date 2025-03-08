#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Package GraphTransGeo implementation for distribution

import os
import shutil
import argparse
import zipfile
import datetime

def create_directory_structure(output_dir):
    """
    Create directory structure for the packaged implementation
    
    Args:
        output_dir: Output directory
    """
    # Create main directories
    os.makedirs(os.path.join(output_dir, 'lib'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'asset', 'model'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'asset', 'log'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'asset', 'figures'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'datasets'), exist_ok=True)

def copy_implementation_files(source_dir, output_dir):
    """
    Copy implementation files to the output directory
    
    Args:
        source_dir: Source directory
        output_dir: Output directory
    """
    # Copy library files
    lib_files = [
        'lib/model_graphtransgeo_gcn_optimized.py',
        'lib/utils.py',
    ]
    
    for file_path in lib_files:
        src_path = os.path.join(source_dir, file_path)
        dst_path = os.path.join(output_dir, file_path)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f'Copied {file_path}')
    
    # Copy main implementation files
    main_files = [
        'train_gcn_optimized.py',
        'gcn_data_loader.py',
        'run_gcn_optimized.sh',
        'visualize_gcn_results.py',
    ]
    
    for file_path in main_files:
        src_path = os.path.join(source_dir, file_path)
        dst_path = os.path.join(output_dir, file_path)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f'Copied {file_path}')
    
    # Copy model files if they exist
    model_files = [f for f in os.listdir(os.path.join(source_dir, 'asset', 'model')) if f.endswith('.pth')]
    for file_name in model_files:
        src_path = os.path.join(source_dir, 'asset', 'model', file_name)
        dst_path = os.path.join(output_dir, 'asset', 'model', file_name)
        shutil.copy2(src_path, dst_path)
        print(f'Copied asset/model/{file_name}')
    
    # Copy log files if they exist
    log_files = [f for f in os.listdir(os.path.join(source_dir, 'asset', 'log')) if f.endswith('.log') or f.endswith('.png')]
    for file_name in log_files:
        src_path = os.path.join(source_dir, 'asset', 'log', file_name)
        dst_path = os.path.join(output_dir, 'asset', 'log', file_name)
        shutil.copy2(src_path, dst_path)
        print(f'Copied asset/log/{file_name}')
    
    # Copy figure files if they exist
    figure_dir = os.path.join(source_dir, 'asset', 'figures')
    if os.path.exists(figure_dir):
        figure_files = [f for f in os.listdir(figure_dir) if f.endswith('.png')]
        for file_name in figure_files:
            src_path = os.path.join(source_dir, 'asset', 'figures', file_name)
            dst_path = os.path.join(output_dir, 'asset', 'figures', file_name)
            shutil.copy2(src_path, dst_path)
            print(f'Copied asset/figures/{file_name}')

def create_readme(output_dir):
    """
    Create README file
    
    Args:
        output_dir: Output directory
    """
    readme_content = """# GraphTransGeo++ Optimized GraphTransGeo Implementation

This package contains an optimized implementation of the GraphTransGeo++ method for IP geolocation using Graph Convolutional Networks (GCNs).

## Directory Structure

- `lib/`: Library files
  - `model_graphtransgeo_gcn_optimized.py`: Optimized GCN model implementation
  - `utils.py`: Utility functions
- `asset/`: Assets directory
  - `model/`: Trained model checkpoints
  - `log/`: Training and testing logs
  - `figures/`: Visualization figures
- `reports/`: Reports and analysis
- `datasets/`: Dataset directory (not included, please add your datasets here)

## Main Files

- `train_gcn_optimized.py`: Script for training the optimized GCN model
- `gcn_data_loader.py`: Data loader for GCN-based GraphTransGeo++
- `run_gcn_optimized.sh`: Shell script to run the optimized GCN model
- `visualize_gcn_results.py`: Script to visualize results

## Requirements

- PyTorch
- PyTorch Geometric
- NumPy
- Matplotlib
- tqdm

## Usage

1. Place your datasets in the `datasets/` directory with the following structure:
   - `datasets/New_York/Clustering_s1234_lm70_train.npz`
   - `datasets/New_York/Clustering_s1234_lm70_test.npz`
   - `datasets/Shanghai/Clustering_s1234_lm70_train.npz`
   - `datasets/Shanghai/Clustering_s1234_lm70_test.npz`
   - `datasets/Los_Angeles/Clustering_s1234_lm70_train.npz`
   - `datasets/Los_Angeles/Clustering_s1234_lm70_test.npz`

2. Run the training script:
   ```
   ./run_gcn_optimized.sh
   ```

3. Visualize the results:
   ```
   python visualize_gcn_results.py --dataset New_York
   ```

## Model Architecture

The optimized GCN model uses an ensemble of Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs) with the following key features:

- Improved graph construction with adaptive k-NN and network topology
- Haversine loss function for more accurate geolocation
- Adversarial training with consistency regularization
- Ensemble learning with learnable weights
- Residual connections and layer normalization

## Performance

The model achieves state-of-the-art performance on IP geolocation tasks, with significant improvements over baseline methods.

## Citation

If you use this implementation in your research, please cite the original GraphTransGeo++ paper.

## License

This implementation is provided for research purposes only.

## Contact

For any questions or issues, please open an issue on the GitHub repository.
"""
    
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print('Created README.md')

def create_zip_archive(output_dir, zip_file_path):
    """
    Create ZIP archive of the packaged implementation
    
    Args:
        output_dir: Output directory
        zip_file_path: Path to the ZIP file
    """
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(output_dir))
                zipf.write(file_path, arcname)
    
    print(f'Created ZIP archive: {zip_file_path}')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Package GCN implementation for distribution')
    parser.add_argument('--source_dir', type=str, default='./', help='Source directory')
    parser.add_argument('--output_dir', type=str, default='./gcn_implementation', help='Output directory')
    parser.add_argument('--create_zip', action='store_true', help='Create ZIP archive')
    args = parser.parse_args()
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directory structure
    create_directory_structure(output_dir)
    
    # Copy implementation files
    copy_implementation_files(args.source_dir, output_dir)
    
    # Create README
    create_readme(output_dir)
    
    # Create ZIP archive if requested
    if args.create_zip:
        zip_file_path = f'graphtransgeo_gcn_optimized_{timestamp}.zip'
        create_zip_archive(output_dir, zip_file_path)
    
    print(f'Packaging complete. Output directory: {output_dir}')

if __name__ == '__main__':
    main()
