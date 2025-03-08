# GraphTransGeo++ Optimized GraphTransGeo Implementation

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
