# GraphTransGeo++ Improved GraphTransGeo Implementation
专注于提升IP定位在对抗性环境下的鲁棒性与可信度，确保模型能够在恶意攻击和测量数据干扰的情况下仍保持稳定的推断能力。

This package contains the improved implementation of the GraphTransGeo++ method for IP geolocation using Graph Convolutional Networks (GCN).


##Environment Setup
## Create and Activate a Python Environment
```bash
conda create -n graphtransgeo python=3.8
conda activate graphtransgeo


## Install Dependencies
```bash
pip install torch torch-geometric numpy pandas matplotlib tqdm
## Key Improvements

1. **Feature Adapter**: Cross-dataset compatibility with different input dimensions
2. **Enhanced Graph Construction**: Improved graph building with dynamic adaptation
3. **Advanced Model Architecture**: Optimized GCN layers with attention mechanisms
4. **Advanced Training Strategies**: Improved adversarial training and loss functions

## Usage

### Training

```bash
./run_gcn_optimized.sh
```

### Testing

```bash
./run_test_on_datasets.sh
```

### Visualization

```bash
python visualize_model_comparison.py
python visualize_training_curves.py --dataset New_York
```

## Key Components

- `lib/model_graphtransgeo_gcn_optimized.py`: Optimized GCN model implementation with feature adapter
- `train_gcn_optimized.py`: Training script with advanced training strategies
- `gcn_data_loader.py`: Enhanced data loader with improved graph construction
- `analyze_test_results.py`: Script for analyzing test results
- `visualize_model_comparison.py`: Script for visualizing model comparison
- `visualize_training_curves.py`: Script for visualizing training curves
## Performance Comparison

| Dataset     | GCN MSE    | MLP MSE    | GCN MAE    | MLP MAE    |
|-------------|------------|------------|------------|------------|
| New York    | 5.90547       | 3.51       | 1.90       | 1.49       | 
| Shanghai    | 24.4035    | 7859.51    | 3.48975       | 76.32      | 
| Los Angeles | 87.76      | 7569.97    | 2.96476       | 76.15      | 

## Cross-Dataset Generalization

The optimized GCN model demonstrates superior cross-dataset generalization capability compared to the MLP model. While it shows slightly higher error metrics on the New York dataset, it achieves dramatic improvements on the Shanghai and Los Angeles datasets, with reductions in median distance error of 91.11% and 92.73% respectively.

## Feature Adapter

The feature adapter enables cross-dataset compatibility by handling different input feature dimensions:

- **Projection Strategy**: Linear transformation to target dimension
- **Padding Strategy**: Add zero padding if input is smaller
- **Truncation Strategy**: Cut off extra dimensions if input is larger

## Advanced Training Strategies

- **Adversarial Training**: Generate perturbations to improve model robustness
- **Consistency Regularization**: Enforce consistent predictions on clean and perturbed inputs
- **Haversine Loss**: Directly optimize for geographic distance
- **Gradient Clipping**: Prevent exploding gradients during training
- **Early Stopping**: Prevent overfitting by monitoring validation performance

## Requirements

- PyTorch
- PyTorch Geometric
- NumPy
- Matplotlib
- Pandas
- Seaborn

## Reference

@inproceedings{tai2023trustgeo,
  title = {TrustGeo: Uncertainty-Aware Dynamic Graph Learning for Trustworthy IP Geolocation},
  author = {Tai, Wenxin and Chen, Bin and Zhou, Fan and Zhong, Ting and Trajcevski, Goce and Wang, Yong and Chen, Kai},
  booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year = {2023}
}

