# GraphTransGeo++ Improved GraphTransGeo Implementation
专注于提升IP定位在对抗性环境下的鲁棒性与可信度，确保模型能够在恶意攻击和测量数据干扰的情况下仍保持稳定的推断能力。

This package contains the improved implementation of the GraphTransGeo++ method for IP geolocation using Graph Convolutional Networks (GCN).


Environment Setup
# Create and Activate a Python Environment
conda create -n graphtransgeo python=3.8
conda activate graphtransgeo

# Install Dependencies
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
## GraphTransGeo Project Structure
# GraphTransGeo Project Structure

## Core Model Files
```
GraphTransGeo/
├── model_graphtransgeo_gcn_optimized.py    # Optimized GCN model implementation
├── improved_gcn_model.py                  # Improved GCN model
├── train_gcn_optimized.py                 # Training script for optimized model
├── train_gcn.py                           # Standard GCN training script
└── gcn_data_loader.py                     # Graph data loader
```

## Data Processing & Analysis
```
├── examine_data_structure.py              # Data structure analysis
├── examine_npz.py                         # NPZ file inspection
├── create_dummy_data.py                   # Generate test data
├── analyze_current_limitations.py         # Analyze current limitations
├── analyze_results.py                     # Result analysis
└── analyze_test_results.py                # Test result analysis
```

## Visualization Tools
```
├── visualize_comprehensive_results.py     # Comprehensive result visualization
├── visualize_cross_dataset_comparison.py  # Cross-dataset comparison visualization
├── visualize_gcn_performance.py           # GCN performance visualization
├── visualize_gcn_results.py               # GCN result visualization
├── visualize_metrics.py                   # Metrics visualization
├── visualize_metrics_english.py           # English version of metrics visualization
├── visualize_model_comparison.py          # Model comparison visualization
├── visualize_model_performance.py         # Model performance visualization
├── visualize_performance.py               # General performance visualization
└── visualize_training_curves.py           # Training curve visualization
```

## Test Scripts
```
├── test_adapter_with_data.py              # Feature adapter testing
├── test_advanced_strategies.py            # Advanced strategy testing
├── test_enhanced_graph_builder.py         # Enhanced graph builder testing
├── test_feature_adapter.py                # Feature adapter testing
└── test_improved_architecture.py          # Improved architecture testing
```

## Execution Scripts
```
├── run_all_datasets.sh                    # Run all datasets
├── run_gcn_model.sh                       # Run the GCN model
├── run_gcn_optimized.sh                   # Run optimized GCN
├── run_improved_gcn.sh                    # Run improved GCN
├── run_optimized_model.sh                 # Run optimized model
├── run_test_model.sh                      # Run test model
└── run_test_on_datasets.sh                # Run tests on datasets
```

## Packaging & Reports
```
├── package_gcn_implementation.py          # Package GCN implementation
├── prepare_final_report.py                # Prepare final report
└── prepare_results_package.py             # Prepare results package
```

## Directory Structure
```
├── lib/                                   # Core library files
│   ├── model/                             # Model core components
│   ├── utils.py                           # Utility functions
│   ├── layers.py                          # Layer implementations
│   ├── sublayers.py                       # Sublayer implementations
│   ├── sag.py                             # Spatial attention graph implementation
│   ├── training/                          # Advanced training strategies
│   ├── adapter/                           # Feature adaptation module
│   ├── adapters/                          # Various adapter implementations
│   └── graph_construction/                # Graph construction tools
```

## GCN Implementation
```
├── gcn_implementation/                    # GCN-specific implementations
│   ├── train_gcn_optimized.py             # Optimized GCN training
│   ├── gcn_data_loader.py                 # GCN data loading
│   ├── verify_normalization.py            # Verify normalization effectiveness
│   ├── verify_all_datasets.py             # Verify all datasets
│   ├── visualize_gcn_results.py           # Visualize GCN results
│   └── run_gcn_optimized.sh               # Run optimized GCN script
```

## Resources & Reports
```
├── asset/                                 # Resource files
│   ├── data/                              # Dataset files
│   │   ├── New_York/                      # New York dataset
│   │   ├── Los_Angeles/                   # Los Angeles dataset
│   │   └── Shanghai/                      # Shanghai dataset
│   ├── model/                             # Model checkpoints
│   ├── figures/                           # Visualization outputs
│   └── log/                               # Training and test logs
```

## Reports
```
├── reports/                               # Report files
│   ├── final_gcn_report.md                # Final GCN report
│   ├── performance_analysis.md            # Performance analysis
│   ├── model_comparison_report.md         # Model comparison report
│   ├── current_limitations.md             # Current limitations analysis
│   └── figures/                           # Performance visualization charts
```

## Additional Packages & Plans
```
├── improved_package/                      # Improved package
├── package_output/                        # Packaged output
└── optimization_plan/                     # Optimization plan
```

## Documentation
```
└── README.md                              # Project documentation (English)
```



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

