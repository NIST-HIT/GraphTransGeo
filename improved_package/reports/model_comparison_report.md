# GraphTransGeo vs MLP Model Comparison Report

## Overview
This report compares the performance of the Graph Convolutional Network (GCN) model with the Multi-Layer Perceptron (MLP) model for IP geolocation using the GraphTransGeo++ approach.

## Model Architectures
- **GCN Model**: Graph Convolutional Network with adversarial training
- **MLP Model**: Multi-Layer Perceptron with adversarial training

## Training Parameters
- Learning Rate: 0.001
- Epsilon: 0.01
- Alpha: 0.01
- Beta: 0.5
- Hidden Dimension: 256
- Number of Layers (GCN): 2

## Performance Metrics

| Metric | GCN | MLP | Improvement |
|--------|-----|-----|-------------|
| MSE | 3272.84 | 3.51 | -93206.12% |
| MAE | 74.90 | 1.49 | -4924.88% |
| Median Distance Error | 5274.75 km | 224.83 km | -2246.08% |

## Analysis

### MSE Comparison
The GCN model achieved an MSE of 3272.84, compared to the MLP model's MSE of 3.51. This represents a decrease of 93206.12% in MSE.

### MAE Comparison
The GCN model achieved an MAE of 74.90, compared to the MLP model's MAE of 1.49. This represents a decrease of 4924.88% in MAE.

### Median Distance Error Comparison
The GCN model achieved a median distance error of 5274.75 km, compared to the MLP model's median distance error of 224.83 km. This represents a decrease of 2246.08% in median distance error.

## Convergence Analysis
The GCN model converged slower than the MLP model. The GCN model trained for 50 epochs, while the MLP model trained for 50 epochs.

## Conclusion
The results show mixed performance between the GCN and MLP models.

Further tuning and analysis may be needed to improve the GCN model performance.

## Generated on
2025-03-07 16:14:14
