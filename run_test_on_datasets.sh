#!/bin/bash

# Script to test the optimized GCN model on different datasets

# Create log directory if it doesn't exist
mkdir -p asset/log
mkdir -p asset/figures

# Test on New York dataset
echo "Testing on New York dataset..."
python train_gcn_optimized.py \
    --dataset New_York \
    --test_only \
    --load_model New_York_gcn_optimized_best.pth \
    --hidden 256 \
    --num_layers 3 \
    --dropout 0.3 \
    --epsilon 0.01 \
    --alpha 0.01 \
    --beta 0.5 \
    --hav_weight 0.3 \
    --ensemble_size 3

# Test on Shanghai dataset
echo "Testing on Shanghai dataset..."
python train_gcn_optimized.py \
    --dataset Shanghai \
    --test_only \
    --load_model New_York_gcn_optimized_best.pth \
    --hidden 256 \
    --num_layers 3 \
    --dropout 0.3 \
    --epsilon 0.01 \
    --alpha 0.01 \
    --beta 0.5 \
    --hav_weight 0.3 \
    --ensemble_size 3

# Test on Los Angeles dataset
echo "Testing on Los Angeles dataset..."
python train_gcn_optimized.py \
    --dataset Los_Angeles \
    --test_only \
    --load_model New_York_gcn_optimized_best.pth \
    --hidden 256 \
    --num_layers 3 \
    --dropout 0.3 \
    --epsilon 0.01 \
    --alpha 0.01 \
    --beta 0.5 \
    --hav_weight 0.3 \
    --ensemble_size 3

# Generate visualizations
echo "Generating visualizations..."
python visualize_model_performance.py

# Analyze test results
echo "Analyzing test results..."
python analyze_test_results.py

echo "All tests completed. Results saved to asset/log/ and asset/figures/"
