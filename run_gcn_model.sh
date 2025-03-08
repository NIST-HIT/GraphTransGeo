#!/bin/bash

# Create log and model directories
mkdir -p asset/log
mkdir -p asset/model

# Install required packages
pip install torch-geometric

# Train GCN model on New York dataset
echo "Training GCN model on New York dataset..."
python train_gcn.py --dataset New_York --epochs 100 --batch_size 32 --hidden 256 --epsilon 0.01 --alpha 0.01 --beta 0.5 --lr 0.001 --num_layers 2

# Test on other datasets
echo "Testing on Shanghai dataset..."
python train_gcn.py --dataset Shanghai --test_only --load_epoch 100 --hidden 256 --epsilon 0.01 --alpha 0.01 --beta 0.5 --num_layers 2

echo "Testing on Los Angeles dataset..."
python train_gcn.py --dataset Los_Angeles --test_only --load_epoch 100 --hidden 256 --epsilon 0.01 --alpha 0.01 --beta 0.5 --num_layers 2

echo "All tests completed. Results saved to asset/log/"
