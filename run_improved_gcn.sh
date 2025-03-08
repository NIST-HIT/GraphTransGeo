#!/bin/bash

# Create log and model directories
mkdir -p asset/log
mkdir -p asset/model

# Install required packages
pip install torch-geometric

# Train improved GCN model on New York dataset
echo "Training improved GCN model on New York dataset..."
python improved_gcn_model.py --dataset New_York --epochs 100 --batch_size 32 --hidden 256 --num_layers 3 --epsilon 0.01 --alpha 0.01 --beta 0.5 --lr 0.001 --weight_decay 1e-5 --ensemble_size 3

# Test on other datasets
echo "Testing on Shanghai dataset..."
python improved_gcn_model.py --dataset Shanghai --test_only --load_epoch 100 --hidden 256 --num_layers 3 --epsilon 0.01 --alpha 0.01 --beta 0.5 --ensemble_size 3

echo "Testing on Los Angeles dataset..."
python improved_gcn_model.py --dataset Los_Angeles --test_only --load_epoch 100 --hidden 256 --num_layers 3 --epsilon 0.01 --alpha 0.01 --beta 0.5 --ensemble_size 3

echo "All tests completed. Results saved to asset/log/"
