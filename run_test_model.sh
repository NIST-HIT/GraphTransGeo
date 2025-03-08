#!/bin/bash

# Create directories
mkdir -p asset/data
mkdir -p asset/log
mkdir -p asset/model
mkdir -p asset/figures

# Generate dummy data
echo "Generating dummy data..."
python create_dummy_data.py

# Install required packages
pip install torch-geometric tqdm matplotlib

# Train on New York dataset with reduced epochs for testing
echo "Training on New York dataset (test run)..."
python train_gcn_optimized.py \
    --dataset New_York \
    --epochs 5 \
    --batch_size 32 \
    --hidden 256 \
    --num_layers 3 \
    --dropout 0.3 \
    --epsilon 0.01 \
    --alpha 0.01 \
    --beta 0.5 \
    --hav_weight 0.3 \
    --clip_grad 1.0 \
    --min_epochs 3 \
    --ensemble_size 3 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --use_feature_adapter

echo "Test run completed!"
