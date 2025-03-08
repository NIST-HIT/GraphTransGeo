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

# Train on New York dataset
echo "Training on New York dataset..."
python train_gcn_optimized.py \
    --dataset New_York \
    --epochs 100 \
    --batch_size 32 \
    --hidden 256 \
    --num_layers 3 \
    --dropout 0.3 \
    --epsilon 0.01 \
    --alpha 0.01 \
    --beta 0.5 \
    --hav_weight 0.3 \
    --clip_grad 1.0 \
    --min_epochs 30 \
    --ensemble_size 3 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --use_feature_adapter

# Train on Shanghai dataset
echo "Training on Shanghai dataset..."
python train_gcn_optimized.py \
    --dataset Shanghai \
    --epochs 100 \
    --batch_size 32 \
    --hidden 256 \
    --num_layers 3 \
    --dropout 0.3 \
    --epsilon 0.01 \
    --alpha 0.01 \
    --beta 0.5 \
    --hav_weight 0.3 \
    --clip_grad 1.0 \
    --min_epochs 30 \
    --ensemble_size 3 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --use_feature_adapter

# Train on Los Angeles dataset
echo "Training on Los Angeles dataset..."
python train_gcn_optimized.py \
    --dataset Los_Angeles \
    --epochs 100 \
    --batch_size 32 \
    --hidden 256 \
    --num_layers 3 \
    --dropout 0.3 \
    --epsilon 0.01 \
    --alpha 0.01 \
    --beta 0.5 \
    --hav_weight 0.3 \
    --clip_grad 1.0 \
    --min_epochs 30 \
    --ensemble_size 3 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --use_feature_adapter

echo "All training completed!"
