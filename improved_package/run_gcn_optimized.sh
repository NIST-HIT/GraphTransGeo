#!/bin/bash

# Create log and model directories
mkdir -p asset/log
mkdir -p asset/model

# Install required packages if needed
pip install torch-geometric tqdm matplotlib

# Train optimized GCN model on New York dataset
echo "Training optimized GCN model on New York dataset..."
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
    --weight_decay 1e-5

# Test on other datasets
echo "Testing on Shanghai dataset..."
python train_gcn_optimized.py \
    --dataset Shanghai \
    --test_only \
    --load_epoch 0 \
    --hidden 256 \
    --num_layers 3 \
    --dropout 0.3 \
    --epsilon 0.01 \
    --alpha 0.01 \
    --beta 0.5 \
    --hav_weight 0.3 \
    --ensemble_size 3

echo "Testing on Los Angeles dataset..."
python train_gcn_optimized.py \
    --dataset Los_Angeles \
    --test_only \
    --load_epoch 0 \
    --hidden 256 \
    --num_layers 3 \
    --dropout 0.3 \
    --epsilon 0.01 \
    --alpha 0.01 \
    --beta 0.5 \
    --hav_weight 0.3 \
    --ensemble_size 3

echo "All tests completed. Results saved to asset/log/"
