#!/bin/bash

# ================================
# 图神经网络城市空间建模训练与测试脚本
# 这个脚本用于训练优化的图卷积网络(GCN)模型，并在不同城市数据集上进行训练
# ================================

# 创建必要的目录结构
# log目录: 存储训练和测试过程中生成的日志文件
# model目录: 存储训练好的模型参数
mkdir -p asset/log
mkdir -p asset/model

# 安装必要的Python依赖库（如果尚未安装）
# torch-geometric: PyTorch几何库，用于图神经网络
# tqdm: 进度条显示工具
# matplotlib: 绘图库，用于可视化结果
pip install torch-geometric tqdm matplotlib

# ================================
# 模型训练部分 - 所有数据集都参与训练
# ================================

# 使用纽约数据集训练模型
echo "训练优化的GCN模型（纽约数据集）..."
python train_gcn_optimized.py \
    --dataset New_York \
    --epochs 2000 \         # 最大训练轮数，模型会根据早停策略可能提前结束
    --batch_size 32 \       # 每批处理的样本数量
    --hidden 256 \          # 隐藏层神经元数量
    --num_layers 3 \        # 图卷积层的数量
    --dropout 0.3 \         # Dropout比率，用于防止过拟合
    --epsilon 0.01 \        # 正则化参数，控制图结构学习
    --alpha 0.01 \          # 损失函数中的权重系数，平衡不同损失项
    --beta 0.5 \            # 损失函数中的权重系数，平衡不同损失项
    --hav_weight 0.3 \      # Haversine距离损失的权重
    --clip_grad 1.0 \       # 梯度裁剪阈值，防止梯度爆炸
    --min_epochs 30 \       # 最小训练轮数，即使验证指标提前达到早停条件也会继续训练
    --ensemble_size 3 \     # 集成学习中模型的数量
    --lr 0.001 \            # 学习率
    --weight_decay 1e-5     # 权重衰减系数，用于L2正则化

# 使用上海数据集训练模型
echo "训练优化的GCN模型（上海数据集）..."
python train_gcn_optimized.py \
    --dataset Shanghai \
    --epochs 2000 \         # 最大训练轮数，模型会根据早停策略可能提前结束
    --batch_size 32 \       # 每批处理的样本数量
    --hidden 256 \          # 隐藏层神经元数量
    --num_layers 3 \        # 图卷积层的数量
    --dropout 0.3 \         # Dropout比率，用于防止过拟合
    --epsilon 0.01 \        # 正则化参数，控制图结构学习
    --alpha 0.01 \          # 损失函数中的权重系数，平衡不同损失项
    --beta 0.5 \            # 损失函数中的权重系数，平衡不同损失项
    --hav_weight 0.3 \      # Haversine距离损失的权重
    --clip_grad 1.0 \       # 梯度裁剪阈值，防止梯度爆炸
    --min_epochs 30 \       # 最小训练轮数，即使验证指标提前达到早停条件也会继续训练
    --ensemble_size 3 \     # 集成学习中模型的数量
    --lr 0.001 \            # 学习率
    --weight_decay 1e-5     # 权重衰减系数，用于L2正则化

# 使用洛杉矶数据集训练模型
echo "训练优化的GCN模型（洛杉矶数据集）..."
python train_gcn_optimized.py \
    --dataset Los_Angeles \
    --epochs 2000 \         # 最大训练轮数，模型会根据早停策略可能提前结束
    --batch_size 32 \       # 每批处理的样本数量
    --hidden 256 \          # 隐藏层神经元数量
    --num_layers 3 \        # 图卷积层的数量
    --dropout 0.3 \         # Dropout比率，用于防止过拟合
    --epsilon 0.01 \        # 正则化参数，控制图结构学习
    --alpha 0.01 \          # 损失函数中的权重系数，平衡不同损失项
    --beta 0.5 \            # 损失函数中的权重系数，平衡不同损失项
    --hav_weight 0.3 \      # Haversine距离损失的权重
    --clip_grad 1.0 \       # 梯度裁剪阈值，防止梯度爆炸
    --min_epochs 30 \       # 最小训练轮数，即使验证指标提前达到早停条件也会继续训练
    --ensemble_size 3 \     # 集成学习中模型的数量
    --lr 0.001 \            # 学习率
    --weight_decay 1e-5     # 权重衰减系数，用于L2正则化

# 训练完成后的提示信息
echo "所有数据集的训练已完成。模型和日志已保存在asset目录下"
echo "模型保存在asset/model/目录，训练日志保存在asset/log/目录"
