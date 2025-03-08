from math import gamma
from re import L
from .layers import *
import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
from .model_spatial import TrustGeoSpatial

class GraphTransGeoImproved(TrustGeoSpatial):
    """
    GraphTransGeo++: 一个集成对抗训练的可靠地理定位方法
    扩展了TrustGeoSpatial模型，增加了对抗训练和不确定性推断
    改进版本：在测试时也使用融合层
    """
    def __init__(self, dim_in, epsilon=0.01, alpha=0.01, beta=0.5):
        super(GraphTransGeoImproved, self).__init__(dim_in)
        self.eps_value = epsilon  # 对抗扰动大小
        self.alpha_value = alpha  # 先验正则化系数
        self.beta_value = beta    # 对抗鲁棒性损失系数
        
        # 对抗训练相关层
        self.adv_layer_graph = nn.Linear(self.dim_z*2, self.dim_z*2)
        self.adv_layer_attri = nn.Linear(self.dim_in, self.dim_in)
        
        # 融合层 - 分别为图视角和属性视角创建融合层
        # 图视角: gamma1_g(1) + gamma2_g(1) + v_g(1) + alpha_g(1) + beta_g(1) + 
        #        gamma1_g_adv(1) + gamma2_g_adv(1) + v_g_adv(1) + alpha_g_adv(1) + beta_g_adv(1) = 10
        self.fusion_layer_graph = nn.Linear(10, 5)  # 10 -> 5 (2+1+1+1)
        
        # 属性视角: gamma1_a(1) + gamma2_a(1) + v_a(1) + alpha_a(1) + beta_a(1) + 
        #          gamma1_a_adv(1) + gamma2_a_adv(1) + v_a_adv(1) + alpha_a_adv(1) + beta_a_adv(1) = 10
        self.fusion_layer_attri = nn.Linear(10, 5)  # 10 -> 5 (2+1+1+1)
        
    def generate_perturbation(self, x, grad, epsilon=None):
        """
        生成对抗扰动，增加数值稳定性
        """
        if epsilon is None:
            epsilon = self.eps_value
            
        # 计算扰动方向
        if grad is not None and not torch.isnan(grad).any() and not torch.isinf(grad).any():
            # 梯度裁剪，防止梯度爆炸
            grad = torch.clamp(grad, min=-10.0, max=10.0)
            
            # 处理不同维度的梯度
            if grad.dim() <= 1:  # 如果梯度是标量或一维向量
                grad_norm = torch.norm(grad, p=2, keepdim=True)
            else:  # 如果梯度是多维张量
                grad_norm = torch.norm(grad, p=2, dim=1, keepdim=True)
                
            # 防止除零
            grad_norm = torch.clamp(grad_norm, min=1e-8)
            
            # 计算扰动
            perturbation = epsilon * grad / grad_norm
            
            # 裁剪扰动，确保数值稳定
            perturbation = torch.clamp(perturbation, min=-epsilon, max=epsilon)
            
            # 检查扰动是否包含NaN或Inf
            if torch.isnan(perturbation).any() or torch.isinf(perturbation).any():
                perturbation = torch.zeros_like(x)
        else:
            perturbation = torch.zeros_like(x)
            
        return perturbation
    
    def forward(self, lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay, add_noise=0, training=True):
        """
        前向传播，包含标准视角和对抗视角
        
        Args:
            lm_X: 地标特征 [..., 30/51]
            lm_Y: 地标位置 [..., 2]
            tg_X: 目标特征 [..., 30/51]
            tg_Y: 目标位置 [..., 2]
            lm_delay: 地标到公共路由器的延迟 [..., 1]
            tg_delay: 目标到公共路由器的延迟 [..., 1]
            add_noise: 是否添加噪声
            training: 是否处于训练模式
            
        Returns:
            如果training=True:
                返回(clean_outputs, adv_outputs, final_outputs)
            如果training=False:
                返回融合后的最终输出
        """
        # 标准视角前向传播
        clean_outputs = super(GraphTransGeoImproved, self).forward(lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay, add_noise)
        
        # 解包标准视角输出
        two_gamma_g, v_g, alpha_g, beta_g, two_gamma_a, v_a, alpha_a, beta_a = clean_outputs
        
        # 检查并替换NaN或Inf值
        two_gamma_g = torch.nan_to_num(two_gamma_g, nan=0.0, posinf=1.0, neginf=-1.0)
        v_g = torch.nan_to_num(v_g, nan=0.0, posinf=1.0, neginf=0.0)
        alpha_g = torch.nan_to_num(alpha_g, nan=1.0, posinf=10.0, neginf=1.0)
        beta_g = torch.nan_to_num(beta_g, nan=0.0, posinf=10.0, neginf=0.0)
        
        two_gamma_a = torch.nan_to_num(two_gamma_a, nan=0.0, posinf=1.0, neginf=-1.0)
        v_a = torch.nan_to_num(v_a, nan=0.0, posinf=1.0, neginf=0.0)
        alpha_a = torch.nan_to_num(alpha_a, nan=1.0, posinf=10.0, neginf=1.0)
        beta_a = torch.nan_to_num(beta_a, nan=0.0, posinf=10.0, neginf=0.0)
        
        if not training:
            # 在测试时，仍然使用标准视角的输出，但应用融合层
            try:
                # 使用相同的特征代替对抗特征
                graph_features = torch.cat([two_gamma_g, v_g, alpha_g, beta_g, 
                                           two_gamma_g, v_g, alpha_g, beta_g], dim=1)
                graph_features_fused = self.fusion_layer_graph(graph_features)
                two_gamma_g_final, v_g_final, alpha_g_final, beta_g_final = torch.split(
                    graph_features_fused, [2, 1, 1, 1], dim=1
                )
                
                attri_features = torch.cat([two_gamma_a, v_a, alpha_a, beta_a,
                                           two_gamma_a, v_a, alpha_a, beta_a], dim=1)
                attri_features_fused = self.fusion_layer_attri(attri_features)
                two_gamma_a_final, v_a_final, alpha_a_final, beta_a_final = torch.split(
                    attri_features_fused, [2, 1, 1, 1], dim=1
                )
                
                # 确保输出不包含NaN或Inf
                two_gamma_g_final = torch.nan_to_num(two_gamma_g_final, nan=0.0, posinf=1.0, neginf=-1.0)
                v_g_final = torch.nan_to_num(v_g_final, nan=0.0, posinf=1.0, neginf=0.0)
                alpha_g_final = torch.nan_to_num(alpha_g_final, nan=1.0, posinf=10.0, neginf=1.0)
                beta_g_final = torch.nan_to_num(beta_g_final, nan=0.0, posinf=10.0, neginf=0.0)
                
                two_gamma_a_final = torch.nan_to_num(two_gamma_a_final, nan=0.0, posinf=1.0, neginf=-1.0)
                v_a_final = torch.nan_to_num(v_a_final, nan=0.0, posinf=1.0, neginf=0.0)
                alpha_a_final = torch.nan_to_num(alpha_a_final, nan=1.0, posinf=10.0, neginf=1.0)
                beta_a_final = torch.nan_to_num(beta_a_final, nan=0.0, posinf=10.0, neginf=0.0)
                
                # 返回融合后的输出
                return (
                    two_gamma_g_final, v_g_final, alpha_g_final, beta_g_final,
                    two_gamma_a_final, v_a_final, alpha_a_final, beta_a_final
                )
            except RuntimeError:
                # 如果融合失败，直接使用标准视角的输出
                return clean_outputs
        
        # 为输入生成对抗扰动
        # 保存输入的梯度信息
        lm_X_tensor = lm_X.detach().clone().requires_grad_(True)
        lm_Y_tensor = lm_Y.detach().clone().requires_grad_(True)
        tg_X_tensor = tg_X.detach().clone().requires_grad_(True)
        tg_Y_tensor = tg_Y.detach().clone().requires_grad_(True)
        lm_delay_tensor = lm_delay.detach().clone().requires_grad_(True)
        tg_delay_tensor = tg_delay.detach().clone().requires_grad_(True)
        
        # 前向传播计算损失
        temp_outputs = super(GraphTransGeoImproved, self).forward(
            lm_X_tensor, lm_Y_tensor, tg_X_tensor, tg_Y_tensor, 
            lm_delay_tensor, tg_delay_tensor, add_noise
        )
        
        # 计算损失
        temp_two_gamma_g, temp_v_g, temp_alpha_g, temp_beta_g, temp_two_gamma_a, temp_v_a, temp_alpha_a, temp_beta_a = temp_outputs
        
        # 使用输出的均值作为损失目标，增加数值稳定性
        # 检查并替换NaN或Inf值
        temp_two_gamma_g = torch.nan_to_num(temp_two_gamma_g, nan=0.0, posinf=1.0, neginf=-1.0)
        temp_v_g = torch.nan_to_num(temp_v_g, nan=0.0, posinf=1.0, neginf=0.0)
        temp_alpha_g = torch.nan_to_num(temp_alpha_g, nan=1.0, posinf=10.0, neginf=1.0)
        temp_beta_g = torch.nan_to_num(temp_beta_g, nan=0.0, posinf=10.0, neginf=0.0)
        
        temp_two_gamma_a = torch.nan_to_num(temp_two_gamma_a, nan=0.0, posinf=1.0, neginf=-1.0)
        temp_v_a = torch.nan_to_num(temp_v_a, nan=0.0, posinf=1.0, neginf=0.0)
        temp_alpha_a = torch.nan_to_num(temp_alpha_a, nan=1.0, posinf=10.0, neginf=1.0)
        temp_beta_a = torch.nan_to_num(temp_beta_a, nan=0.0, posinf=10.0, neginf=0.0)
        
        # 计算损失，使用clamp确保数值稳定
        loss = torch.mean(torch.clamp(temp_two_gamma_g, min=-10.0, max=10.0)) + \
               torch.mean(torch.clamp(temp_v_g, min=0.0, max=10.0)) + \
               torch.mean(torch.clamp(temp_alpha_g, min=1.0, max=10.0)) + \
               torch.mean(torch.clamp(temp_beta_g, min=0.0, max=10.0)) + \
               torch.mean(torch.clamp(temp_two_gamma_a, min=-10.0, max=10.0)) + \
               torch.mean(torch.clamp(temp_v_a, min=0.0, max=10.0)) + \
               torch.mean(torch.clamp(temp_alpha_a, min=1.0, max=10.0)) + \
               torch.mean(torch.clamp(temp_beta_a, min=0.0, max=10.0))
        
        # 确保损失不是NaN或Inf
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.1, requires_grad=True)
        
        # 计算梯度，增加数值稳定性
        try:
            grads = torch.autograd.grad(loss, [lm_X_tensor, lm_Y_tensor, tg_X_tensor, tg_Y_tensor, lm_delay_tensor, tg_delay_tensor], 
                                       retain_graph=False, create_graph=False, allow_unused=True)
        except RuntimeError:
            # 如果梯度计算失败，返回零梯度
            grads = [torch.zeros_like(lm_X_tensor), torch.zeros_like(lm_Y_tensor), 
                     torch.zeros_like(tg_X_tensor), torch.zeros_like(tg_Y_tensor),
                     torch.zeros_like(lm_delay_tensor), torch.zeros_like(tg_delay_tensor)]
        
        # 生成对抗扰动
        lm_X_pert = self.generate_perturbation(lm_X, grads[0])
        lm_Y_pert = self.generate_perturbation(lm_Y, grads[1])
        tg_X_pert = self.generate_perturbation(tg_X, grads[2])
        tg_Y_pert = self.generate_perturbation(tg_Y, grads[3])
        lm_delay_pert = self.generate_perturbation(lm_delay, grads[4])
        tg_delay_pert = self.generate_perturbation(tg_delay, grads[5])
        
        # 添加扰动
        lm_X_adv = lm_X + lm_X_pert
        lm_Y_adv = lm_Y + lm_Y_pert
        tg_X_adv = tg_X + tg_X_pert
        tg_Y_adv = tg_Y + tg_Y_pert
        lm_delay_adv = lm_delay + lm_delay_pert
        tg_delay_adv = tg_delay + tg_delay_pert
        
        # 对抗视角前向传播
        adv_outputs = super(GraphTransGeoImproved, self).forward(
            lm_X_adv, lm_Y_adv, tg_X_adv, tg_Y_adv, 
            lm_delay_adv, tg_delay_adv, add_noise
        )
        
        # 解包对抗视角输出
        two_gamma_g_adv, v_g_adv, alpha_g_adv, beta_g_adv, two_gamma_a_adv, v_a_adv, alpha_a_adv, beta_a_adv = adv_outputs
        
        # 融合标准视角和对抗视角，增加数值稳定性
        # 检查并替换NaN或Inf值
        two_gamma_g_adv = torch.nan_to_num(two_gamma_g_adv, nan=0.0, posinf=1.0, neginf=-1.0)
        v_g_adv = torch.nan_to_num(v_g_adv, nan=0.0, posinf=1.0, neginf=0.0)
        alpha_g_adv = torch.nan_to_num(alpha_g_adv, nan=1.0, posinf=10.0, neginf=1.0)
        beta_g_adv = torch.nan_to_num(beta_g_adv, nan=0.0, posinf=10.0, neginf=0.0)
        
        two_gamma_a_adv = torch.nan_to_num(two_gamma_a_adv, nan=0.0, posinf=1.0, neginf=-1.0)
        v_a_adv = torch.nan_to_num(v_a_adv, nan=0.0, posinf=1.0, neginf=0.0)
        alpha_a_adv = torch.nan_to_num(alpha_a_adv, nan=1.0, posinf=10.0, neginf=1.0)
        beta_a_adv = torch.nan_to_num(beta_a_adv, nan=0.0, posinf=10.0, neginf=0.0)
        
        # 图视角融合
        try:
            graph_features = torch.cat([two_gamma_g, v_g, alpha_g, beta_g, 
                                       two_gamma_g_adv, v_g_adv, alpha_g_adv, beta_g_adv], dim=1)
            graph_features_fused = self.fusion_layer_graph(graph_features)
            two_gamma_g_final, v_g_final, alpha_g_final, beta_g_final = torch.split(
                graph_features_fused, [2, 1, 1, 1], dim=1
            )
        except RuntimeError:
            # 如果融合失败，直接使用标准视角的输出
            two_gamma_g_final, v_g_final, alpha_g_final, beta_g_final = two_gamma_g, v_g, alpha_g, beta_g
        
        # 属性视角融合
        try:
            attri_features = torch.cat([two_gamma_a, v_a, alpha_a, beta_a,
                                       two_gamma_a_adv, v_a_adv, alpha_a_adv, beta_a_adv], dim=1)
            attri_features_fused = self.fusion_layer_attri(attri_features)
            two_gamma_a_final, v_a_final, alpha_a_final, beta_a_final = torch.split(
                attri_features_fused, [2, 1, 1, 1], dim=1
            )
        except RuntimeError:
            # 如果融合失败，直接使用标准视角的输出
            two_gamma_a_final, v_a_final, alpha_a_final, beta_a_final = two_gamma_a, v_a, alpha_a, beta_a
            
        # 确保输出不包含NaN或Inf
        two_gamma_g_final = torch.nan_to_num(two_gamma_g_final, nan=0.0, posinf=1.0, neginf=-1.0)
        v_g_final = torch.nan_to_num(v_g_final, nan=0.0, posinf=1.0, neginf=0.0)
        alpha_g_final = torch.nan_to_num(alpha_g_final, nan=1.0, posinf=10.0, neginf=1.0)
        beta_g_final = torch.nan_to_num(beta_g_final, nan=0.0, posinf=10.0, neginf=0.0)
        
        two_gamma_a_final = torch.nan_to_num(two_gamma_a_final, nan=0.0, posinf=1.0, neginf=-1.0)
        v_a_final = torch.nan_to_num(v_a_final, nan=0.0, posinf=1.0, neginf=0.0)
        alpha_a_final = torch.nan_to_num(alpha_a_final, nan=1.0, posinf=10.0, neginf=1.0)
        beta_a_final = torch.nan_to_num(beta_a_final, nan=0.0, posinf=10.0, neginf=0.0)
        
        # 最终输出
        final_outputs = (
            two_gamma_g_final, v_g_final, alpha_g_final, beta_g_final,
            two_gamma_a_final, v_a_final, alpha_a_final, beta_a_final
        )
        
        return clean_outputs, adv_outputs, final_outputs
