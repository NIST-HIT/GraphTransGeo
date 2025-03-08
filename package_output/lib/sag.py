import torch
import torch.nn as nn
import torch.nn.functional as F

class SAG:
    """
    Self-Adaptive Adversarial Training (SAG)
    动态调整对抗扰动的强度
    """
    def __init__(self, model, epsilon_init=0.01, epsilon_min=0.001, epsilon_max=0.1, decay=0.9):
        self.model = model
        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.decay = decay
        self.prev_loss = float('inf')
    
    def generate_perturbation(self, lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay, outputs, loss_fn):
        """
        生成对抗扰动
        """
        # 创建需要计算梯度的张量
        lm_X_adv = lm_X.detach().clone().requires_grad_(True)
        lm_Y_adv = lm_Y.detach().clone().requires_grad_(True)
        tg_X_adv = tg_X.detach().clone().requires_grad_(True)
        tg_Y_adv = tg_Y.detach().clone().requires_grad_(True)
        lm_delay_adv = lm_delay.detach().clone().requires_grad_(True)
        tg_delay_adv = tg_delay.detach().clone().requires_grad_(True)
        
        # 前向传播
        adv_outputs = self.model(lm_X_adv, lm_Y_adv, tg_X_adv, tg_Y_adv, lm_delay_adv, tg_delay_adv)
        
        # 计算损失
        loss = loss_fn(adv_outputs, outputs)
        
        # 计算梯度
        grads = torch.autograd.grad(loss, [lm_X_adv, lm_Y_adv, tg_X_adv, tg_Y_adv, lm_delay_adv, tg_delay_adv], 
                                   retain_graph=True)
        
        # 生成对抗扰动
        lm_X_pert = self._normalize_perturbation(grads[0])
        lm_Y_pert = self._normalize_perturbation(grads[1])
        tg_X_pert = self._normalize_perturbation(grads[2])
        tg_Y_pert = self._normalize_perturbation(grads[3])
        lm_delay_pert = self._normalize_perturbation(grads[4])
        tg_delay_pert = self._normalize_perturbation(grads[5])
        
        return lm_X_pert, lm_Y_pert, tg_X_pert, tg_Y_pert, lm_delay_pert, tg_delay_pert
    
    def _normalize_perturbation(self, grad):
        """
        归一化扰动
        """
        if grad is None:
            return 0
        norm = torch.norm(grad, p=2)
        if norm > 0:
            return self.epsilon * grad / (norm + 1e-12)
        return torch.zeros_like(grad)
    
    def update_epsilon(self, current_loss):
        """
        根据当前损失动态调整epsilon
        """
        if current_loss < self.prev_loss:
            # 如果损失减小，增加扰动强度
            self.epsilon = min(self.epsilon / self.decay, self.epsilon_max)
        else:
            # 如果损失增加，减小扰动强度
            self.epsilon = max(self.epsilon * self.decay, self.epsilon_min)
        
        self.prev_loss = current_loss
        return self.epsilon
