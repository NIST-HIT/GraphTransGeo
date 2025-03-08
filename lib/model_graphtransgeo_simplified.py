import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        if isinstance(adj, torch.sparse.FloatTensor):
            output = torch.sparse.mm(adj, support)
        else:
            output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphTransGeoSimplified(nn.Module):
    """
    Simplified GraphTransGeo++ model for IP geolocation
    Adapted to work with our data loader's output format
    """
    def __init__(self, dim_in, dim_out=2, hidden=128, epsilon=0.01, alpha=0.01, beta=0.5):
        super(GraphTransGeoSimplified, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden = hidden
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        
        # Graph convolutional layers
        self.gc1 = GraphConvolution(dim_in, hidden)
        self.gc2 = GraphConvolution(hidden, hidden)
        
        # Output layers for NIG parameters
        self.out_gamma = nn.Linear(hidden, dim_out)
        self.out_v = nn.Linear(hidden, 1)
        self.out_alpha = nn.Linear(hidden, 1)
        self.out_beta = nn.Linear(hidden, 1)
        
        # Adversarial layers
        self.adv_layer = nn.Linear(hidden, hidden)
        
        # Fusion layer
        self.fusion_layer = nn.Linear(hidden*2, hidden)
        
    def evidence(self, x):
        """Convert network output to evidence"""
        return F.softplus(x)
    
    def generate_perturbation(self, x, grad, epsilon=None):
        """Generate adversarial perturbation"""
        if epsilon is None:
            epsilon = self.epsilon
            
        # Calculate perturbation direction
        if grad is not None and not torch.isnan(grad).any() and not torch.isinf(grad).any():
            # Gradient clipping to prevent explosion
            grad = torch.clamp(grad, min=-10.0, max=10.0)
            
            # Handle different gradient dimensions
            if grad.dim() <= 1:
                grad_norm = torch.norm(grad, p=2, keepdim=True)
            else:
                grad_norm = torch.norm(grad, p=2, dim=1, keepdim=True)
                
            # Prevent division by zero
            grad_norm = torch.clamp(grad_norm, min=1e-8)
            
            # Calculate perturbation
            perturbation = epsilon * grad / grad_norm
            
            # Clip perturbation for numerical stability
            perturbation = torch.clamp(perturbation, min=-epsilon, max=epsilon)
            
            # Check for NaN or Inf values
            if torch.isnan(perturbation).any() or torch.isinf(perturbation).any():
                perturbation = torch.zeros_like(x)
        else:
            perturbation = torch.zeros_like(x)
            
        return perturbation
    
    def forward(self, features, adj, batch_indices=None, training=True):
        """
        Forward pass with standard and adversarial views
        
        Args:
            features: Node features [num_nodes, dim_in]
            adj: Adjacency matrix [num_nodes, num_nodes]
            batch_indices: Indices of nodes to process (optional)
            training: Whether in training mode
            
        Returns:
            If training=True:
                Returns (gamma, v, alpha, beta) for both standard and adversarial views
            If training=False:
                Returns (gamma, v, alpha, beta) for standard view only
        """
        # Standard view forward pass
        x = F.relu(self.gc1(features, adj))
        x = F.dropout(x, 0.5, training=training)
        x = F.relu(self.gc2(x, adj))
        
        # If batch_indices is provided, select only those nodes
        if batch_indices is not None:
            x_batch = x[batch_indices]
        else:
            x_batch = x
        
        # Output NIG parameters
        gamma = self.out_gamma(x_batch)
        v = self.evidence(self.out_v(x_batch))
        alpha = self.evidence(self.out_alpha(x_batch)) + 1.0  # alpha > 1
        beta = self.evidence(self.out_beta(x_batch))
        
        # Return standard view if not training
        if not training:
            return gamma, v, alpha, beta
        
        # Generate adversarial perturbation
        if batch_indices is not None:
            x_tensor = features[batch_indices].detach().clone().requires_grad_(True)
        else:
            x_tensor = features.detach().clone().requires_grad_(True)
        
        # Forward pass for gradient calculation
        x_temp = F.relu(self.gc1(x_tensor, adj))
        x_temp = F.dropout(x_temp, 0.5, training=False)
        x_temp = F.relu(self.gc2(x_temp, adj))
        
        # Calculate loss for gradient
        gamma_temp = self.out_gamma(x_temp)
        v_temp = self.evidence(self.out_v(x_temp))
        alpha_temp = self.evidence(self.out_alpha(x_temp)) + 1.0
        beta_temp = self.evidence(self.out_beta(x_temp))
        
        # Use mean of outputs as loss target
        loss = torch.mean(gamma_temp) + torch.mean(v_temp) + torch.mean(alpha_temp) + torch.mean(beta_temp)
        
        # Calculate gradients
        try:
            grads = torch.autograd.grad(loss, x_tensor, retain_graph=False, create_graph=False)
            grad = grads[0]
        except:
            grad = torch.zeros_like(x_tensor)
        
        # Generate perturbation
        perturbation = self.generate_perturbation(x_tensor, grad)
        
        # Adversarial view forward pass
        x_adv = x_tensor + perturbation
        x_adv = F.relu(self.gc1(x_adv, adj))
        x_adv = F.dropout(x_adv, 0.5, training=training)
        x_adv = F.relu(self.gc2(x_adv, adj))
        
        # Output NIG parameters for adversarial view
        gamma_adv = self.out_gamma(x_adv)
        v_adv = self.evidence(self.out_v(x_adv))
        alpha_adv = self.evidence(self.out_alpha(x_adv)) + 1.0
        beta_adv = self.evidence(self.out_beta(x_adv))
        
        # Fusion of standard and adversarial views
        try:
            # Concatenate features from both views
            fused_features = torch.cat([x_batch, x_adv], dim=1)
            fused_features = self.fusion_layer(fused_features)
            
            # Output final NIG parameters
            gamma_final = self.out_gamma(fused_features)
            v_final = self.evidence(self.out_v(fused_features))
            alpha_final = self.evidence(self.out_alpha(fused_features)) + 1.0
            beta_final = self.evidence(self.out_beta(fused_features))
        except:
            # If fusion fails, use standard view
            gamma_final, v_final, alpha_final, beta_final = gamma, v, alpha, beta
        
        # Return all views
        return (gamma, v, alpha, beta), (gamma_adv, v_adv, alpha_adv, beta_adv), (gamma_final, v_final, alpha_final, beta_final)
    
    def compute_loss(self, pred, target, training=True):
        """
        Compute NIG loss for regression with uncertainty
        
        Args:
            pred: Model predictions (gamma, v, alpha, beta)
            target: Ground truth coordinates [batch_size, 2]
            training: Whether in training mode
            
        Returns:
            loss: Total loss
            mse: Mean squared error
        """
        if training:
            # Unpack predictions from all views
            (gamma, v, alpha, beta), (gamma_adv, v_adv, alpha_adv, beta_adv), (gamma_final, v_final, alpha_final, beta_final) = pred
            
            # Calculate NIG loss for standard view
            loss_nig, mse = self.nig_loss(gamma, v, alpha, beta, target)
            
            # Calculate NIG loss for adversarial view
            loss_nig_adv, mse_adv = self.nig_loss(gamma_adv, v_adv, alpha_adv, beta_adv, target)
            
            # Calculate NIG loss for fused view
            loss_nig_final, mse_final = self.nig_loss(gamma_final, v_final, alpha_final, beta_final, target)
            
            # Calculate KL divergence between standard and adversarial views
            kl_div = self.kl_divergence(gamma, v, alpha, beta, gamma_adv, v_adv, alpha_adv, beta_adv)
            
            # Total loss
            loss = loss_nig + self.beta * loss_nig_adv + loss_nig_final + self.alpha * kl_div
            
            # Return average MSE
            mse = (mse + mse_adv + mse_final) / 3.0
        else:
            # For testing, use only the standard view
            gamma, v, alpha, beta = pred
            loss, mse = self.nig_loss(gamma, v, alpha, beta, target)
        
        return loss, mse
    
    def nig_loss(self, gamma, v, alpha, beta, target):
        """
        Calculate NIG loss for regression with uncertainty
        
        Args:
            gamma: Predicted mean
            v: Predicted variance
            alpha: Predicted alpha parameter
            beta: Predicted beta parameter
            target: Ground truth coordinates
            
        Returns:
            loss: NIG loss
            mse: Mean squared error
        """
        # Ensure proper shapes
        if gamma.shape != target.shape:
            if gamma.shape[0] == target.shape[0]:
                if gamma.shape[1] != target.shape[1]:
                    # Adjust target shape if needed
                    if target.shape[1] > gamma.shape[1]:
                        target = target[:, :gamma.shape[1]]
                    else:
                        # Pad target with zeros
                        padding = torch.zeros(target.shape[0], gamma.shape[1] - target.shape[1], device=target.device)
                        target = torch.cat([target, padding], dim=1)
        
        # Calculate mean squared error
        mse = torch.mean((gamma - target)**2)
        
        # Calculate NIG negative log likelihood
        nll = 0.5 * torch.log(np.pi / v) - alpha * torch.log(2 * beta * (1 + v))
        nll = nll + (alpha + 0.5) * torch.log(v * (gamma - target)**2 + 2 * beta)
        nll = nll + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        
        # Replace NaN or Inf values
        nll = torch.nan_to_num(nll, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Calculate regularization term
        reg = (gamma - target)**2 * (2 * alpha + v)
        reg = torch.nan_to_num(reg, nan=0.0, posinf=10.0, neginf=0.0)
        
        # Total NIG loss
        loss = torch.mean(nll + reg)
        
        return loss, mse
    
    def kl_divergence(self, gamma1, v1, alpha1, beta1, gamma2, v2, alpha2, beta2):
        """
        Calculate KL divergence between two NIG distributions
        
        Args:
            gamma1, v1, alpha1, beta1: Parameters of first NIG distribution
            gamma2, v2, alpha2, beta2: Parameters of second NIG distribution
            
        Returns:
            kl_div: KL divergence
        """
        # Simplified KL divergence calculation
        kl_div = torch.mean((gamma1 - gamma2)**2)
        kl_div = kl_div + torch.mean((v1 - v2)**2)
        kl_div = kl_div + torch.mean((alpha1 - alpha2)**2)
        kl_div = kl_div + torch.mean((beta1 - beta2)**2)
        
        return kl_div
