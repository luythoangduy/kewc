# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.CPR import CPR
from collections import defaultdict
import numpy as np

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Kronecker-Factored EWC.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--e_lambda', type=float, required=True,
                        help='lambda weight for K-EWC')
    parser.add_argument('--gamma', type=float, required=True,
                        help='gamma parameter for K-EWC online')
    parser.add_argument('--damping', type=float, default=1e-3,
                        help='damping factor for numerical stability')

    return parser


class KEwcOn(ContinualModel):
    NAME = 'kewc_on'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(KEwcOn, self).__init__(backbone, loss, args, transform)

        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None
        
        # Store Kronecker factors for each layer
        self.A_factors = {}  # Activation covariance matrices
        self.G_factors = {}  # Gradient covariance matrices
        
        # Hook handles for cleanup
        self.forward_handles = []
        self.backward_handles = []
        
        # Temporary storage for collecting statistics
        self.activations = {}
        self.gradients = {}
        
        self.cpr = CPR()
        self.beta = 0.0
        self.CPRon = False
        
        self.damping = args.damping if hasattr(args, 'damping') else 1e-3

    def _get_layer_name(self, module, name):
        """Generate unique layer name for indexing"""
        return f"{name}_{id(module)}"

    def _register_hooks(self):
        """Register forward and backward hooks for collecting activations and gradients"""
        def make_forward_hook(layer_name):
            def forward_hook(module, input, output):
                if isinstance(module, nn.Linear):
                    # For linear layers, store input activations
                    act = input[0].detach()
                    if act.dim() > 2:
                        act = act.view(act.size(0), -1)
                    
                    # Add bias column for bias terms
                    ones = torch.ones(act.size(0), 1, device=act.device)
                    act_with_bias = torch.cat([act, ones], dim=1)
                    
                    if layer_name not in self.activations:
                        self.activations[layer_name] = []
                    self.activations[layer_name].append(act_with_bias)
            return forward_hook

        def make_backward_hook(layer_name):
            def backward_hook(module, grad_input, grad_output):
                if isinstance(module, nn.Linear) and grad_output[0] is not None:
                    # For linear layers, store output gradients
                    grad = grad_output[0].detach()
                    if grad.dim() > 2:
                        grad = grad.view(grad.size(0), -1)
                    
                    if layer_name not in self.gradients:
                        self.gradients[layer_name] = []
                    self.gradients[layer_name].append(grad)
            return backward_hook

        # Register hooks for all Linear layers
        for name, module in self.net.named_modules():
            if isinstance(module, nn.Linear):
                layer_name = self._get_layer_name(module, name)
                
                fh = module.register_forward_hook(make_forward_hook(layer_name))
                bh = module.register_full_backward_hook(make_backward_hook(layer_name))
                
                self.forward_handles.append(fh)
                self.backward_handles.append(bh)

    def _cleanup_hooks(self):
        """Remove all registered hooks"""
        for handle in self.forward_handles + self.backward_handles:
            handle.remove()
        self.forward_handles = []
        self.backward_handles = []

    def _compute_kronecker_factors(self):
        """Compute Kronecker factors A and G for each layer"""
        new_A_factors = {}
        new_G_factors = {}
        
        for layer_name in self.activations:
            if layer_name in self.gradients:
                # Concatenate all activations and gradients for this layer
                all_activations = torch.cat(self.activations[layer_name], dim=0)
                all_gradients = torch.cat(self.gradients[layer_name], dim=0)
                
                # Compute empirical covariance matrices
                # A = E[aa^T] where a is activation
                A = torch.mm(all_activations.t(), all_activations) / all_activations.size(0)
                
                # G = E[gg^T] where g is gradient  
                G = torch.mm(all_gradients.t(), all_gradients) / all_gradients.size(0)
                
                # Add damping for numerical stability
                A.add_(torch.eye(A.size(0), device=A.device) * self.damping)
                G.add_(torch.eye(G.size(0), device=G.device) * self.damping)
                
                new_A_factors[layer_name] = A
                new_G_factors[layer_name] = G
        
        return new_A_factors, new_G_factors

    def _update_kronecker_factors(self, new_A_factors, new_G_factors):
        """Update Kronecker factors with exponential moving average"""
        if not self.A_factors:  # First task
            self.A_factors = new_A_factors
            self.G_factors = new_G_factors
        else:
            # Online update with gamma decay
            for layer_name in new_A_factors:
                if layer_name in self.A_factors:
                    self.A_factors[layer_name] = (self.args.gamma * self.A_factors[layer_name] + 
                                                 new_A_factors[layer_name])
                    self.G_factors[layer_name] = (self.args.gamma * self.G_factors[layer_name] + 
                                                 new_G_factors[layer_name])
                else:
                    self.A_factors[layer_name] = new_A_factors[layer_name]
                    self.G_factors[layer_name] = new_G_factors[layer_name]

    def penalty(self):
        """Compute K-EWC penalty using Kronecker-factored approximation"""
        if self.checkpoint is None or not self.A_factors:
            return torch.tensor(0.0).to(self.device)
        
        total_penalty = torch.tensor(0.0).to(self.device)
        
        # Get current parameters organized by layer
        current_params = {}
        checkpoint_params = {}
        
        param_idx = 0
        for name, module in self.net.named_modules():
            if isinstance(module, nn.Linear):
                layer_name = self._get_layer_name(module, name)
                
                # Get weight and bias parameters
                weight_size = module.weight.numel()
                bias_size = module.bias.numel() if module.bias is not None else 0
                total_size = weight_size + bias_size
                
                # Extract current parameters
                current_layer_params_flat = self.net.get_params()[param_idx:param_idx + total_size]
                checkpoint_layer_params_flat = self.checkpoint[param_idx:param_idx + total_size]
                
                # Reshape to [out_features, in_features + 1] (including bias)
                if module.bias is not None:
                    current_W = current_layer_params_flat[:weight_size].view(module.out_features, module.in_features)
                    current_b = current_layer_params_flat[weight_size:].view(module.out_features, 1)
                    current_params[layer_name] = torch.cat([current_W, current_b], dim=1)
                    
                    checkpoint_W = checkpoint_layer_params_flat[:weight_size].view(module.out_features, module.in_features)
                    checkpoint_b = checkpoint_layer_params_flat[weight_size:].view(module.out_features, 1)
                    checkpoint_params[layer_name] = torch.cat([checkpoint_W, checkpoint_b], dim=1)
                else:
                    current_params[layer_name] = current_layer_params_flat.view(module.out_features, module.in_features)
                    checkpoint_params[layer_name] = checkpoint_layer_params_flat.view(module.out_features, module.in_features)
                
                param_idx += total_size
        
        # Compute penalty for each layer
        for layer_name in current_params:
            if layer_name in self.A_factors and layer_name in self.G_factors:
                delta_W = current_params[layer_name] - checkpoint_params[layer_name]
                
                A = self.A_factors[layer_name]
                G = self.G_factors[layer_name]
                
                # **SỬA LỖI TÍNH TOÁN**
                # Công thức đúng: Tr(G * delta_W.T * A * delta_W)
                # Cách tính hiệu quả: sum((delta_W.T @ G @ delta_W) * A)
                
                # Bước 1: (delta_W.T @ G) -> [(in+1), out] @ [out, out] = [(in+1), out]
                # Bước 2: (result_step1 @ delta_W) -> [(in+1), out] @ [out, (in+1)] = [(in+1), (in+1)]
                temp = torch.mm(torch.mm(delta_W.t(), G), delta_W)
                
                # Bước 3: Element-wise product rồi sum, tương đương với Trace
                penalty_layer = 0.5 * torch.sum(A * temp)
                
                total_penalty += penalty_layer
                
        return total_penalty

    def end_task(self, dataset):
        """End of task processing to compute and update Kronecker factors"""
        # Clear previous collections
        self.activations = {}
        self.gradients = {}
        
        # Register hooks for data collection
        self._register_hooks()
        
        # Collect activations and gradients over the dataset
        self.net.eval()
        with torch.enable_grad():
            for j, data in enumerate(dataset.train_loader):
                inputs, labels, _ = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                for ex, lab in zip(inputs, labels):
                    self.opt.zero_grad()
                    output = self.net(ex.unsqueeze(0))
                    
                    # Compute negative log-likelihood as in original EWC
                    loss = -F.nll_loss(self.logsoft(output), lab.unsqueeze(0), reduction='none')
                    exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                    loss = torch.mean(loss) * exp_cond_prob
                    
                    loss.backward()
        
        # Compute new Kronecker factors
        new_A_factors, new_G_factors = self._compute_kronecker_factors()
        
        # Update factors with online averaging
        self._update_kronecker_factors(new_A_factors, new_G_factors)
        
        # Store checkpoint parameters
        self.checkpoint = self.net.get_params().data.clone()
        
        # Clean up hooks
        self._cleanup_hooks()
        
        # Clear temporary storage
        self.activations = {}
        self.gradients = {}
        
        self.net.train()

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        penalty = self.penalty()
        
        if self.CPRon:
            loss = self.loss(outputs, labels) + self.args.e_lambda * penalty - self.beta * self.cpr(outputs)
        else:
            loss = self.loss(outputs, labels) + self.args.e_lambda * penalty
            
        assert not torch.isnan(loss)
        loss.backward()
        self.opt.step()

        return loss.item()