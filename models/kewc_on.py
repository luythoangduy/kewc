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
        self.A_factors = {}  # Final activation covariance matrices
        self.G_factors = {}  # Final gradient covariance matrices
        
        # Running factors for online updates
        self.A_running = {}  # Running A factors with EMA
        self.G_running = {}  # Running G factors with EMA
        
        # Hook handles for cleanup
        self.forward_handles = []
        self.backward_handles = []
        
        # Online training state
        self.training_steps = 0
        self.ema_decay = getattr(args, 'kfac_ema_decay', 0.95)  # EMA decay rate
        self.hooks_registered = False
        
        self.cpr = CPR()
        self.beta = 0.0
        self.CPRon = False
        
        self.damping = args.damping if hasattr(args, 'damping') else 1e-3
        
        # Memory optimization settings
        self.conv_sample_ratio = getattr(args, 'conv_sample_ratio', 0.1)  # Sample 10% of conv patches
        self.update_freq = getattr(args, 'kfac_update_freq', 1)  # Update K-FAC every N steps

    def _get_layer_name(self, module, name):
        """Generate unique layer name for indexing"""
        return f"{name}_{id(module)}"
    
    def _get_conv_patches_info(self, module, input_tensor):
        """Helper function to compute patch information for Conv2d layers"""
        # Calculate output dimensions
        batch_size, in_channels, h_in, w_in = input_tensor.shape
        kernel_h, kernel_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        padding_h, padding_w = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
        stride_h, stride_w = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
        
        h_out = (h_in + 2 * padding_h - kernel_h) // stride_h + 1
        w_out = (w_in + 2 * padding_w - kernel_w) // stride_w + 1
        
        return batch_size, in_channels, kernel_h, kernel_w, h_out, w_out

    def _register_hooks_for_observe(self):
        """Register hooks for online K-FAC updates during training"""
        if self.hooks_registered:
            return
            
        def make_forward_hook(layer_name, module):
            def forward_hook(module_inner, input, output):
                # Only update during training and at specified frequency
                if not self.net.training or self.training_steps % self.update_freq != 0:
                    return
                    
                if isinstance(module_inner, nn.Linear):
                    act = input[0].detach()
                    if act.dim() > 2:
                        act = act.view(act.size(0), -1)
                    
                    # Add bias column
                    ones = torch.ones(act.size(0), 1, device=act.device)
                    act_with_bias = torch.cat([act, ones], dim=1)
                    
                    self._update_A_factor_online(layer_name, act_with_bias)
                    
                elif isinstance(module_inner, nn.Conv2d):
                    act = input[0].detach()
                    
                    patches = F.unfold(
                        act,
                        kernel_size=module_inner.kernel_size,
                        padding=module_inner.padding,
                        stride=module_inner.stride,
                        dilation=module_inner.dilation
                    )
                    
                    batch_size, patch_dim, num_patches = patches.shape
                    patches = patches.transpose(1, 2).contiguous().view(-1, patch_dim)
                    
                    # Sample patches to reduce memory
                    if patches.size(0) > 1000:
                        n_samples = max(100, int(patches.size(0) * self.conv_sample_ratio))
                        idx = torch.randperm(patches.size(0), device=patches.device)[:n_samples]
                        patches = patches[idx]
                    
                    # Add bias column
                    ones = torch.ones(patches.size(0), 1, device=patches.device)
                    patches_with_bias = torch.cat([patches, ones], dim=1)
                    
                    self._update_A_factor_online(layer_name, patches_with_bias)
                    
            return forward_hook

        def make_backward_hook(layer_name, module):
            def backward_hook(module_inner, grad_input, grad_output):
                # Only update during training and at specified frequency
                if not self.net.training or self.training_steps % self.update_freq != 0:
                    return
                    
                if isinstance(module_inner, nn.Linear) and grad_output[0] is not None:
                    grad = grad_output[0].detach()
                    if grad.dim() > 2:
                        grad = grad.view(grad.size(0), -1)
                    
                    self._update_G_factor_online(layer_name, grad)
                    
                elif isinstance(module_inner, nn.Conv2d) and grad_output[0] is not None:
                    grad = grad_output[0].detach()
                    batch_size, out_channels, h_out, w_out = grad.shape
                    grad = grad.permute(0, 2, 3, 1).contiguous().view(-1, out_channels)
                    
                    # Sample gradients
                    if grad.size(0) > 1000:
                        n_samples = max(100, int(grad.size(0) * self.conv_sample_ratio))
                        idx = torch.randperm(grad.size(0), device=grad.device)[:n_samples]
                        grad = grad[idx]
                    
                    self._update_G_factor_online(layer_name, grad)
                    
            return backward_hook

        # Register hooks for all Linear and Conv2d layers
        for name, module in self.net.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layer_name = self._get_layer_name(module, name)
                
                fh = module.register_forward_hook(make_forward_hook(layer_name, module))
                bh = module.register_full_backward_hook(make_backward_hook(layer_name, module))
                
                self.forward_handles.append(fh)
                self.backward_handles.append(bh)
        
        self.hooks_registered = True

    def _cleanup_hooks(self):
        """Remove all registered hooks"""
        for handle in self.forward_handles + self.backward_handles:
            handle.remove()
        self.forward_handles = []
        self.backward_handles = []

    def _update_A_factor_online(self, layer_name, activations):
        """Update A factor using Exponential Moving Average during training"""
        # Compute batch covariance
        A_batch = torch.mm(activations.t(), activations) / activations.size(0)
        
        if layer_name not in self.A_running:
            self.A_running[layer_name] = A_batch
        else:
            # EMA update: A_new = decay * A_old + (1 - decay) * A_batch
            self.A_running[layer_name] = (self.ema_decay * self.A_running[layer_name] + 
                                        (1 - self.ema_decay) * A_batch)
    
    def _update_G_factor_online(self, layer_name, gradients):
        """Update G factor using Exponential Moving Average during training"""
        # Compute batch covariance
        G_batch = torch.mm(gradients.t(), gradients) / gradients.size(0)
        
        if layer_name not in self.G_running:
            self.G_running[layer_name] = G_batch
        else:
            # EMA update: G_new = decay * G_old + (1 - decay) * G_batch
            self.G_running[layer_name] = (self.ema_decay * self.G_running[layer_name] + 
                                        (1 - self.ema_decay) * G_batch)

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
        """Memory-efficient K-EWC penalty computation"""
        if self.checkpoint is None or not self.A_factors:
            return torch.tensor(0.0).to(self.device)
        
        total_penalty = torch.tensor(0.0).to(self.device)
        
        # Simplified parameter extraction using EWC_ON style
        current_params = self.net.get_params()
        checkpoint_params = self.checkpoint
        
        # Organize parameters by layers for K-FAC computation
        layer_params = {}
        layer_checkpoint_params = {}
        
        param_idx = 0
        for name, module in self.net.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layer_name = self._get_layer_name(module, name)
                
                # Get parameter sizes
                weight_size = module.weight.numel()
                bias_size = module.bias.numel() if module.bias is not None else 0
                total_size = weight_size + bias_size
                
                # Extract parameters for this layer
                layer_params[layer_name] = current_params[param_idx:param_idx + total_size]
                layer_checkpoint_params[layer_name] = checkpoint_params[param_idx:param_idx + total_size]
                
                param_idx += total_size
        
        # Compute K-FAC penalty for each layer
        for layer_name in layer_params:
            if layer_name in self.A_factors and layer_name in self.G_factors:
                # Get parameter difference
                delta_params = layer_params[layer_name] - layer_checkpoint_params[layer_name]
                
                # Get corresponding module to reshape parameters correctly
                module = None
                for name, mod in self.net.named_modules():
                    if self._get_layer_name(mod, name) == layer_name:
                        module = mod
                        break
                
                if module is not None:
                    # Reshape parameters back to weight matrix form for K-FAC computation
                    if isinstance(module, nn.Linear):
                        weight_size = module.weight.numel()
                        if module.bias is not None:
                            delta_W = delta_params[:weight_size].view(module.out_features, module.in_features)
                            delta_b = delta_params[weight_size:].view(module.out_features, 1)
                            delta_W_full = torch.cat([delta_W, delta_b], dim=1)
                        else:
                            delta_W_full = delta_params.view(module.out_features, module.in_features)
                    
                    elif isinstance(module, nn.Conv2d):
                        weight_size = module.weight.numel()
                        out_channels, in_channels, kernel_h, kernel_w = module.weight.shape
                        flat_weight_dim = in_channels * kernel_h * kernel_w
                        
                        if module.bias is not None:
                            delta_W = delta_params[:weight_size].view(out_channels, flat_weight_dim)
                            delta_b = delta_params[weight_size:].view(out_channels, 1)
                            delta_W_full = torch.cat([delta_W, delta_b], dim=1)
                        else:
                            delta_W_full = delta_params.view(out_channels, flat_weight_dim)
                    
                    # Apply K-FAC formula: Tr(G * delta_W.T * A * delta_W)
                    A = self.A_factors[layer_name]
                    G = self.G_factors[layer_name]
                    
                    temp = torch.mm(torch.mm(delta_W_full.t(), G), delta_W_full)
                    penalty_layer = 0.5 * torch.sum(A * temp)
                    total_penalty += penalty_layer
                    
                    # Clear intermediate tensors
                    del temp
                
        return total_penalty

    def begin_task(self, dataset):
        """Initialize for new task - register hooks and reset running factors"""
        # Reset running factors for new task
        self.A_running = {}
        self.G_running = {}
        self.training_steps = 0
        
        # Register hooks for online updates
        self._register_hooks_for_observe()
    
    def end_task(self, dataset):
        """Finalize task - update global factors from running estimates"""
        # Convert running factors to final factors with damping
        for layer_name in self.A_running:
            if layer_name in self.G_running:
                # Add damping for numerical stability
                A = self.A_running[layer_name].clone()
                G = self.G_running[layer_name].clone()
                
                A.add_(torch.eye(A.size(0), device=A.device) * self.damping)
                G.add_(torch.eye(G.size(0), device=G.device) * self.damping)
                
                # Update global factors with online averaging (like EWC_ON)
                if layer_name not in self.A_factors:
                    self.A_factors[layer_name] = A
                    self.G_factors[layer_name] = G
                else:
                    self.A_factors[layer_name] = (self.args.gamma * self.A_factors[layer_name] + A)
                    self.G_factors[layer_name] = (self.args.gamma * self.G_factors[layer_name] + G)
        
        # Store checkpoint parameters
        self.checkpoint = self.net.get_params().data.clone()
        
        # Clean up hooks
        self._cleanup_hooks()
        self.hooks_registered = False
        
        # Clear running factors to save memory
        self.A_running = {}
        self.G_running = {}

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        
        # Forward pass - hooks will capture activations
        outputs = self.net(inputs)
        
        # Compute loss
        main_loss = self.loss(outputs, labels)
        penalty = self.penalty()
        
        if self.CPRon:
            loss = main_loss + self.args.e_lambda * penalty - self.beta * self.cpr(outputs)
        else:
            loss = main_loss + self.args.e_lambda * penalty
            
        assert not torch.isnan(loss)
        
        # Backward pass - hooks will capture gradients
        loss.backward()
        
        # Update training step counter for hook frequency control
        self.training_steps += 1
        
        self.opt.step()

        return loss.item()