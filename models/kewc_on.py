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
        
        # Memory-efficient streaming computation
        self.A_running = {}  # Running sum for A factors
        self.G_running = {}  # Running sum for G factors
        self.sample_count = 0
        
        self.cpr = CPR()
        self.beta = 0.0
        self.CPRon = False
        
        self.damping = args.damping if hasattr(args, 'damping') else 1e-3
        
        # Memory optimization settings
        self.max_samples_per_batch = getattr(args, 'kfac_max_samples', 32)
        self.use_cpu_storage = getattr(args, 'kfac_cpu_storage', True)
        self.conv_sample_ratio = getattr(args, 'conv_sample_ratio', 0.1)  # Sample 10% of conv patches

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

    def _register_hooks(self):
        """Register memory-efficient hooks for streaming computation"""
        def make_forward_hook(layer_name, module):
            def forward_hook(module_inner, input, output):
                if isinstance(module_inner, nn.Linear):
                    # For linear layers, compute A factor incrementally
                    act = input[0].detach()
                    if act.dim() > 2:
                        act = act.view(act.size(0), -1)
                    
                    # Add bias column
                    ones = torch.ones(act.size(0), 1, device=act.device)
                    act_with_bias = torch.cat([act, ones], dim=1)
                    
                    # Compute A factor incrementally to save memory
                    self._update_A_factor_streaming(layer_name, act_with_bias)
                    
                elif isinstance(module_inner, nn.Conv2d):
                    # Memory-efficient conv processing with sampling
                    act = input[0].detach()
                    
                    # Sample patches to reduce memory usage
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
                    if patches.size(0) > 1000:  # Only sample if too many patches
                        n_samples = max(100, int(patches.size(0) * self.conv_sample_ratio))
                        idx = torch.randperm(patches.size(0), device=patches.device)[:n_samples]
                        patches = patches[idx]
                    
                    # Add bias column
                    ones = torch.ones(patches.size(0), 1, device=patches.device)
                    patches_with_bias = torch.cat([patches, ones], dim=1)
                    
                    self._update_A_factor_streaming(layer_name, patches_with_bias)
                    
            return forward_hook

        def make_backward_hook(layer_name, module):
            def backward_hook(module_inner, grad_input, grad_output):
                if isinstance(module_inner, nn.Linear) and grad_output[0] is not None:
                    grad = grad_output[0].detach()
                    if grad.dim() > 2:
                        grad = grad.view(grad.size(0), -1)
                    
                    self._update_G_factor_streaming(layer_name, grad)
                    
                elif isinstance(module_inner, nn.Conv2d) and grad_output[0] is not None:
                    grad = grad_output[0].detach()
                    batch_size, out_channels, h_out, w_out = grad.shape
                    grad = grad.permute(0, 2, 3, 1).contiguous().view(-1, out_channels)
                    
                    # Sample gradients to reduce memory
                    if grad.size(0) > 1000:
                        n_samples = max(100, int(grad.size(0) * self.conv_sample_ratio))
                        idx = torch.randperm(grad.size(0), device=grad.device)[:n_samples]
                        grad = grad[idx]
                    
                    self._update_G_factor_streaming(layer_name, grad)
                    
            return backward_hook

        # Register hooks for all Linear and Conv2d layers
        for name, module in self.net.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layer_name = self._get_layer_name(module, name)
                
                fh = module.register_forward_hook(make_forward_hook(layer_name, module))
                bh = module.register_full_backward_hook(make_backward_hook(layer_name, module))
                
                self.forward_handles.append(fh)
                self.backward_handles.append(bh)

    def _cleanup_hooks(self):
        """Remove all registered hooks"""
        for handle in self.forward_handles + self.backward_handles:
            handle.remove()
        self.forward_handles = []
        self.backward_handles = []

    def _update_A_factor_streaming(self, layer_name, activations):
        """Update A factor incrementally to save memory"""
        # Compute outer product incrementally
        A_batch = torch.mm(activations.t(), activations)
        
        if layer_name not in self.A_running:
            self.A_running[layer_name] = A_batch.cpu() if self.use_cpu_storage else A_batch
        else:
            if self.use_cpu_storage:
                self.A_running[layer_name] += A_batch.cpu()
            else:
                self.A_running[layer_name] += A_batch
    
    def _update_G_factor_streaming(self, layer_name, gradients):
        """Update G factor incrementally to save memory"""
        # Compute outer product incrementally
        G_batch = torch.mm(gradients.t(), gradients)
        
        if layer_name not in self.G_running:
            self.G_running[layer_name] = G_batch.cpu() if self.use_cpu_storage else G_batch
        else:
            if self.use_cpu_storage:
                self.G_running[layer_name] += G_batch.cpu()
            else:
                self.G_running[layer_name] += G_batch
    
    def _finalize_kronecker_factors(self):
        """Finalize Kronecker factors from streaming computation"""
        new_A_factors = {}
        new_G_factors = {}
        
        for layer_name in self.A_running:
            if layer_name in self.G_running:
                # Normalize by sample count and add damping
                A = self.A_running[layer_name] / self.sample_count
                G = self.G_running[layer_name] / self.sample_count
                
                # Move back to GPU if stored on CPU
                if self.use_cpu_storage:
                    A = A.to(self.device)
                    G = G.to(self.device)
                
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
        """Memory-efficient K-EWC penalty computation"""
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
            
            elif isinstance(module, nn.Conv2d):
                layer_name = self._get_layer_name(module, name)
                
                # Get weight and bias parameters
                weight_size = module.weight.numel()
                bias_size = module.bias.numel() if module.bias is not None else 0
                total_size = weight_size + bias_size
                
                # Extract current parameters
                current_layer_params_flat = self.net.get_params()[param_idx:param_idx + total_size]
                checkpoint_layer_params_flat = self.checkpoint[param_idx:param_idx + total_size]
                
                # Reshape conv weights from 4D to 2D matrix
                # [out_channels, in_channels, kernel_h, kernel_w] -> [out_channels, in_channels * kernel_h * kernel_w]
                out_channels, in_channels, kernel_h, kernel_w = module.weight.shape
                flat_weight_dim = in_channels * kernel_h * kernel_w
                
                if module.bias is not None:
                    current_W = current_layer_params_flat[:weight_size].view(out_channels, flat_weight_dim)
                    current_b = current_layer_params_flat[weight_size:].view(out_channels, 1)
                    current_params[layer_name] = torch.cat([current_W, current_b], dim=1)
                    
                    checkpoint_W = checkpoint_layer_params_flat[:weight_size].view(out_channels, flat_weight_dim)
                    checkpoint_b = checkpoint_layer_params_flat[weight_size:].view(out_channels, 1)
                    checkpoint_params[layer_name] = torch.cat([checkpoint_W, checkpoint_b], dim=1)
                else:
                    current_params[layer_name] = current_layer_params_flat.view(out_channels, flat_weight_dim)
                    checkpoint_params[layer_name] = checkpoint_layer_params_flat.view(out_channels, flat_weight_dim)
                
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
                
                # Memory-efficient trace computation
                # Move matrices to CPU if they're large to save GPU memory
                if A.numel() > 10000 and self.use_cpu_storage:
                    A_cpu = A.cpu()
                    temp_cpu = temp.cpu()
                    penalty_layer = 0.5 * torch.sum(A_cpu * temp_cpu)
                    penalty_layer = penalty_layer.to(self.device)
                else:
                    penalty_layer = 0.5 * torch.sum(A * temp)
                
                total_penalty += penalty_layer
                
                # Clear intermediate tensors
                del temp
                if 'A_cpu' in locals():
                    del A_cpu, temp_cpu
                
        return total_penalty

    def end_task(self, dataset):
        """Memory-efficient end of task processing"""
        # Initialize streaming computation
        self.A_running = {}
        self.G_running = {}
        self.sample_count = 0
        
        # Register hooks for streaming data collection
        self._register_hooks()
        
        # Process data in smaller batches to avoid memory issues
        self.net.eval()
        with torch.enable_grad():
            for j, data in enumerate(dataset.train_loader):
                inputs, labels, _ = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Process in smaller mini-batches if needed
                batch_size = inputs.size(0)
                mini_batch_size = min(self.max_samples_per_batch, batch_size)
                
                for i in range(0, batch_size, mini_batch_size):
                    mini_inputs = inputs[i:i+mini_batch_size]
                    mini_labels = labels[i:i+mini_batch_size]
                    
                    self.opt.zero_grad()
                    output = self.net(mini_inputs)
                    
                    # Compute loss for each sample
                    for k, (ex_out, lab) in enumerate(zip(output, mini_labels)):
                        loss = -F.nll_loss(self.logsoft(ex_out.unsqueeze(0)), lab.unsqueeze(0), reduction='none')
                        exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                        loss = torch.mean(loss) * exp_cond_prob
                        loss.backward(retain_graph=k < len(mini_labels)-1)
                        
                        self.sample_count += 1
                    
                    # Clear GPU cache periodically
                    if j % 10 == 0:
                        torch.cuda.empty_cache()
        
        # Finalize Kronecker factors from streaming computation
        new_A_factors, new_G_factors = self._finalize_kronecker_factors()
        
        # Update factors with online averaging
        self._update_kronecker_factors(new_A_factors, new_G_factors)
        
        # Store checkpoint parameters
        self.checkpoint = self.net.get_params().data.clone()
        
        # Clean up
        self._cleanup_hooks()
        self.A_running = {}
        self.G_running = {}
        
        # Final GPU cleanup
        torch.cuda.empty_cache()
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