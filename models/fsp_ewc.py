# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via FSP-EWC with Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--lambda_fsp', type=float, default=10.0,
                        help='FSP penalty coefficient')
    parser.add_argument('--buffer_size', type=int, default=500,
                        help='Memory buffer capacity')
    parser.add_argument('--minibatch_size', type=int, default=32,
                        help='Minibatch size for replay')
    parser.add_argument('--length_scale', type=float, default=1.0,
                        help='RBF kernel length scale')
    parser.add_argument('--epsilon', type=float, default=1e-6,
                        help='Numerical stability constant')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Decay factor for online FSP penalty accumulation')
    
    return parser


class FspEwc(ContinualModel):
    NAME = 'fsp_ewc'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(FspEwc, self).__init__(backbone, loss, args, transform)
        
        # Memory buffer để lưu các mẫu cũ
        self.memory_buffer = Buffer(args.buffer_size, self.device)
        
        # Siêu tham số
        self.lambda_fsp = args.lambda_fsp
        self.minibatch_size = args.minibatch_size
        self.length_scale = args.length_scale
        self.epsilon = args.epsilon
        self.gamma = args.gamma
        
        # Online FSP penalty components
        self.fsp_penalty = None  # Accumulated FSP penalty
        self.checkpoint = None   # Parameters checkpoint
        self.context_points = None  # Context points for FSP
        self.kernel_matrix = None   # Precomputed kernel matrix
        self.task = 0

    def rbf_kernel(self, X1, X2):
        """
        Tính toán ma trận kernel RBF giữa hai tập điểm X1 và X2
        """
        # X1: (N1, D), X2: (N2, D)
        # Normalize inputs để tránh giá trị quá lớn
        X1 = X1 / (torch.norm(X1, dim=1, keepdim=True) + 1e-8)
        X2 = X2 / (torch.norm(X2, dim=1, keepdim=True) + 1e-8)
        
        # Tính khoảng cách Euclidean
        X1_norm = (X1**2).sum(dim=1, keepdim=True)  # (N1, 1)
        X2_norm = (X2**2).sum(dim=1)                # (N2,)
        
        # Broadcasting để tính ||x1 - x2||^2
        dist_sq = X1_norm - 2 * torch.matmul(X1, X2.t()) + X2_norm
        
        # Clamp để tránh giá trị âm do floating point errors
        dist_sq = torch.clamp(dist_sq, min=0.0)
        
        # RBF kernel với capping để tránh underflow/overflow
        exponent = -dist_sq / (2 * self.length_scale**2)
        exponent = torch.clamp(exponent, min=-50, max=50)  # Giới hạn để tránh exp overflow
        K = torch.exp(exponent)
        
        return K

    def penalty(self):
        """
        Tính toán FSP penalty online tích lũy qua các task
        """
        if self.checkpoint is None or self.context_points is None:
            return torch.tensor(0.0).to(self.device)
            
        try:
            # Tính output hiện tại và output tại checkpoint
            with torch.no_grad():
                f_checkpoint = self.checkpoint['outputs']  # Outputs tại checkpoint
            
            f_current = self.net(self.context_points)
            
            # Tính chênh lệch function outputs
            delta_f = f_current - f_checkpoint
            
            # Kiểm tra validity
            if torch.isnan(delta_f).any() or torch.isinf(delta_f).any():
                return torch.tensor(0.0).to(self.device)
            
            # Tính FSP penalty sử dụng kernel matrix đã tính
            if self.kernel_matrix is not None:
                penalty = 0.5 * torch.trace(torch.matmul(torch.matmul(delta_f.t(), self.kernel_matrix), delta_f))
                penalty = torch.clamp(penalty, 0, 1000.0)
                return penalty * self.fsp_penalty if self.fsp_penalty is not None else penalty
            else:
                return torch.tensor(0.0).to(self.device)
                
        except Exception as e:
            return torch.tensor(0.0).to(self.device)

    def observe(self, inputs, labels, not_aug_inputs):
        """
        Thực hiện một bước training với online FSP-EWC và Experience Replay
        """
        self.opt.zero_grad()
        
        # 1. Loss trên dữ liệu mới (Standard Loss)
        outputs_new = self.net(inputs)
        loss_new = self.loss(outputs_new, labels)
        
        # 2. FSP penalty (online accumulated)
        fsp_penalty = self.penalty()
        
        # 3. Experience Replay Loss
        loss_replay = torch.tensor(0.0).to(self.device)
        if not self.memory_buffer.is_empty():
            buffer_size = min(self.minibatch_size, len(self.memory_buffer))
            buffer_data = self.memory_buffer.get_data(buffer_size, transform=self.transform)
            x_replay, y_replay = buffer_data[0], buffer_data[1]
            
            outputs_replay = self.net(x_replay)
            loss_replay = self.loss(outputs_replay, y_replay)

        # Tổng hợp loss
        total_loss = loss_new + loss_replay + self.lambda_fsp * fsp_penalty
        
        # Kiểm tra NaN/Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: NaN/Inf detected - loss_new: {loss_new.item()}, loss_replay: {loss_replay.item()}, fsp_penalty: {fsp_penalty.item()}")
            return 0.0
        
        # Backpropagation với gradient clipping
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.opt.step()

        return total_loss.item()

    def end_task(self, dataset):
        """
        Tính toán và tích lũy FSP penalty online sau khi hoàn thành task
        """
        # Thu thập context points từ dataset hiện tại (subsample để tiết kiệm memory)
        context_inputs = []
        context_outputs = []
        max_samples = 100  # Giới hạn số samples để tránh memory overflow
        
        sample_count = 0
        for data in dataset.train_loader:
            if sample_count >= max_samples:
                break
            inputs, labels, _ = data
            inputs = inputs.to(self.device)
            
            # Lấy outputs của model hiện tại
            with torch.no_grad():
                outputs = self.net(inputs)
            
            # Chỉ lấy một số samples từ batch
            batch_samples = min(inputs.shape[0], max_samples - sample_count)
            context_inputs.append(inputs[:batch_samples])
            context_outputs.append(outputs[:batch_samples])
            sample_count += batch_samples
        
        if len(context_inputs) > 0:
            # Concatenate tất cả context points
            new_context_points = torch.cat(context_inputs, dim=0)
            new_context_outputs = torch.cat(context_outputs, dim=0)
            
            # Tính kernel matrix cho context points mới
            num_context = new_context_points.shape[0]
            if num_context > 0:
                try:
                    x_flat = new_context_points.view(num_context, -1)
                    K = self.rbf_kernel(x_flat, x_flat)
                    
                    # Regularization
                    epsilon = max(self.epsilon, 1e-4)
                    identity = torch.eye(num_context).to(self.device)
                    K_inv = torch.inverse(K + epsilon * identity)
                    
                    # Update context points và kernel matrix
                    self.context_points = new_context_points
                    self.kernel_matrix = K_inv
                    
                    # Update checkpoint với outputs hiện tại
                    self.checkpoint = {
                        'outputs': new_context_outputs.detach().clone()
                    }
                    
                    # Tích lũy FSP penalty weight (giống gamma trong EWC-ON)
                    if self.fsp_penalty is None:
                        self.fsp_penalty = 1.0
                    else:
                        self.fsp_penalty = self.fsp_penalty * self.gamma + 1.0
                        
                except Exception as e:
                    print(f"Error in FSP computation: {e}")
        
        # Cập nhật memory buffer
        for data in dataset.train_loader:
            inputs, labels, _ = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.memory_buffer.add_data(examples=inputs, labels=labels)
        
        self.task += 1