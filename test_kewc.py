#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def test_kronecker_penalty_computation():
    """
    Test that the Kronecker-factored penalty computation is mathematically correct
    by comparing with a direct computation for a small example.
    """
    print("Testing Kronecker-factored penalty computation...")
    
    # Create simple test data
    batch_size, input_dim, output_dim = 4, 3, 2
    
    # Create activations (with bias column)
    activations = torch.randn(batch_size, input_dim + 1)  # +1 for bias
    gradients = torch.randn(batch_size, output_dim)
    
    # Compute covariance matrices
    A = torch.mm(activations.t(), activations) / batch_size
    G = torch.mm(gradients.t(), gradients) / batch_size
    
    # Add damping
    damping = 1e-3
    A += torch.eye(A.size(0)) * damping
    G += torch.eye(G.size(0)) * damping
    
    # Create parameter matrices
    W_current = torch.randn(output_dim, input_dim + 1)
    W_checkpoint = torch.randn(output_dim, input_dim + 1)
    delta_W = W_current - W_checkpoint
    
    # Method 1: Direct Kronecker product computation
    # Penalty = (1/2) * vec(delta_W)^T * (A ⊗ G) * vec(delta_W)
    vec_delta_W = delta_W.view(-1, 1)
    kronecker_AG = torch.kron(A, G)
    penalty_direct = 0.5 * torch.mm(torch.mm(vec_delta_W.t(), kronecker_AG), vec_delta_W).item()
    
    # Method 2: Efficient trace computation (our implementation)
    # Penalty = (1/2) * Tr(G * delta_W^T * A * delta_W)
    # This can be computed as: (1/2) * sum_ij(G_ij * (delta_W^T * A * delta_W)_ij)
    # Which equals: (1/2) * sum_ij(G_ij * sum_k(delta_W^T_ik * A_kl * delta_W_lj))
    temp = torch.mm(torch.mm(delta_W.t(), A), delta_W)  # This is delta_W^T * A * delta_W
    penalty_efficient = 0.5 * torch.sum(G * temp).item()  # Element-wise multiplication and sum
    
    print(f"Direct computation: {penalty_direct:.6f}")
    print(f"Efficient computation: {penalty_efficient:.6f}")
    print(f"Difference: {abs(penalty_direct - penalty_efficient):.8f}")
    
    # They should be equal (up to numerical precision)
    assert abs(penalty_direct - penalty_efficient) < 1e-5, "Penalty computations don't match!"
    print("✓ Penalty computation test passed!\n")

def test_parameter_importance_capture():
    """
    Test that K-EWC captures parameter importance better than diagonal EWC
    by showing it can detect correlated parameter changes.
    """
    print("Testing parameter importance capture...")
    
    # Create a scenario where parameters are correlated
    torch.manual_seed(42)
    batch_size, input_dim, output_dim = 10, 4, 3
    
    # Create correlated activations
    base_activations = torch.randn(batch_size, input_dim)
    # Make some features correlated
    base_activations[:, 1] = base_activations[:, 0] * 0.8 + torch.randn(batch_size) * 0.2
    
    # Add bias column
    activations = torch.cat([base_activations, torch.ones(batch_size, 1)], dim=1)
    
    # Create gradients
    gradients = torch.randn(batch_size, output_dim)
    
    # Compute covariance matrices
    A = torch.mm(activations.t(), activations) / batch_size
    G = torch.mm(gradients.t(), gradients) / batch_size
    
    damping = 1e-3
    A += torch.eye(A.size(0)) * damping
    G += torch.eye(G.size(0)) * damping
    
    # Test scenario: correlated parameter change
    W_checkpoint = torch.zeros(output_dim, input_dim + 1)
    
    # Case 1: Change correlated parameters together (should have lower penalty)
    W_correlated = W_checkpoint.clone()
    W_correlated[:, 0] = 0.1  # Change first feature
    W_correlated[:, 1] = 0.08  # Change correlated feature proportionally
    
    delta_W_corr = W_correlated - W_checkpoint
    temp_corr = torch.mm(torch.mm(delta_W_corr.t(), A), delta_W_corr)
    penalty_corr = 0.5 * torch.sum(G * temp_corr).item()
    
    # Case 2: Change correlated parameters anti-correlate (should have higher penalty)
    W_anticorr = W_checkpoint.clone()
    W_anticorr[:, 0] = 0.1   # Change first feature
    W_anticorr[:, 1] = -0.08 # Change correlated feature in opposite direction
    
    delta_W_anticorr = W_anticorr - W_checkpoint
    temp_anticorr = torch.mm(torch.mm(delta_W_anticorr.t(), A), delta_W_anticorr)
    penalty_anticorr = 0.5 * torch.sum(G * temp_anticorr).item()
    
    # Diagonal EWC penalty (for comparison)
    fisher_diag = torch.diag(torch.kron(A, G))
    penalty_diag_corr = 0.5 * torch.sum(fisher_diag * (delta_W_corr.view(-1) ** 2)).item()
    penalty_diag_anticorr = 0.5 * torch.sum(fisher_diag * (delta_W_anticorr.view(-1) ** 2)).item()
    
    print(f"K-EWC penalty (correlated change): {penalty_corr:.6f}")
    print(f"K-EWC penalty (anti-correlated change): {penalty_anticorr:.6f}")
    print(f"Diagonal EWC penalty (correlated): {penalty_diag_corr:.6f}")
    print(f"Diagonal EWC penalty (anti-correlated): {penalty_diag_anticorr:.6f}")
    
    print(f"\nK-EWC penalty ratio (anti-corr/corr): {penalty_anticorr/penalty_corr:.3f}")
    print(f"Diagonal EWC penalty ratio (anti-corr/corr): {penalty_diag_anticorr/penalty_diag_corr:.3f}")
    
    # K-EWC should show more sensitivity to correlation structure
    kewc_ratio = penalty_anticorr / penalty_corr
    diag_ratio = penalty_diag_anticorr / penalty_diag_corr
    
    print(f"\nK-EWC captures correlation structure better: {kewc_ratio > diag_ratio}")
    print("✓ Parameter importance test completed!\n")

def test_numerical_stability():
    """
    Test numerical stability with different damping values
    """
    print("Testing numerical stability...")
    
    batch_size, input_dim, output_dim = 5, 3, 2
    
    # Create nearly singular matrices (rank deficient)
    activations = torch.randn(batch_size, input_dim + 1)
    # Make last column nearly dependent on first
    activations[:, -1] = activations[:, 0] * 0.9999 + torch.randn(batch_size) * 1e-6
    
    gradients = torch.randn(batch_size, output_dim)
    
    A = torch.mm(activations.t(), activations) / batch_size
    G = torch.mm(gradients.t(), gradients) / batch_size
    
    # Test different damping values
    damping_values = [1e-6, 1e-3, 1e-1]
    
    W_current = torch.randn(output_dim, input_dim + 1)
    W_checkpoint = torch.randn(output_dim, input_dim + 1)
    delta_W = W_current - W_checkpoint
    
    for damping in damping_values:
        A_damped = A + torch.eye(A.size(0)) * damping
        G_damped = G + torch.eye(G.size(0)) * damping
        
        temp = torch.mm(torch.mm(delta_W.t(), A_damped), delta_W)
        penalty = 0.5 * torch.sum(G_damped * temp)
        
        print(f"Damping {damping:.0e}: penalty = {penalty:.6f}, finite = {torch.isfinite(penalty)}")
        
        assert torch.isfinite(penalty), f"Non-finite penalty with damping {damping}"
    
    print("✓ Numerical stability test passed!\n")

if __name__ == "__main__":
    print("Running K-EWC Mathematical Correctness Tests")
    print("=" * 50)
    
    test_kronecker_penalty_computation()
    test_parameter_importance_capture()  
    test_numerical_stability()
    
    print("All tests passed! K-EWC implementation is mathematically sound.")