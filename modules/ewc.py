"""
Elastic Weight Consolidation (EWC) implementation for POLARIS.

This module provides functionality to calculate Fisher information matrices
and EWC loss for mitigating catastrophic forgetting in neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, List, Tuple, Optional, Union, Set


class EWCLoss:
    """
    Elastic Weight Consolidation (EWC) loss for preventing catastrophic forgetting.
    
    This implementation supports both standard EWC and online EWC variants.
    """
    
    def __init__(
        self,
        model: nn.Module,
        importance: float = 1000.0,
        online: bool = False,
        gamma: float = 0.95,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the EWC loss.
        
        Args:
            model: The model to protect from catastrophic forgetting
            importance: The importance factor for the EWC penalty (lambda)
            online: Whether to use online EWC (True) or standard EWC (False)
            gamma: Decay factor for online EWC (only used if online=True)
            device: Device to use for computations
        """
        self.model = model
        self.importance = importance
        self.online = online
        self.gamma = gamma
        self.device = device if device is not None else torch.device('cpu')
        
        # Initialize dictionaries to store parameters and Fisher information
        self.params = {}
        self.fisher_matrices = {}
        self.task_count = 0
        
        # Store parameter names for easier access
        self.param_names = [name for name, _ in model.named_parameters() if _.requires_grad]
    
    def register_task(self, fisher_matrix: Dict[str, torch.Tensor]):
        """
        Register a new task with its Fisher information matrix.
        
        Args:
            fisher_matrix: Dictionary mapping parameter names to their Fisher information matrices
        """
        # Increment task counter
        self.task_count += 1
        
        # Store current model parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Clone the parameter to avoid reference issues
                self.params[name] = param.data.clone()
        
        # Update Fisher information matrices
        if self.online and self.task_count > 1:
            # Online EWC: update existing Fisher matrices
            for name in self.param_names:
                if name in self.fisher_matrices and name in fisher_matrix:
                    # Apply decay to old Fisher and add new Fisher
                    self.fisher_matrices[name] = self.gamma * self.fisher_matrices[name] + fisher_matrix[name]
                elif name in fisher_matrix:
                    # First time seeing this parameter
                    self.fisher_matrices[name] = fisher_matrix[name]
        else:
            # Standard EWC: store Fisher matrices separately for each task
            self.fisher_matrices = fisher_matrix
    
    def calculate_loss(self) -> torch.Tensor:
        """
        Calculate the EWC loss based on registered tasks.
        
        Returns:
            The EWC loss tensor
        """
        loss = torch.tensor(0.0, device=self.device)
        
        # Skip if no tasks have been registered
        if self.task_count == 0:
            return loss
        
        # Calculate EWC loss
        for name, param in self.model.named_parameters():
            if name in self.fisher_matrices and name in self.params and param.requires_grad:
                # Get the Fisher information matrix for this parameter
                fisher = self.fisher_matrices[name]
                
                # Get the old parameter values
                old_param = self.params[name]
                
                # Calculate the squared difference weighted by Fisher information
                # Use L1 regularization for more stability
                param_diff = (param - old_param).abs()
                
                # Combine L1 and L2 regularization for better stability
                l2_penalty = (fisher * (param - old_param).pow(2)).sum()
                l1_penalty = (fisher.sqrt() * param_diff).sum()
                
                # Add both penalties with more weight on L2
                loss += l2_penalty + 0.1 * l1_penalty
        
        # Apply importance factor
        loss *= self.importance / 2.0
        
        return loss


def calculate_fisher_matrix(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: callable,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Calculate the Fisher information matrix for a model using a dataset.
    
    Args:
        model: The model to calculate Fisher information for
        data_loader: DataLoader containing the dataset
        loss_fn: Loss function to use for calculating gradients
        device: Device to use for computations
        
    Returns:
        Dictionary mapping parameter names to their Fisher information matrices
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize Fisher matrices
    fisher_matrices = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_matrices[name] = torch.zeros_like(param, device=device)
    
    # Process each batch
    for batch in data_loader:
        # Forward pass
        model.zero_grad()
        loss = loss_fn(model, batch)
        
        # Backward pass
        loss.backward()
        
        # Accumulate squared gradients (Fisher information)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_matrices[name] += param.grad.pow(2).data
    
    # Normalize by the number of samples
    for name in fisher_matrices:
        fisher_matrices[name] /= len(data_loader)
    
    return fisher_matrices


def calculate_fisher_from_replay_buffer(
    model: nn.Module,
    replay_buffer,
    loss_fn: callable,
    device: torch.device,
    batch_size: int = 64,
    num_batches: int = 20  # Increased from 10 to 20 for better estimation
) -> Dict[str, torch.Tensor]:
    """
    Calculate the Fisher information matrix using samples from a replay buffer.
    
    Args:
        model: The model to calculate Fisher information for
        replay_buffer: Replay buffer containing transitions
        loss_fn: Loss function to use for calculating gradients
        device: Device to use for computations
        batch_size: Batch size for sampling from replay buffer
        num_batches: Number of batches to process
        
    Returns:
        Dictionary mapping parameter names to their Fisher information matrices
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize Fisher matrices
    fisher_matrices = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_matrices[name] = torch.zeros_like(param, device=device)
    
    # Process multiple batches
    valid_batches = 0
    max_attempts = num_batches * 2  # Allow more attempts to get valid batches
    attempts = 0
    
    while valid_batches < num_batches and attempts < max_attempts:
        attempts += 1
        
        # Sample batch from replay buffer
        batch = replay_buffer.sample(batch_size)
        if batch is None:
            continue
        
        try:
            # Forward pass
            model.zero_grad()
            
            # Make sure all tensors require gradients
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.requires_grad_(True)
            
            # Calculate loss
            loss = loss_fn(model, batch)
            
            # Skip if loss is not a tensor or doesn't require grad
            if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                continue
                
            # Backward pass
            loss.backward()
            
            # Check if gradients are valid (not NaN or Inf)
            valid_grads = True
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        valid_grads = False
                        break
            
            if not valid_grads:
                print("Skipping batch with NaN or Inf gradients")
                continue
                
            # Accumulate squared gradients (Fisher information)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Use absolute values to ensure positive Fisher values
                    fisher_matrices[name] += param.grad.pow(2).abs().data
            
            valid_batches += 1
            if valid_batches % 5 == 0:
                print(f"Processed {valid_batches}/{num_batches} valid batches for Fisher calculation")
                
        except Exception as e:
            print(f"Error calculating Fisher matrix for batch: {e}")
            continue
    
    # Normalize by the number of valid batches
    if valid_batches > 0:
        print(f"Normalizing Fisher matrices using {valid_batches} valid batches")
        for name in fisher_matrices:
            fisher_matrices[name] /= valid_batches
            
            # Add a small constant to avoid zeros in the Fisher matrix
            fisher_matrices[name] += 1e-8
    else:
        print("Warning: No valid batches for Fisher calculation!")
    
    # Check if Fisher matrices contain valid values
    for name, matrix in fisher_matrices.items():
        if torch.isnan(matrix).any() or torch.isinf(matrix).any():
            print(f"Warning: Fisher matrix for {name} contains NaN or Inf values!")
            # Replace with small constant
            fisher_matrices[name] = torch.ones_like(matrix) * 1e-8
    
    return fisher_matrices