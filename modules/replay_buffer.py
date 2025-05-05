from modules.utils import get_best_device
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    """Continuous replay buffer for storing transitions in FURTHER+."""
    def __init__(self, capacity, observation_dim, belief_dim, latent_dim, device=None, sequence_length=8):
        # Use the best available device if none is specified
        if device is None:
            device = get_best_device()
        self.capacity = capacity
        self.device = device
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=capacity)
        
    def push(self, observation, belief, latent, action, reward, 
             next_observation, next_belief, next_latent, mean=None, logvar=None, neighbor_actions=None):
        """Save a transition to the buffer."""
        transition = (observation, belief, latent, action, neighbor_actions, reward, 
                    next_observation, next_belief, next_latent, mean, logvar)
        self.buffer.append(transition)
    
    def end_trajectory(self):
        """
        Backward compatibility method - does nothing in the continuous version.
        """
        pass
    
    def sample(self, batch_size, sequence_length=None):
        """Sample a batch of sequential transitions from the buffer.
        
        Args:
            batch_size: Number of sequences to sample
            sequence_length: Length of each sequence (uses self.sequence_length if None)
            
        Returns:
            Batch of sequential transitions
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        # Make sure we have enough transitions
        if len(self.buffer) < sequence_length:
            return None
            
        # Adjust batch size if needed
        max_possible_sequences = len(self.buffer) - sequence_length + 1
        batch_size = min(batch_size, max_possible_sequences)
        
        # Sample starting indices for sequences
        start_indices = np.random.choice(max_possible_sequences, batch_size, replace=False)
        
        # Extract sequences
        sequences = []
        for start_idx in start_indices:
            sequence = [self.buffer[start_idx + i] for i in range(sequence_length)]
            sequences.append(sequence)
        
        # Organize the data
        batch_data = []
        for t in range(sequence_length):
            # Get all transitions at time step t across all sequences
            time_step_transitions = [seq[t] for seq in sequences]
            
            # Unpack transitions
            observations, beliefs, latents, actions, neighbor_actions, rewards, \
            next_observations, next_beliefs, next_latents, means, logvars = zip(*time_step_transitions)
            
            # Convert to tensors
            observations = torch.FloatTensor(np.array(observations)).to(self.device)
            
            # For tensors that are already torch tensors, we need to detach them
            beliefs_list = [b.detach() for b in beliefs]
            latents_list = [l.detach() for l in latents]
            next_beliefs_list = [nb.detach() for nb in next_beliefs]
            next_latents_list = [nl.detach() for nl in next_latents]
            
            # Handle means and logvars which might be None for older entries
            means_list = []
            logvars_list = []
            for m, lv in zip(means, logvars):
                if m is not None and lv is not None:
                    means_list.append(m.detach())
                logvars_list.append(lv.detach() if lv is not None else None)
            
            beliefs = torch.cat(beliefs_list).to(self.device)
            latents = torch.cat(latents_list).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            
            # Handle neighbor_actions which might be None or a dictionary
            if all(na is None for na in neighbor_actions):
                # If all neighbor_actions are None, create a tensor of zeros
                neighbor_actions_tensor = torch.zeros(len(actions), dtype=torch.long).to(self.device)
            elif any(isinstance(na, dict) for na in neighbor_actions):
                # If any neighbor_actions are dictionaries, create a tensor of zeros
                # This is a simplification - in a full implementation, we would need to encode all neighbor actions
                neighbor_actions_tensor = torch.zeros(len(actions), dtype=torch.long).to(self.device)
            else:
                # Replace None values with 0 (or another default value)
                neighbor_actions_list = [0 if na is None else na for na in neighbor_actions]
                neighbor_actions_tensor = torch.LongTensor(neighbor_actions_list).to(self.device)
                
            # Use the tensor for further processing
            neighbor_actions = neighbor_actions_tensor
                
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_observations = torch.FloatTensor(np.array(next_observations)).to(self.device)
            next_beliefs = torch.cat(next_beliefs_list).to(self.device)
            next_latents = torch.cat(next_latents_list).to(self.device)
            
            # Only create means and logvars tensors if we have data
            means = torch.cat(means_list).to(self.device) if means_list else None
            logvars = torch.cat(logvars_list).to(self.device) if logvars_list else None
            
            time_step_data = (observations, beliefs, latents, actions, neighbor_actions, rewards, 
                            next_observations, next_beliefs, next_latents, means, logvars)
            batch_data.append(time_step_data)
        
        return batch_data
    
    def __len__(self):
        # Count total transitions in the buffer
        return len(self.buffer)

