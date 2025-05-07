from modules.utils import get_best_device
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    """Enhanced replay buffer supporting both sequence sampling and temporal processing."""
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
        """Backward compatibility method - does nothing in the continuous version."""
        pass
    
    def sample(self, batch_size, sequence_length=None, mode="sequence"):
        """Enhanced sampling with multiple modes:
        - sequence: Sample sequences for GRU training
        - random: Sample random transitions (FURTHER-style)
        - all: Return all transitions in order (for temporal KL)
        """
        # For sequence-based sampling (GRU training)
        if mode == "sequence":
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
            
            # Process and return sequence batch
            return self._process_sequence_batch(sequences)
        
        # For FURTHER-style random sampling
        elif mode == "random":
            indices = np.random.randint(0, len(self), size=min(batch_size, len(self)))
            return self._process_transitions([self.buffer[i] for i in indices])
            
        # For temporal KL calculation (all transitions in order)
        elif mode == "all":
            # If requesting all but buffer is too small, return what we have
            return self._process_transitions(list(self.buffer))
            
        else:
            raise ValueError(f"Invalid sampling mode: {mode}")
    
    def _process_transitions(self, transitions):
        """Process a list of transitions into batched tensors."""
        if not transitions:
            return None
            
        # Unpack transitions
        observations, beliefs, latents, actions, neighbor_actions, rewards, \
        next_observations, next_beliefs, next_latents, means, logvars = zip(*transitions)
        
        # Convert to tensors
        # Handle both numpy arrays and torch tensors
        observations_list = []
        for obs in observations:
            if isinstance(obs, np.ndarray):
                # Convert numpy array to tensor
                obs_tensor = torch.FloatTensor(obs)
                if obs.ndim == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                observations_list.append(obs_tensor)
            else:
                # It's already a tensor
                if obs.dim() == 1:
                    observations_list.append(obs.unsqueeze(0))
                else:
                    observations_list.append(obs)
        
        observations = torch.cat(observations_list)
        
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
                logvars_list.append(lv.detach())
        
        beliefs = torch.cat(beliefs_list).to(self.device)
        latents = torch.cat(latents_list).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        
        # Handle neighbor_actions which might be None or a dictionary
        if all(na is None for na in neighbor_actions):
            # If all neighbor_actions are None, create a tensor of zeros
            neighbor_actions_tensor = torch.zeros(len(actions), dtype=torch.long).to(self.device)
        elif any(isinstance(na, dict) for na in neighbor_actions):
            # If any neighbor_actions are dictionaries, process them accordingly
            neighbor_actions_tensor = self._process_neighbor_actions_dict(neighbor_actions)
        else:
            # Replace None values with 0 (or another default value)
            neighbor_actions_list = [0 if na is None else na for na in neighbor_actions]
            neighbor_actions_tensor = torch.LongTensor(neighbor_actions_list).to(self.device)
                
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        
        # Handle next_observations the same way as observations
        next_observations_list = []
        for obs in next_observations:
            if isinstance(obs, np.ndarray):
                # Convert numpy array to tensor
                obs_tensor = torch.FloatTensor(obs)
                if obs.ndim == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                next_observations_list.append(obs_tensor)
            else:
                # It's already a tensor
                if obs.dim() == 1:
                    next_observations_list.append(obs.unsqueeze(0))
                else:
                    next_observations_list.append(obs)
        
        next_observations = torch.cat(next_observations_list)
        next_beliefs = torch.cat(next_beliefs_list).to(self.device)
        next_latents = torch.cat(next_latents_list).to(self.device)
        
        # Only create means and logvars tensors if we have data
        means = torch.cat(means_list).to(self.device) if means_list else None
        logvars = torch.cat(logvars_list).to(self.device) if logvars_list else None
        
        return (observations, beliefs, latents, actions, neighbor_actions_tensor, rewards,
                next_observations, next_beliefs, next_latents, means, logvars)
    
    def _process_sequence_batch(self, sequences):
        """Process a batch of sequences for GRU training."""
        batch_data = []
        sequence_length = len(sequences[0])
        
        for t in range(sequence_length):
            # Get all transitions at time step t across all sequences
            time_step_transitions = [seq[t] for seq in sequences]
            
            # Process these transitions into batched tensors
            time_step_data = self._process_transitions(time_step_transitions)
            batch_data.append(time_step_data)
        
        return batch_data
    
    def _process_neighbor_actions_dict(self, neighbor_actions_dicts):
        """Process neighbor_actions when they are dictionaries."""
        batch_size = len(neighbor_actions_dicts)
        # Create a tensor to hold the result
        result = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        
        # For each sample in the batch
        for i, na_dict in enumerate(neighbor_actions_dicts):
            # If it's a valid dictionary with values, use the first value
            if isinstance(na_dict, dict) and len(na_dict) > 0:
                # Get the first agent's action
                first_agent = next(iter(na_dict))
                result[i] = na_dict[first_agent]
        
        return result

    def get_sequential_latent_params(self):
        """Get all means and logvars in chronological order for temporal KL calculation."""
        if len(self.buffer) < 2:
            return None, None
            
        transitions = list(self.buffer)
        means = [t[9] for t in transitions if t[9] is not None]  # Means at index 9
        logvars = [t[10] for t in transitions if t[10] is not None]  # Logvars at index 10
        
        if not means or not logvars or len(means) < 2 or len(logvars) < 2:
            return None, None
            
        # Convert to tensors
        means_tensor = torch.cat([m.unsqueeze(0) if m.dim() == 1 else m for m in means])
        logvars_tensor = torch.cat([lv.unsqueeze(0) if lv.dim() == 1 else lv for lv in logvars])
        
        return means_tensor, logvars_tensor
    
    def __len__(self):
        return len(self.buffer)