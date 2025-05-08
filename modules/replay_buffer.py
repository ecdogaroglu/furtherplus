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
        self.belief_dim = belief_dim
        
        # Add episode tracking to help with multi-episode sampling
        self.episode_indices = []  # List of (start_idx, end_idx) tuples for each episode
        self.current_episode_start = 0  # Index tracking the start of the current episode
        
    def push(self, signal, neighbor_actions, belief, latent, action, reward, 
             next_signal, next_belief, next_latent, mean=None, logvar=None):
        """Save a transition to the buffer."""
        transition = (signal, neighbor_actions, belief, latent, action, reward, 
                     next_signal, next_belief, next_latent, mean, logvar)
        self.buffer.append(transition)
    
    def end_trajectory(self):
        """Mark the end of the current episode for episodic sampling."""
        # Store the episode boundary
        if len(self.buffer) > self.current_episode_start:
            episode_end = len(self.buffer) - 1
            # Only add this episode if it has enough transitions for a sequence
            if episode_end - self.current_episode_start + 1 >= self.sequence_length:
                self.episode_indices.append((self.current_episode_start, episode_end))
            # Start a new episode
            self.current_episode_start = len(self.buffer)
    
    def __len__(self):
        return len(self.buffer)
        
    def sample(self, batch_size, sequence_length=None, mode="single"):
        """Sample a batch of transitions or sequences from the buffer."""
        if mode == "sequence":
            # Sample sequences of transitions
            if sequence_length is None:
                sequence_length = self.sequence_length
                
            # Ensure we have enough transitions
            if len(self.buffer) < sequence_length:
                return None
                
            # Sample random starting points
            start_indices = np.random.randint(0, len(self.buffer) - sequence_length + 1, size=batch_size)
            
            # Extract sequences
            sequences = []
            for start_idx in start_indices:
                sequence = [self.buffer[i] for i in range(start_idx, start_idx + sequence_length)]
                sequences.append(sequence)
                
            return self._process_sequence_batch(sequences)
        else:
            # Sample individual transitions
            if len(self.buffer) < batch_size:
                return None
                
            # Sample indices instead of transitions directly
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            transitions = [self.buffer[i] for i in indices]
            return self._process_transitions(transitions)
    
    def sample_sequence_batch(self, batch_size, sequence_length=None):
        """Sample sequences for training, avoiding episode boundaries."""
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        # Ensure we have enough transitions
        if len(self.buffer) < sequence_length:
            return None
            
        # Sample random starting points that don't cross episode boundaries
        valid_start_indices = []
        
        # If we have episode boundaries recorded, use them
        if self.episode_indices:
            for start_idx, end_idx in self.episode_indices:
                # Only include episodes with enough transitions for a full sequence
                if end_idx - start_idx + 1 >= sequence_length:
                    # Add all valid starting indices from this episode
                    valid_start_indices.extend(range(start_idx, end_idx - sequence_length + 2))
            
            # Add the current episode if it has enough transitions
            current_length = len(self.buffer) - self.current_episode_start
            if current_length >= sequence_length:
                valid_start_indices.extend(range(self.current_episode_start, len(self.buffer) - sequence_length + 1))
        else:
            # If no episode boundaries, just use all possible starting indices
            valid_start_indices = list(range(0, len(self.buffer) - sequence_length + 1))
            
        # If we don't have enough valid starting points, return None
        if len(valid_start_indices) < batch_size:
            return self.sample(batch_size, sequence_length, mode="sequence")  # Fall back to standard sampling
            
        # Sample from valid starting points
        chosen_indices = np.random.choice(valid_start_indices, size=batch_size, replace=len(valid_start_indices) < batch_size)
        
        # Extract sequences
        sequences = []
        for start_idx in chosen_indices:
            sequence = [self.buffer[i] for i in range(start_idx, start_idx + sequence_length)]
            sequences.append(sequence)
            
        return self._process_sequence_batch(sequences)
    
    def sample_from_previous_episodes(self, batch_size, sequence_length=None):
        """Sample sequences specifically from previous episodes to promote state diversity."""
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        # Need at least one completed episode
        if len(self.episode_indices) < 1:
            return None
            
        # Only sample from past episodes, not the current one
        past_episode_indices = self.episode_indices[:-1] if len(self.episode_indices) > 1 else self.episode_indices
        
        # Collect valid starting indices from past episodes
        valid_start_indices = []
        for start_idx, end_idx in past_episode_indices:
            # Only include episodes with enough transitions for a full sequence
            if end_idx - start_idx + 1 >= sequence_length:
                # Add all valid starting indices from this episode
                valid_start_indices.extend(range(start_idx, end_idx - sequence_length + 2))
                
        # If no valid starting points from past episodes, return None
        if len(valid_start_indices) < batch_size:
            return None
            
        # Sample from valid starting points
        chosen_indices = np.random.choice(valid_start_indices, size=batch_size, replace=len(valid_start_indices) < batch_size)
        
        # Extract sequences
        sequences = []
        for start_idx in chosen_indices:
            sequence = [self.buffer[i] for i in range(start_idx, start_idx + sequence_length)]
            sequences.append(sequence)
            
        return self._process_sequence_batch(sequences)
    
    def _standardize_belief_state(self, belief):
        """Ensure belief state has consistent shape [1, batch_size, hidden_dim]."""
        if belief is None:
            return None
            
        # Add batch dimension if missing
        if belief.dim() == 1:  # [hidden_dim]
            belief = belief.unsqueeze(0)  # [1, hidden_dim]
            
        # Add sequence dimension if missing
        if belief.dim() == 2:  # [batch_size, hidden_dim]
            belief = belief.unsqueeze(0)  # [1, batch_size, hidden_dim]
            
        # Transpose if dimensions are in wrong order
        if belief.dim() == 3 and belief.size(0) != 1:
            belief = belief.transpose(0, 1).contiguous()
            
        return belief
    
    def _process_transitions(self, transitions):
        """Process a list of transitions into batched tensors."""
        if not transitions:
            return None
            
        # Unpack transitions
        signals, neighbor_actions, beliefs, latents, actions, rewards, \
        next_signals, next_beliefs, next_latents, means, logvars = zip(*transitions)
        
        # For tensors that are already torch tensors, we need to detach them
        signals_list = [s.detach() for s in signals]
        neighbor_actions_list = [na.detach() if isinstance(na, torch.Tensor) else na for na in neighbor_actions]
        
        # Process belief states to ensure consistent shape [1, batch_size, hidden_dim]
        beliefs_list = []
        next_beliefs_list = []
        for b, nb in zip(beliefs, next_beliefs):
            # Handle current belief
            b = self._standardize_belief_state(b.detach())
            beliefs_list.append(b)
            
            # Handle next belief
            nb = self._standardize_belief_state(nb.detach())
            next_beliefs_list.append(nb)
        
        # Process latent states to ensure consistent shape [1, batch_size, latent_dim]
        latents_list = []
        next_latents_list = []
        for l, nl in zip(latents, next_latents):
            # Handle current latent
            if l.dim() == 1:  # [latent_dim]
                l = l.unsqueeze(0)  # [1, latent_dim]
            if l.dim() == 2:  # [batch_size, latent_dim]
                l = l.unsqueeze(0)  # [1, batch_size, latent_dim]
            latents_list.append(l.detach())
            
            # Handle next latent
            if nl.dim() == 1:  # [latent_dim]
                nl = nl.unsqueeze(0)  # [1, latent_dim]
            if nl.dim() == 2:  # [batch_size, latent_dim]
                nl = nl.unsqueeze(0)  # [1, batch_size, latent_dim]
            next_latents_list.append(nl.detach())
        
        next_signals_list = [ns.detach() for ns in next_signals]
        
        # Handle means and logvars which might be None for older entries
        means_list = []
        logvars_list = []
        for m, lv in zip(means, logvars):
            if m is not None and lv is not None:
                means_list.append(m.detach())
                logvars_list.append(lv.detach())
        
        # Stack tensors with consistent shapes
        signals = torch.stack(signals_list).to(self.device)
        neighbor_actions = torch.stack(neighbor_actions_list).to(self.device)
        beliefs = torch.cat([b.view(1, 1, -1) for b in beliefs_list], dim=1).to(self.device)  # Ensure consistent shape
        latents = torch.cat([l.view(1, 1, -1) for l in latents_list], dim=1).to(self.device)  # Ensure consistent shape
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_signals = torch.stack(next_signals_list).to(self.device)
        next_beliefs = torch.cat([nb.view(1, 1, -1) for nb in next_beliefs_list], dim=1).to(self.device)  # Ensure consistent shape
        next_latents = torch.cat([nl.view(1, 1, -1) for nl in next_latents_list], dim=1).to(self.device)  # Ensure consistent shape
        
        # Only create means and logvars tensors if we have data
        means = torch.cat(means_list).to(self.device) if means_list else None
        logvars = torch.cat(logvars_list).to(self.device) if logvars_list else None
        
        return (signals, neighbor_actions, beliefs, latents, actions, rewards,
                next_signals, next_beliefs, next_latents, means, logvars)
    
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