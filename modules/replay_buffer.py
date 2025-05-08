from modules.utils import get_best_device
from collections import deque, defaultdict
import numpy as np
import torch
import random

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
        
    def push(self, signal, neighbor_actions, belief, latent, action, reward, 
             next_signal, next_belief, next_latent, mean=None, logvar=None):
        """Save a transition to the buffer."""
        transition = (signal, neighbor_actions, belief, latent, action, reward, 
                     next_signal, next_belief, next_latent, mean, logvar)
        self.buffer.append(transition)
    
    def end_trajectory(self):
        """Backward compatibility method - does nothing in the continuous version."""
        pass
    
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


class EpisodeAwareReplayBuffer(ReplayBuffer):
    """
    Enhanced replay buffer that tracks episode boundaries and maintains memory of previous episodes.
    
    This buffer is designed to prevent catastrophic forgetting by retaining and sampling
    from transitions across multiple episodes, helping agents maintain state diversity
    and prevent overfitting to the current state.
    """
    def __init__(self, capacity, observation_dim, belief_dim, latent_dim, 
                 device=None, sequence_length=8, 
                 episodes_to_retain=4, samples_per_episode=None):
        super().__init__(capacity, observation_dim, belief_dim, latent_dim, device, sequence_length)
        
        # Replace the standard buffer with episode-specific buffers
        self.buffer = None  # Override the parent's buffer
        
        # Store transitions by episode
        self.episode_buffers = []
        self.current_episode_buffer = deque(maxlen=capacity)
        
        # Configuration for episode retention
        self.episodes_to_retain = episodes_to_retain  # Number of episodes to keep in memory
        
        # If samples_per_episode is not specified, use capacity/episodes_to_retain
        self.samples_per_episode = samples_per_episode or (capacity // episodes_to_retain)
        
        # Track episode statistics
        self.current_episode = 0
        self.true_states_per_episode = {}
    
    def push(self, signal, neighbor_actions, belief, latent, action, reward, 
             next_signal, next_belief, next_latent, mean=None, logvar=None):
        """Save a transition to the current episode buffer."""
        transition = (signal, neighbor_actions, belief, latent, action, reward, 
                     next_signal, next_belief, next_latent, mean, logvar)
        self.current_episode_buffer.append(transition)
    
    def end_episode(self, true_state=None):
        """Mark the end of the current episode and prepare for a new one."""
        # If we have transitions in the current episode, store them
        if len(self.current_episode_buffer) > 0:
            # Store true state for this episode if provided
            if true_state is not None:
                self.true_states_per_episode[self.current_episode] = true_state
                
            # Convert to list to ensure we have a copy
            self.episode_buffers.append(list(self.current_episode_buffer))
            self.current_episode_buffer.clear()
            
            # Limit the number of stored episodes
            if len(self.episode_buffers) > self.episodes_to_retain:
                # Remove the oldest episode
                self.episode_buffers.pop(0)
                
            # Increment episode counter
            self.current_episode += 1
    
    def __len__(self):
        """Return the total number of transitions across all stored episodes and current episode."""
        episode_lengths = sum(len(buffer) for buffer in self.episode_buffers)
        return episode_lengths + len(self.current_episode_buffer)
    
    def sample(self, batch_size, sequence_length=None, mode="single"):
        """Sample transitions across episodes to maintain diversity."""
        if mode == "sequence":
            return self._sample_sequences(batch_size, sequence_length)
        else:
            return self._sample_transitions(batch_size)
    
    def _sample_transitions(self, batch_size):
        """Sample individual transitions while ensuring episode diversity."""
        # Ensure we have enough transitions total
        total_transitions = len(self)
        if total_transitions < batch_size:
            return None
        
        # Collect all available buffers including the current one
        all_buffers = self.episode_buffers + [self.current_episode_buffer] if len(self.current_episode_buffer) > 0 else self.episode_buffers
        
        if not all_buffers:
            return None
            
        # Calculate how many samples to take from each episode
        num_episodes = len(all_buffers)
        
        # At least one sample from each episode, distributed evenly
        samples_per_buffer = [max(1, batch_size // num_episodes)] * num_episodes
        
        # Distribute any remaining samples
        remaining = batch_size - sum(samples_per_buffer)
        for i in range(remaining):
            samples_per_buffer[i % num_episodes] += 1
        
        # Sample transitions from each buffer
        transitions = []
        for i, buffer in enumerate(all_buffers):
            # Adjust if buffer has fewer transitions than requested
            num_samples = min(samples_per_buffer[i], len(buffer))
            if num_samples > 0:
                indices = np.random.choice(len(buffer), num_samples, replace=False)
                transitions.extend([buffer[idx] for idx in indices])
        
        # If we still need more transitions, sample more from available buffers
        remaining = batch_size - len(transitions)
        if remaining > 0:
            # Instead of checking if transitions are already in the list (which doesn't work with tensors),
            # just sample from the largest buffers with replacement
            buffer_sizes = [len(buffer) for buffer in all_buffers]
            if sum(buffer_sizes) > 0:  # Make sure we have buffers with data
                # Sample from buffers proportionally to their size
                probs = np.array(buffer_sizes) / sum(buffer_sizes)
                buffer_indices = np.random.choice(len(all_buffers), remaining, p=probs)
                
                for buffer_idx in buffer_indices:
                    buffer = all_buffers[buffer_idx]
                    if len(buffer) > 0:
                        idx = np.random.randint(0, len(buffer))
                        transitions.append(buffer[idx])
        
        # Process the transitions
        return self._process_transitions(transitions)
    
    def _sample_sequences(self, batch_size, sequence_length=None):
        """Sample sequences of transitions while ensuring episode diversity."""
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        # Collect all episode buffers including the current one if it has enough transitions
        valid_buffers = []
        for buffer in self.episode_buffers:
            if len(buffer) >= sequence_length:
                valid_buffers.append(buffer)
                
        # Add current episode buffer if valid
        if len(self.current_episode_buffer) >= sequence_length:
            valid_buffers.append(list(self.current_episode_buffer))
        
        # If not enough valid buffers, return None
        if not valid_buffers:
            return None
        
        # Calculate samples per buffer
        num_buffers = len(valid_buffers)
        samples_per_buffer = [batch_size // num_buffers] * num_buffers
        
        # Distribute any remaining samples
        remaining = batch_size - sum(samples_per_buffer)
        for i in range(remaining):
            samples_per_buffer[i % num_buffers] += 1
        
        # Sample sequences from each buffer
        sequences = []
        for i, buffer in enumerate(valid_buffers):
            num_samples = samples_per_buffer[i]
            if num_samples > 0:
                max_start = len(buffer) - sequence_length + 1
                start_indices = np.random.randint(0, max_start, size=num_samples)
                
                for start_idx in start_indices:
                    sequence = [buffer[start_idx + j] for j in range(sequence_length)]
                    sequences.append(sequence)
        
        # Process the sequences
        return self._process_sequence_batch(sequences)
    
    def get_episode_counts(self):
        """Return the number of transitions stored per episode."""
        counts = [len(buffer) for buffer in self.episode_buffers]
        if len(self.current_episode_buffer) > 0:
            counts.append(len(self.current_episode_buffer))
        return counts
    
    def get_true_states(self):
        """Return the true states for each episode."""
        return self.true_states_per_episode