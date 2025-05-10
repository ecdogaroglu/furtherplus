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
        elif mode == "all":
            # Sample all available transitions (up to batch_size)
            if len(self.buffer) == 0:
                return None
            
            # Take all transitions (or up to batch_size if specified)
            num_samples = min(len(self.buffer), batch_size) if batch_size > 0 else len(self.buffer)
            # Use sequential sampling if few samples available
            if num_samples <= 5:  # For very small buffers, take them in order
                indices = range(num_samples)
            else:
                indices = np.random.choice(len(self.buffer), num_samples, replace=False)
            
            transitions = [self.buffer[i] for i in indices]
            return self._process_transitions(transitions)
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
        signals_list = []
        for s in signals:
            if isinstance(s, torch.Tensor):
                signals_list.append(s.detach())
            elif isinstance(s, (int, float)):
                signals_list.append(torch.tensor([s], dtype=torch.float32))
            else:
                signals_list.append(s)
        
        neighbor_actions_list = []
        for na in neighbor_actions:
            if isinstance(na, torch.Tensor):
                neighbor_actions_list.append(na.detach())
            elif isinstance(na, (int, float)):
                neighbor_actions_list.append(torch.tensor([na], dtype=torch.float32))
            else:
                neighbor_actions_list.append(na)
        
        # Process belief states to ensure consistent shape [1, batch_size, hidden_dim]
        beliefs_list = []
        next_beliefs_list = []
        for b, nb in zip(beliefs, next_beliefs):
            # Handle current belief
            if isinstance(b, torch.Tensor):
                b = self._standardize_belief_state(b.detach())
                beliefs_list.append(b)
            else:
                beliefs_list.append(torch.zeros(1, 1, self.belief_dim, device=self.device))
            
            # Handle next belief
            if isinstance(nb, torch.Tensor):
                nb = self._standardize_belief_state(nb.detach())
                next_beliefs_list.append(nb)
            else:
                next_beliefs_list.append(torch.zeros(1, 1, self.belief_dim, device=self.device))
        
        # Process latent states to ensure consistent shape [1, batch_size, latent_dim]
        latents_list = []
        next_latents_list = []
        for l, nl in zip(latents, next_latents):
            # Handle current latent
            if isinstance(l, torch.Tensor):
                if l.dim() == 1:  # [latent_dim]
                    l = l.unsqueeze(0)  # [1, latent_dim]
                if l.dim() == 2:  # [batch_size, latent_dim]
                    l = l.unsqueeze(0)  # [1, batch_size, latent_dim]
                latents_list.append(l.detach())
            else:
                # Create zero tensor if not a tensor
                latents_list.append(torch.zeros(1, 1, self.belief_dim, device=self.device))
            
            # Handle next latent
            if isinstance(nl, torch.Tensor):
                if nl.dim() == 1:  # [latent_dim]
                    nl = nl.unsqueeze(0)  # [1, latent_dim]
                if nl.dim() == 2:  # [batch_size, latent_dim]
                    nl = nl.unsqueeze(0)  # [1, batch_size, latent_dim]
                next_latents_list.append(nl.detach())
            else:
                # Create zero tensor if not a tensor
                next_latents_list.append(torch.zeros(1, 1, self.belief_dim, device=self.device))
        
        next_signals_list = []
        for ns in next_signals:
            if isinstance(ns, torch.Tensor):
                next_signals_list.append(ns.detach())
            elif isinstance(ns, (int, float)):
                next_signals_list.append(torch.tensor([ns], dtype=torch.float32))
            else:
                next_signals_list.append(ns)
        
        # Handle means and logvars which might be None or float for older entries
        means_list = []
        logvars_list = []
        for m, lv in zip(means, logvars):
            if m is not None and lv is not None:
                if isinstance(m, torch.Tensor) and isinstance(lv, torch.Tensor):
                    means_list.append(m.detach())
                    logvars_list.append(lv.detach())
                elif isinstance(m, (int, float)) and isinstance(lv, (int, float)):
                    means_list.append(torch.tensor([m], dtype=torch.float32))
                    logvars_list.append(torch.tensor([lv], dtype=torch.float32))
        
        try:
            # Stack tensors with consistent shapes
            signals = torch.stack(signals_list).to(self.device)
            neighbor_actions = torch.stack(neighbor_actions_list).to(self.device)
            beliefs = torch.cat([b.view(1, 1, -1) for b in beliefs_list], dim=1).to(self.device)  # Ensure consistent shape
            latents = torch.cat([l.view(1, 1, -1) for l in latents_list], dim=1).to(self.device)  # Ensure consistent shape
            
            # Convert actions to tensor if they're not already
            actions_tensor = []
            for a in actions:
                if isinstance(a, torch.Tensor):
                    actions_tensor.append(a.item())
                else:
                    actions_tensor.append(int(a))
            actions = torch.LongTensor(actions_tensor).to(self.device)
            
            # Convert rewards to tensor
            rewards_tensor = []
            for r in rewards:
                if isinstance(r, torch.Tensor):
                    rewards_tensor.append(r.item())
                else:
                    rewards_tensor.append(float(r))
            rewards = torch.FloatTensor(rewards_tensor).unsqueeze(1).to(self.device)
            
            next_signals = torch.stack(next_signals_list).to(self.device)
            next_beliefs = torch.cat([nb.view(1, 1, -1) for nb in next_beliefs_list], dim=1).to(self.device)  # Ensure consistent shape
            next_latents = torch.cat([nl.view(1, 1, -1) for nl in next_latents_list], dim=1).to(self.device)  # Ensure consistent shape
            
            # Only create means and logvars tensors if we have data
            means = torch.cat(means_list).to(self.device) if means_list else None
            logvars = torch.cat(logvars_list).to(self.device) if logvars_list else None
            
            return (signals, neighbor_actions, beliefs, latents, actions, rewards,
                    next_signals, next_beliefs, next_latents, means, logvars)
        except Exception as e:
            print(f"Error processing transitions: {e}")
            return None
    
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