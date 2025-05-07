import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.utils import get_best_device

class GRUBeliefProcessor(nn.Module):
    """GRU-based belief state processor for FURTHER+."""
    def __init__(self, hidden_dim, action_dim, device=None, num_belief_states=None):
        # Use the best available device if none is specified

        super(GRUBeliefProcessor, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.input_dim = action_dim* num_belief_states + num_belief_states
        self.action_dim = action_dim
        self.num_belief_states = num_belief_states
        
        # GRU for processing observation history
        self.gru = nn.GRU(self.input_dim, hidden_dim, batch_first=True)
        
        # Initialize parameters
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # Add softmax head for belief distribution
        self.belief_head = nn.Linear(hidden_dim, num_belief_states)
        nn.init.xavier_normal_(self.belief_head.weight)
        nn.init.constant_(self.belief_head.bias, 0)

    
    def forward(self, signal, neighbor_actions, current_belief=None):
        """Update belief state based on new observation and action."""
        # If no previous belief, initialize with zeros
        if current_belief is None:
            current_belief = torch.zeros(1, 1, self.hidden_dim, device=self.device)

        combined = torch.cat([
            signal,
            neighbor_actions,
        ], dim=0).unsqueeze(0)  # Add batch dimension
        
        # Process through GRU
        _, new_belief = self.gru(combined, current_belief)
        
        # Calculate belief distribution 
        # Pass through linear layer and apply softmax
        logits = self.belief_head(new_belief.squeeze(0))
        temperature = 0.5  # Temperature for softmax
        belief_distribution = F.softmax(logits/temperature, dim=-1)
        return new_belief, belief_distribution
    
class EncoderNetwork(nn.Module):
    """Encoder network for inference of other agents' policies."""
    def __init__(self, action_dim, latent_dim, hidden_dim, num_agents, device=None, num_belief_states=None):
        # Use the best available device if none is specified

        super(EncoderNetwork, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.num_belief_states = num_belief_states
        
        # Combined input: observation, actions, reward, next_obs and current latent
        input_dim = num_belief_states + action_dim * num_agents + 1 + num_belief_states + latent_dim
        
        # Encoder network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize parameters
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc_mean.weight)
        nn.init.xavier_normal_(self.fc_logvar.weight)
        
        # Add softmax head for opponent belief distribution 
        self.opponent_belief_head = nn.Linear(hidden_dim, num_belief_states)
        nn.init.xavier_normal_(self.opponent_belief_head.weight)
        nn.init.constant_(self.opponent_belief_head.bias, 0)

    
    def forward(self, signal, actions, reward, next_signal, current_latent):
        """Encode the state into a latent distribution."""

        current_latent = current_latent.squeeze(0) 
        # Ensure current_latent has the correct shape
        # Combine inputs
        combined = torch.cat([
            signal,
            actions,
            reward,
            next_signal,
            current_latent
        ], dim=0)
        
        # Forward pass
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        
        # Calculate opponent belief distribution
        logits = self.opponent_belief_head(x)
        temperature = 0.5  # Temperature for softmax
        opponent_belief_distribution = F.softmax(logits/temperature, dim=-1)
        
        return mean, logvar, opponent_belief_distribution

class DecoderNetwork(nn.Module):
    """Decoder network for predicting other agents' actions."""
    def __init__(self, action_dim, latent_dim, hidden_dim, num_agents, num_belief_states, device=None):
        # Use the best available device if none is specified
        if device is None:
            device = get_best_device()
        super(DecoderNetwork, self).__init__()
        self.device = device
        self.action_dim = action_dim
        
        # Combined input: observation and latent
        input_dim = num_belief_states + latent_dim
        
        # Decoder network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim*num_agents)
        
        # Initialize parameters
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
    
    def forward(self, signal, latent):
        """Predict peer actions from observation and latent."""
        # Combine inputs
        combined = torch.cat([signal, latent], dim=0)
        
        # Forward pass
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        
        return action_logits

class PolicyNetwork(nn.Module):
    """Policy network for deciding actions."""
    def __init__(self, belief_dim, latent_dim, action_dim, hidden_dim, device=None):
        # Use the best available device if none is specified
        if device is None:
            device = get_best_device()
        super(PolicyNetwork, self).__init__()
        self.device = device
        self.action_dim = action_dim
        
        # Combined input: belief and latent
        input_dim = belief_dim + latent_dim
        
        # Policy network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize parameters
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
    
    def forward(self, belief, latent):
        """Compute action logits given belief and latent."""
        # Combine inputs

        combined = torch.cat([belief.squeeze(0), latent.squeeze(0)], dim=0)
        # Forward pass
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        
        return action_logits

class QNetwork(nn.Module):
    """Q-function network for evaluating state-action values."""
    def __init__(self, belief_dim, latent_dim, action_dim, hidden_dim, num_agents=10, device=None):
        # Use the best available device if none is specified
        if device is None:
            device = get_best_device()
        super(QNetwork, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.num_agents = num_agents
        
        # Combined input: belief, latent, and neighbor actions (one-hot encoded for all neighbors)
        # We use action_dim * num_agents to represent all possible neighbor actions
        input_dim = belief_dim + latent_dim + action_dim * num_agents
        
        # Q-network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize parameters
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
    
    def forward(self, belief, latent, neighbor_actions=None):
        """Compute Q-values given belief, latent, and neighbor actions."""
        batch_size = belief.size(0)
        
        # Create a tensor to hold all neighbor actions (one-hot encoded)
        neighbor_actions_one_hot = torch.zeros(batch_size, self.num_agents * self.action_dim, device=self.device)
        
        # If neighbor_actions is provided and not None, encode them
        if neighbor_actions is not None:
            # If neighbor_actions is a dictionary (for a single batch item)
            if isinstance(neighbor_actions, dict):
                # Process each neighbor's action for the first batch item
                for neighbor_id, action in neighbor_actions.items():
                    if isinstance(neighbor_id, int) and 0 <= neighbor_id < self.num_agents:
                        # Calculate the index for this neighbor's action in the one-hot encoding
                        start_idx = neighbor_id * self.action_dim
                        # Set the corresponding bit to 1
                        if isinstance(action, int) and 0 <= action < self.action_dim:
                            neighbor_actions_one_hot[0, start_idx + action] = 1.0
            
            # If neighbor_actions is a list of dictionaries (for multiple batch items)
            elif isinstance(neighbor_actions, list) and all(isinstance(na, dict) for na in neighbor_actions):
                for b, na_dict in enumerate(neighbor_actions):
                    if b < batch_size:  # Ensure we don't exceed batch size
                        for neighbor_id, action in na_dict.items():
                            if isinstance(neighbor_id, int) and 0 <= neighbor_id < self.num_agents:
                                start_idx = neighbor_id * self.action_dim
                                if isinstance(action, int) and 0 <= action < self.action_dim:
                                    neighbor_actions_one_hot[b, start_idx + action] = 1.0
            
            # If neighbor_actions is a tensor
            elif isinstance(neighbor_actions, torch.Tensor):
                if neighbor_actions.dim() == 1:
                    # Single batch, single action per neighbor
                    for i, action in enumerate(neighbor_actions):
                        if i < self.num_agents:
                            action_idx = action.item()
                            if 0 <= action_idx < self.action_dim:
                                neighbor_actions_one_hot[0, i * self.action_dim + action_idx] = 1.0
                elif neighbor_actions.dim() == 2:
                    # Multiple batches, single action per neighbor
                    for b in range(min(batch_size, neighbor_actions.size(0))):
                        for i, action in enumerate(neighbor_actions[b]):
                            if i < self.num_agents:
                                action_idx = action.item()
                                if 0 <= action_idx < self.action_dim:
                                    neighbor_actions_one_hot[b, i * self.action_dim + action_idx] = 1.0
        
        # Combine inputs
        combined = torch.cat([belief, latent, neighbor_actions_one_hot], dim=1)
        
        # Forward pass
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        
        return q_values
