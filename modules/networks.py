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
        self.input_dim = action_dim + num_belief_states  # Fixed input dimension calculation
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

    def standardize_belief_state(self, belief):
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
    
    def forward(self, signal, neighbor_actions, current_belief=None):
        """Update belief state based on new observation."""
        # Handle both batched and single inputs
        
        # Ensure we have batch dimension for both inputs
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        if neighbor_actions.dim() == 1:
            neighbor_actions = neighbor_actions.unsqueeze(0)
            
        batch_size = signal.size(0)
            
        # Ensure signal and neighbor_actions have correct dimensions
        if signal.size(1) != self.num_belief_states:
            signal = signal[:, :self.num_belief_states]
        if neighbor_actions.size(1) != self.action_dim:
            neighbor_actions = neighbor_actions[:, :self.action_dim]
            
        # Initialize or standardize current_belief
        if current_belief is None:
            current_belief = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
        else:
            current_belief = self.standardize_belief_state(current_belief)
            
        # Combine inputs along feature dimension
        combined = torch.cat([signal, neighbor_actions], dim=1)
        # Add sequence dimension (GRU expects [batch, seq_len, features])
        combined = combined.unsqueeze(1).to(self.device)  # [batch_size, 1, input_dim]
        
        # Process through GRU
        _, new_belief = self.gru(combined, current_belief)
        
        # Calculate belief distribution
        logits = self.belief_head(new_belief.squeeze(0))
        temperature = 0.5  # Temperature for softmax
        belief_distribution = F.softmax(logits/temperature, dim=-1)
        
        # Ensure new_belief maintains shape [1, batch_size, hidden_dim]
        new_belief = self.standardize_belief_state(new_belief)
        
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
        signal = signal.to(self.device)
        actions = actions.to(self.device)   
        reward = reward.to(self.device) 
        next_signal = next_signal.to(self.device)   
        current_latent = current_latent.to(self.device) 
        # Handle different dimensions
        if current_latent.dim() == 3:  # [batch_size, 1, latent_dim]
            current_latent = current_latent.squeeze(1)
            
        # Ensure all inputs have batch dimension and are 2D tensors
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
            
        # Handle reward which could be a scalar or tensor
        if isinstance(reward, (int, float)):
            reward = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        elif isinstance(reward, torch.Tensor):
            if reward.dim() == 0:
                reward = reward.unsqueeze(0).unsqueeze(0)
            elif reward.dim() == 1:
                reward = reward.unsqueeze(1)
                
        if next_signal.dim() == 1:
            next_signal = next_signal.unsqueeze(0)
            
        if current_latent.dim() == 1:
            current_latent = current_latent.unsqueeze(0)
            
        # Combine inputs along feature dimension
        combined = torch.cat([
            signal,
            actions,
            reward,
            next_signal,
            current_latent
        ], dim=1).to(self.device)  # [batch_size, input_dim]
        
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
        self.num_agents = num_agents
        self.num_belief_states = num_belief_states
        
        # Store dimensions for debugging
        self.observation_dim = num_belief_states
        self.latent_dim = latent_dim
        
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
        signal = signal.to(self.device)
        latent = latent.to(self.device) 
        
        """Predict peer actions from observation and latent."""
        # Handle different dimensions
        if latent.dim() == 3:  # [batch_size, 1, latent_dim]
            latent = latent.squeeze(0)
        
        # Create a new input tensor with the correct dimensions
        batch_size = signal.size(0) if signal.dim() > 1 else 1
        
        # Ensure latent has the right shape
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        
        # Combine inputs
        combined = torch.cat([signal, latent], dim=1)
        
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
        # Handle both batched and single inputs
        if belief.dim() == 3:  # [batch_size, 1, belief_dim]
            belief = belief.squeeze(0)
        if latent.dim() == 3:  # [batch_size, 1, latent_dim]
            latent = latent.squeeze(0)
            
        # For batched inputs
        if belief.dim() == 2 and latent.dim() == 2:
            # Combine inputs along feature dimension
            combined = torch.cat([belief, latent], dim=1)
        # For single inputs
        else:
            # Ensure we have batch dimension
            if belief.dim() == 1:
                belief = belief.unsqueeze(0)
            if latent.dim() == 1:
                latent = latent.unsqueeze(0)
            combined = torch.cat([belief, latent], dim=1)
            
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
        # Handle different dimensions
        if belief.dim() == 3:  # [batch_size, 1, belief_dim]
            belief = belief.squeeze(0)
        if latent.dim() == 3:  # [batch_size, 1, latent_dim]
            latent = latent.squeeze(0)

        # Combine inputs
        combined = torch.cat([belief, latent, neighbor_actions], dim=1)
        
        # Forward pass
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        
        return q_values
