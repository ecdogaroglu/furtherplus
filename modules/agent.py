import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple

def get_best_device():
    """Get the best available device (CUDA, MPS, or CPU).
    
    For MPS (Apple Silicon), we perform a quick benchmark to ensure it's actually
    faster than CPU, as there are known issues with MPS in some PyTorch versions.
    
    The device can be forced by setting the TORCH_DEVICE environment variable.
    """
    import os
    
    # Check if device is forced via environment variable
    forced_device = os.environ.get('TORCH_DEVICE')
    if forced_device:
        if forced_device.lower() in ['cuda', 'mps', 'cpu']:
            print(f"Using device '{forced_device}' from TORCH_DEVICE environment variable")
            return forced_device.lower()
        else:
            print(f"Warning: Invalid TORCH_DEVICE value '{forced_device}'. Must be 'cuda', 'mps', or 'cpu'.")
    
    # Auto-select the best device
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Check if MPS is actually faster than CPU with a quick benchmark
        if is_mps_faster_than_cpu():
            return 'mps'
        else:
            print("MPS available but benchmark shows it's slower than CPU. Using CPU instead.")
            return 'cpu'
    else:
        return 'cpu'

def is_mps_faster_than_cpu(test_size=256, repeat=10):
    """Run a benchmark to check if MPS is faster than CPU for neural network operations.
    
    Args:
        test_size: Size of test matrices/tensors
        repeat: Number of times to repeat the test
        
    Returns:
        bool: True if MPS is faster, False otherwise
    """
    import time
    
    # Skip benchmark if MPS is not available
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        return False
    
    # Create a small neural network for testing
    class TestNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(test_size, test_size)
            self.fc2 = nn.Linear(test_size, test_size)
            self.gru = nn.GRU(test_size, test_size//2, batch_first=True)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = x.unsqueeze(1)  # Add sequence dimension for GRU
            x, _ = self.gru(x)
            return x
    
    # Create input data
    batch_size = 32
    input_data = torch.randn(batch_size, test_size)
    
    # Test on CPU
    try:
        cpu_model = TestNetwork().to('cpu')
        
        # Warm-up
        for _ in range(2):
            _ = cpu_model(input_data)
        
        # Benchmark
        cpu_start = time.time()
        for _ in range(repeat):
            _ = cpu_model(input_data)
        cpu_time = time.time() - cpu_start
        
        # Test on MPS
        mps_model = TestNetwork().to('mps')
        mps_input = input_data.to('mps')
        
        # Warm-up
        for _ in range(2):
            _ = mps_model(mps_input)
            torch.mps.synchronize()
        
        # Benchmark
        mps_start = time.time()
        for _ in range(repeat):
            _ = mps_model(mps_input)
            torch.mps.synchronize()
        mps_time = time.time() - mps_start
        
        # Compare times (with a small margin to prefer MPS if it's close)
        print(f"Neural network benchmark - CPU: {cpu_time:.4f}s, MPS: {mps_time:.4f}s")
        
        # Only use MPS if it's significantly faster (at least 20% faster)
        return mps_time < cpu_time * 0.8
        
    except Exception as e:
        print(f"Error during MPS benchmark: {e}")
        return False

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


class GRUBeliefProcessor(nn.Module):
    """GRU-based belief state processor for FURTHER+."""
    def __init__(self, input_dim, hidden_dim, action_dim, device=None, num_belief_states=None):
        # Use the best available device if none is specified
        if device is None:
            device = get_best_device()
        super(GRUBeliefProcessor, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.num_belief_states = num_belief_states
        
        # GRU for processing observation history
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # Initialize parameters
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # Add softmax head for belief distribution if num_belief_states is provided
        if num_belief_states is not None:
            self.belief_head = nn.Linear(hidden_dim, num_belief_states)
            nn.init.xavier_normal_(self.belief_head.weight)
            nn.init.constant_(self.belief_head.bias, 0)
        else:
            self.belief_head = None
    
    def forward(self, observation, current_belief=None):
        """Update belief state based on new observation and action."""
        # If no previous belief, initialize with zeros
        if current_belief is None:
            current_belief = torch.zeros(1, 1, self.hidden_dim, device=self.device)
        
        # Ensure observation has batch dimension
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        
        # Add sequence dimension
        observation = observation.unsqueeze(1)
        
        # Ensure current_belief has correct dimensions [num_layers=1, batch_size, hidden_dim]
        if current_belief.dim() == 2:
            current_belief = current_belief.unsqueeze(0)
        
        # Process through GRU
        _, new_belief = self.gru(observation, current_belief)
        
        # Calculate belief distribution if softmax head is available
        belief_distribution = None
        if self.belief_head is not None:
            # Extract the belief state (remove the first dimension which is num_layers)
            belief_for_head = new_belief.squeeze(0)
            # Pass through linear layer and apply softmax
            logits = self.belief_head(belief_for_head)
            temperature = 0.1  # Temperature for softmax
            belief_distribution = F.softmax(logits/temperature, dim=-1)
        
        return new_belief, belief_distribution
    

class EncoderNetwork(nn.Module):
    """Encoder network for inference of other agents' policies."""
    def __init__(self, observation_dim, action_dim, latent_dim, hidden_dim, num_agents, device=None):
        # Use the best available device if none is specified
        if device is None:
            device = get_best_device()
        super(EncoderNetwork, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.num_agents = num_agents
        
        # Combined input: observation, action, reward, next_obs and current latent
        input_dim = observation_dim + action_dim * num_agents + 1 + observation_dim + latent_dim
        
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
    
    def forward(self, observation, actions, reward, next_observation, current_latent):
        """Encode the state into a latent distribution."""
        # Handle actions tensor which could be 2D [batch, num_agents]
        batch_size = observation.size(0)
        
        # For compatibility with the network initialization, we need to ensure
        # actions_one_hot has shape [batch, num_agents*action_dim]
        if isinstance(actions, dict):
            # Convert dictionary to tensor
            actions_list = [actions.get(i, 0) for i in range(self.num_agents)]
            actions = torch.tensor([actions_list], dtype=torch.long).to(self.device)
        
        # Ensure actions has the right shape
        if actions.dim() == 1:
            # Single action, expand to [batch, 1]
            actions = actions.unsqueeze(0) if batch_size == 1 else actions.unsqueeze(1)
        
        # Create a fixed-size one-hot tensor for all agents
        # If we have fewer agents than expected, pad with zeros
        actions_one_hot = torch.zeros(batch_size, self.num_agents * self.action_dim, device=self.device)
        
        # Fill in the one-hot encodings for the available agents
        for i in range(min(actions.size(1), self.num_agents)):
            agent_actions = actions[:, i]
            for b in range(batch_size):
                action_idx = agent_actions[b].item()
                if 0 <= action_idx < self.action_dim:
                    # Set the corresponding one-hot bit
                    one_hot_idx = i * self.action_dim + action_idx
                    actions_one_hot[b, one_hot_idx] = 1.0
        
        # Combine inputs
        combined = torch.cat([
            observation,
            actions_one_hot,
            reward,
            next_observation,
            current_latent
        ], dim=1)
        
        # Forward pass
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        
        return mean, logvar


class DecoderNetwork(nn.Module):
    """Decoder network for predicting other agents' actions."""
    def __init__(self, observation_dim, action_dim, latent_dim, hidden_dim, num_agents, device=None):
        # Use the best available device if none is specified
        if device is None:
            device = get_best_device()
        super(DecoderNetwork, self).__init__()
        self.device = device
        self.action_dim = action_dim
        
        # Combined input: observation and latent
        input_dim = observation_dim + latent_dim
        
        # Decoder network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim*num_agents)
        
        # Initialize parameters
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
    
    def forward(self, observation, latent):
        """Predict peer actions from observation and latent."""
        # Combine inputs
        combined = torch.cat([observation, latent], dim=1)
        
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


class FURTHERPlusAgent:
    """FURTHER+ agent for social learning."""
    def __init__(
        self,
        agent_id,
        num_agents,
        observation_dim,
        action_dim,
        hidden_dim=64,
        belief_dim=64,
        latent_dim=16,
        learning_rate=1e-3,
        discount_factor=0.99,
        entropy_weight=0.01,
        kl_weight=0.01,
        target_update_rate=0.005,
        device=None,
        buffer_capacity=1000,
        max_trajectory_length=50
    ):
        # Use the best available device if none is specified
        if device is None:
            device = get_best_device()
        print(f"Using device: {device}")
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = device
        self.discount_factor = discount_factor
        self.entropy_weight = entropy_weight
        self.kl_weight = kl_weight
        self.target_update_rate = target_update_rate
        self.max_trajectory_length = max_trajectory_length
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            observation_dim=observation_dim,
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            device=device,
            sequence_length=max_trajectory_length
        )
        
        # Initialize networks
        self.belief_processor = GRUBeliefProcessor(
            input_dim=observation_dim,
            hidden_dim=belief_dim,
            action_dim=action_dim,
            device=device,
            num_belief_states=num_agents  # Use num_agents as the number of belief states
        ).to(device)
        
        self.encoder = EncoderNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        
        self.decoder = DecoderNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        
        self.policy = PolicyNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device
        ).to(device)
        
        self.q_network1 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        
        self.q_network2 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        
        # Create target networks
        self.target_q_network1 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        
        self.target_q_network2 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        
        # Copy parameters to target networks
        self.target_q_network1.load_state_dict(self.q_network1.state_dict())
        self.target_q_network2.load_state_dict(self.q_network2.state_dict())
        
        # Average reward estimate (for average reward formulation)
        self.gain_parameter = nn.Parameter(torch.tensor(0.0, device=device))
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            list(self.policy.parameters()), 
            lr=learning_rate
        )
        self.q_optimizer = torch.optim.Adam(
            list(self.q_network1.parameters()) + 
            list(self.q_network2.parameters()),
            lr=learning_rate
        )
        self.belief_optimizer = torch.optim.Adam(
            list(self.belief_processor.parameters()),
            lr=learning_rate
        )
        self.encoder_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
        )
        self.gain_optimizer = torch.optim.Adam([self.gain_parameter], lr=learning_rate)
        
        # Initialize belief and latent states with small random values
        # This helps break symmetry and provides a more diverse starting point
        self.current_belief = torch.randn(1, belief_dim, device=device) * 0.01
        self.current_latent = torch.randn(1, latent_dim, device=device) * 0.01
        self.current_mean = torch.randn(1, latent_dim, device=device) * 0.01
        self.current_logvar = torch.randn(1, latent_dim, device=device) * 0.01
        
        # Initialize belief distribution
        if self.belief_processor.belief_head is not None:
            self.current_belief_distribution = torch.ones(1, num_agents, device=device) / num_agents
        else:
            self.current_belief_distribution = None
        
        # For tracking learning metrics
        self.action_probs_history = []
        
        # Episode tracking
        self.episode_step = 0
    
    def observe(self, observation):
        """Update belief state based on new observation."""
        # Check if this is the first observation of the episode
        is_first_obs = (self.episode_step == 0)
        self.episode_step += 1
        
        # Process the observation
        observation_tensor = torch.FloatTensor(observation).to(self.device)
        
        # Add batch dimension if needed
        if observation_tensor.dim() == 1:
            observation_tensor = observation_tensor.unsqueeze(0)
        
        # If this is the first observation of an episode, use None to force GRU to initialize fresh
        current_belief = None if is_first_obs else self.current_belief
        
        # Pass observation and belief to the belief processor
        new_belief, belief_distribution = self.belief_processor(
            observation_tensor, 
            current_belief
        )
        self.current_belief = new_belief.squeeze(0)
        
        # Store the belief distribution if available
        if belief_distribution is not None:
            self.current_belief_distribution = belief_distribution.squeeze(0)
        else:
            self.current_belief_distribution = None
        
        return self.current_belief, self.current_belief_distribution
        
    def store_transition(self, observation, belief, latent, action, reward, 
                         next_observation, next_belief, next_latent, mean=None, logvar=None, neighbor_actions=None):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(
            observation, belief, latent, action, reward,
            next_observation, next_belief, next_latent, mean, logvar, neighbor_actions
        )
        
    def reset_internal_state(self):
        """Reset the agent's internal state (belief and latent variables)."""
        # Use zeros instead of small random values for a complete reset
        self.current_belief = torch.zeros(1, self.belief_processor.hidden_dim, device=self.device)
        self.current_latent = torch.zeros(1, self.encoder.fc_mean.out_features, device=self.device)
        self.current_mean = torch.zeros(1, self.encoder.fc_mean.out_features, device=self.device)
        self.current_logvar = torch.zeros(1, self.encoder.fc_logvar.out_features, device=self.device)
        
        # Reset belief distribution if it exists
        if self.belief_processor.belief_head is not None:
            num_belief_states = self.belief_processor.num_belief_states
            self.current_belief_distribution = torch.ones(1, num_belief_states, device=self.device) / num_belief_states
        
        # Reset episode step counter to ensure next observation is treated as first in episode
        self.episode_step = 0
        
        # Detach all tensors to ensure no gradient flow between episodes
        self.current_belief = self.current_belief.detach()
        self.current_latent = self.current_latent.detach()
        self.current_mean = self.current_mean.detach()
        self.current_logvar = self.current_logvar.detach()
        if self.current_belief_distribution is not None:
            self.current_belief_distribution = self.current_belief_distribution.detach()
        
        # Also reset any cached states in the GRU
        for name, param in self.belief_processor.gru.named_parameters():
            if 'bias_hh' in name:  # Reset the bias related to hidden state
                nn.init.constant_(param, 0)
    
    def get_belief_state(self):
        """Return the current belief state.
        
        Returns:
            belief: Current belief state tensor
        """
        return self.current_belief
    
    def get_latent_state(self):
        """Return the current latent state.
        
        Returns:
            latent: Current latent state tensor
        """
        return self.current_latent
    
    def get_belief_distribution(self):
        """Return the current belief distribution.
        
        Returns:
            belief_distribution: Current belief distribution tensor or None if not available
        """
        return self.current_belief_distribution
    
    def get_latent_distribution_params(self):
        """Return the current latent distribution parameters (mean and logvar).
        
        Returns:
            mean: Current mean of the latent distribution
            logvar: Current log variance of the latent distribution
        """
        return self.current_mean, self.current_logvar
        
    def end_episode(self):
        """
        Backward compatibility method - does nothing in the continuous version.
        The internal state is maintained across what would have been episode boundaries.
        """
        pass
    
    def infer_latent(self, observation, actions, reward, next_observation):
        """Infer latent state of peer agent.
        
        Args:
            observation: Current observation
            actions: Dictionary of actions for each neighbor or a single action
            reward: Reward received
            next_observation: Next observation
            
        Returns:
            new_latent: The inferred latent state
        """
        # Convert to tensors
        observation_tensor = torch.FloatTensor(observation).to(self.device)
        if observation_tensor.dim() == 1:
            observation_tensor = observation_tensor.unsqueeze(0)
            
        next_observation_tensor = torch.FloatTensor(next_observation).to(self.device)
        if next_observation_tensor.dim() == 1:
            next_observation_tensor = next_observation_tensor.unsqueeze(0)
        
        # Handle actions - could be a dictionary or a single value
        if isinstance(actions, dict):
            # Convert dictionary of actions to a list in agent ID order
            actions_list = [actions.get(i, 0) for i in range(self.num_agents)]
            actions_tensor = torch.tensor([actions_list], dtype=torch.long).to(self.device)
        elif isinstance(actions, (list, tuple)):
            # List of actions
            actions_tensor = torch.tensor([actions], dtype=torch.long).to(self.device)
        else:
            # Single action value
            actions_tensor = torch.tensor([[actions]], dtype=torch.long).to(self.device)
            
        # Convert reward to tensor
        if isinstance(reward, (int, float)):
            reward_tensor = torch.tensor([[reward]], dtype=torch.float32).to(self.device)
        else:
            # Handle case where reward might be a numpy array or tensor
            reward_tensor = torch.FloatTensor([reward]).to(self.device)
            if reward_tensor.dim() == 1:
                reward_tensor = reward_tensor.unsqueeze(1)
        
        # Get latent distribution
        mean, logvar = self.encoder(
            observation_tensor,
            actions_tensor,
            reward_tensor,
            next_observation_tensor,
            self.current_latent
        )
        
        # Sample from distribution
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        new_latent = mean + eps * std
        
        # Store the current latent, mean, and logvar
        self.current_latent = new_latent
        self.current_mean = mean
        self.current_logvar = logvar
        
        return new_latent
    
    def select_action(self):
        """Select action based on current belief and latent."""
        # Get action logits
        action_logits = self.policy(self.current_belief, self.current_latent)
        
        # Convert to probabilities
        action_probs = F.softmax(action_logits, dim=1)
        
        # Store probability of incorrect action for learning rate calculation
        self.action_probs_history.append(action_probs.squeeze(0).detach().cpu().numpy())
        
        # Sample action from distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample().item()
        
        return action, action_probs.squeeze(0).detach().cpu().numpy()
    
    def update(self, batch_sequences):
        """Update all networks using sequential data.
        
        Args:
            batch_sequences: List of batches, where each batch contains data for one time step
                            across all sampled sequences.
        """
        if batch_sequences is None or len(batch_sequences) == 0:
            return
            
        # Initialize losses
        total_inference_loss = 0
        total_critic_loss = 0
        total_policy_loss = 0
        total_belief_loss = 0
        
        # Process each time step in the sequence
        for t, batch in enumerate(batch_sequences):
            # Unpack the batch, which now includes means and logvars
            if len(batch) == 11:  # New format with means and logvars
                (observations, beliefs, latents, actions, neighbor_actions, rewards, 
                next_observations, next_beliefs, next_latents, means, logvars) = batch
            else:  # Old format without means and logvars
                (observations, beliefs, latents, actions, neighbor_actions, rewards, 
                next_observations, next_beliefs, next_latents) = batch
                means, logvars = None, None
            
            # Update encoder-decoder (inference module)
            inference_result = self._update_inference(
                observations, beliefs, latents, actions, neighbor_actions, 
                rewards, next_observations, next_beliefs, next_latents,
                means, logvars
            )
            inference_loss = inference_result[0]  # First element is the loss value
            total_inference_loss += inference_loss
            
            # Update policy
            policy_result = self._update_policy(beliefs, latents, neighbor_actions)
            policy_loss = policy_result[0]  # First element is the loss value
            total_policy_loss += policy_loss
            
            # Update belief processor (GRU) using only policy components
            belief_loss = self._update_belief_processor(
                observations, beliefs, latents, actions, neighbor_actions, 
                rewards, next_observations, next_beliefs, next_latents,
                means, logvars,
                policy_components=policy_result,
                inference_components=None  # Not using inference components
            )
            total_belief_loss += belief_loss
            
            # Update Q-networks
            critic_loss = self._update_critics(
                observations, beliefs, latents, actions, neighbor_actions, 
                rewards, next_observations, next_beliefs, next_latents
            )
            total_critic_loss += critic_loss
        
        # Update target networks once per sequence
        self._update_targets()
        
        # Return average losses
        sequence_length = len(batch_sequences)
        return {
            'inference_loss': total_inference_loss / sequence_length,
            'critic_loss': total_critic_loss / sequence_length,
            'policy_loss': total_policy_loss / sequence_length,
            'belief_loss': total_belief_loss / sequence_length
        }
        
    def train(self, batch_size=32, sequence_length=8):
        """Train the agent using sequential data from the replay buffer.
        
        This method ensures belief states are processed sequentially for each trajectory,
        not just as independent samples.
        
        Args:
            batch_size: Number of sequences to sample
            sequence_length: Length of each sequence
            
        Returns:
            Dictionary of losses
        """
        # Sample sequential data from the replay buffer
        batch_sequences = self.replay_buffer.sample(batch_size, sequence_length)
        
        # Update networks using sequential data
        return self.update(batch_sequences)
    
    def _update_inference(self, observations, beliefs, latents, actions, neighbor_actions, 
                         rewards, next_observations, next_beliefs, next_latents, means=None, logvars=None):
        """Update inference module (encoder-decoder).
        
        Returns:
            tuple: (inference_loss_value, recon_loss, kl_loss, neighbor_action_logits)
                  The components needed for belief processor training
        """
        # Use the latents that were already calculated during inference
        # instead of recalculating them here
        
        # Predict neighbor actions using the provided latents
        neighbor_action_logits = self.decoder(observations, latents)
        
        # For backward compatibility, if neighbor_actions is a tensor of zeros, use a default target
        if isinstance(neighbor_actions, torch.Tensor) and torch.all(neighbor_actions == 0):
            # Create a default target (e.g., all zeros)
            target_actions = torch.zeros_like(neighbor_actions)
            recon_loss = F.cross_entropy(neighbor_action_logits, target_actions)
        else:
            # Use the actual neighbor actions
            recon_loss = F.cross_entropy(neighbor_action_logits, neighbor_actions)
        
        # For the KL loss, we need the mean and logvar
        # If they're not provided in the batch, we'll need to extract them from the replay buffer
        # or use the current values (which might not be accurate for all samples in the batch)
        if means is None or logvars is None:
            # We'll use the means and logvars from the batch if available
            # Otherwise, we'll use the current values (this is a simplification)
            batch_size = observations.size(0)
            means = self.current_mean.expand(batch_size, -1)
            logvars = self.current_logvar.expand(batch_size, -1)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvars - means.pow(2) - logvars.exp()) / means.size(0)
        
        # Total loss
        inference_loss = recon_loss + self.kl_weight * kl_loss
        
        # Update networks
        self.encoder_optimizer.zero_grad()
        inference_loss.backward()
        self.encoder_optimizer.step()
        
        # Return the loss value and components needed for belief processor training
        return inference_loss.item(), recon_loss, kl_loss, neighbor_action_logits
    
    def _update_belief_processor(self, observations, beliefs, latents, actions, neighbor_actions, 
                             rewards, next_observations, next_beliefs, next_latents, means=None, logvars=None,
                             policy_components=None, inference_components=None):
        """Update belief processor (GRU) network using only policy loss.
        
        Args:
            observations, beliefs, etc.: Standard inputs
            policy_components: Optional tuple of components from _update_policy
            inference_components: Not used, kept for API compatibility
        
        Returns:
            float: The belief processor loss value
        """
        # Step 1: Process the current observation to get updated beliefs
        # Reshape observations to have a sequence dimension for the GRU
        batch_size = observations.size(0)
        observations_seq = observations.unsqueeze(1)  # [batch_size, 1, obs_dim]
        
        # Reshape beliefs to [1, batch_size, belief_dim] for GRU input
        # We detach to avoid backpropagating through previous time steps
        beliefs_reshaped = beliefs.detach().unsqueeze(0)
        
        # Forward pass through the belief processor
        updated_belief_seq, _ = self.belief_processor.gru(observations_seq, beliefs_reshaped)
        updated_belief = updated_belief_seq.squeeze(1)  # [batch_size, belief_dim]
        
        # Step 2: Compute policy loss using the updated beliefs
        if policy_components is not None:
            # Use the provided policy components
            _, _, _, _, entropy, expected_q = policy_components
            
            # Recompute action probabilities with updated beliefs
            action_logits = self.policy(updated_belief, latents)
            action_probs = F.softmax(action_logits, dim=1)
            log_probs = F.log_softmax(action_logits, dim=1)
            
            # Compute entropy with the new probabilities
            new_entropy = -torch.sum(action_probs * log_probs, dim=1, keepdim=True)
            
            # Get Q-values (detached to only train the belief processor)
            with torch.no_grad():
                q1 = self.q_network1(updated_belief, latents, neighbor_actions)
                q2 = self.q_network2(updated_belief, latents, neighbor_actions)
                q = torch.min(q1, q2)
            
            # Compute expected Q-value with the new probabilities
            new_expected_q = torch.sum(action_probs * q, dim=1, keepdim=True)
            
            # Policy loss (negative because we want to maximize it)
            belief_loss = -(new_expected_q + self.entropy_weight * new_entropy).mean()
        else:
            # Compute policy components from scratch
            action_logits = self.policy(updated_belief, latents)
            action_probs = F.softmax(action_logits, dim=1)
            log_probs = F.log_softmax(action_logits, dim=1)
            
            # Compute entropy
            entropy = -torch.sum(action_probs * log_probs, dim=1, keepdim=True)
            
            # Get Q-values (detached to only train the belief processor)
            with torch.no_grad():
                q1 = self.q_network1(updated_belief, latents, neighbor_actions)
                q2 = self.q_network2(updated_belief, latents, neighbor_actions)
                q = torch.min(q1, q2)
            
            # Compute expected Q-value
            expected_q = torch.sum(action_probs * q, dim=1, keepdim=True)
            
            # Policy loss (negative because we want to maximize it)
            belief_loss = -(expected_q + self.entropy_weight * entropy).mean()
        
        # Update belief processor
        self.belief_optimizer.zero_grad()
        belief_loss.backward()
        self.belief_optimizer.step()
        
        return belief_loss.item()
    
    def _update_critics(self, observations, beliefs, latents, actions, neighbor_actions, 
                        rewards, next_observations, next_beliefs, next_latents):
        """Update Q-networks."""
        # Get current Q-values
        q1 = self.q_network1(beliefs, latents, neighbor_actions).gather(1, actions.unsqueeze(1))
        q2 = self.q_network2(beliefs, latents, neighbor_actions).gather(1, actions.unsqueeze(1))
        
        # Compute next action probabilities
        with torch.no_grad():
            next_action_logits = self.policy(next_beliefs, next_latents)
            next_action_probs = F.softmax(next_action_logits, dim=1)
            next_log_probs = F.log_softmax(next_action_logits, dim=1)
            entropy = -torch.sum(next_action_probs * next_log_probs, dim=1, keepdim=True)
            
            # Predict next neighbor actions using the decoder
            # First, get the predicted action logits for all agents
            next_neighbor_logits = self.decoder(next_observations, next_latents)
            
            # Convert logits to probabilities
            next_neighbor_probs = F.softmax(next_neighbor_logits, dim=1)
            
            # Get the most likely action for each agent
            _, predicted_actions = torch.max(next_neighbor_probs, dim=1)
            
            # Use these predicted actions as the next neighbor actions
            next_neighbor_actions = predicted_actions
            
            # Compute Q-values with predicted next neighbor actions
            next_q1 = self.target_q_network1(next_beliefs, next_latents, next_neighbor_actions)
            next_q2 = self.target_q_network2(next_beliefs, next_latents, next_neighbor_actions)
            
            # Take minimum
            next_q = torch.min(next_q1, next_q2)
            
            # Expected Q-value
            expected_q = (next_action_probs * next_q).sum(dim=1, keepdim=True)
            
            # Add entropy
            expected_q = expected_q + self.entropy_weight * entropy
            
            # Compute target
            if self.discount_factor > 0:  # Discounted return
                target_q = rewards + self.discount_factor * expected_q
            else:  # Average reward
                target_q = rewards - self.gain_parameter + expected_q
        
        # Compute loss
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss
        
        # Update networks
        self.q_optimizer.zero_grad()
        if self.discount_factor == 0:  # Only update gain parameter for average reward
            self.gain_optimizer.zero_grad()
        
        q_loss.backward()
        self.q_optimizer.step()
        
        if self.discount_factor == 0:
            self.gain_optimizer.step()
        
        return q_loss.item()
    
    def _update_policy(self, beliefs, latents, neighbor_actions):
        """Update policy network.
        
        Returns:
            tuple: (policy_loss_value, action_logits, action_probs, log_probs, entropy, expected_q)
                  The components needed for belief processor training
        """
        # Get action probabilities
        action_logits = self.policy(beliefs, latents)
        action_probs = F.softmax(action_logits, dim=1)
        log_probs = F.log_softmax(action_logits, dim=1)
        
        # Compute entropy
        entropy = -torch.sum(action_probs * log_probs, dim=1, keepdim=True)
        
        # Get Q-values
        with torch.no_grad():
            q1 = self.q_network1(beliefs, latents, neighbor_actions)
            q2 = self.q_network2(beliefs, latents, neighbor_actions)
            q = torch.min(q1, q2)
        
        # Compute expected Q-value
        expected_q = torch.sum(action_probs * q, dim=1, keepdim=True)
        
        # Policy loss is negative of expected Q-value plus entropy
        policy_loss = -(expected_q + self.entropy_weight * entropy).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Return the loss value and components needed for belief processor training
        return policy_loss.item(), action_logits, action_probs, log_probs, entropy, expected_q
    
    def _update_targets(self):
        """Update target networks."""
        for target_param, param in zip(self.target_q_network1.parameters(), self.q_network1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.target_update_rate) + 
                param.data * self.target_update_rate
            )
        
        for target_param, param in zip(self.target_q_network2.parameters(), self.q_network2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.target_update_rate) + 
                param.data * self.target_update_rate
            )
    
    def save(self, path):
        """Save agent model."""
        torch.save({
            'belief_processor': self.belief_processor.state_dict(),
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'policy': self.policy.state_dict(),
            'q_network1': self.q_network1.state_dict(),
            'q_network2': self.q_network2.state_dict(),
            'target_q_network1': self.target_q_network1.state_dict(),
            'target_q_network2': self.target_q_network2.state_dict(),
            'gain_parameter': self.gain_parameter.data,
        }, path)
    
    def load(self, path):
        """Load agent model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load components that haven't changed
        self.belief_processor.load_state_dict(checkpoint['belief_processor'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.policy.load_state_dict(checkpoint['policy'])
        
        # Handle Q-networks with potential architecture changes
        try:
            self.q_network1.load_state_dict(checkpoint['q_network1'])
            self.q_network2.load_state_dict(checkpoint['q_network2'])
            self.target_q_network1.load_state_dict(checkpoint['target_q_network1'])
            self.target_q_network2.load_state_dict(checkpoint['target_q_network2'])
        except RuntimeError as e:
            print(f"Warning: Could not load Q-networks due to architecture changes: {e}")
            print("Initializing new Q-networks. You may need to retrain the model.")
            # Initialize new Q-networks with random weights
            # The target networks will be updated from the main networks in the next update step
        
        self.gain_parameter.data = checkpoint['gain_parameter']