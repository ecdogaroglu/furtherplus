import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.networks import EncoderNetwork, DecoderNetwork, PolicyNetwork, QNetwork, TransformerBeliefProcessor
from modules.replay_buffer import ReplayBuffer
from modules.utils import get_best_device, encode_observation

class FURTHERPlusAgent:
    """FURTHER+ agent for social learning with additional advantage-based Transformer training."""
    def __init__(
        self,
        agent_id,
        num_agents,
        num_states,
        observation_dim,
        action_dim,
        hidden_dim=64,
        belief_dim=64,
        latent_dim=64,
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
        self.num_states = num_states
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = device
        self.discount_factor = discount_factor
        self.entropy_weight = entropy_weight
        self.kl_weight = kl_weight
        self.target_update_rate = target_update_rate
        self.max_trajectory_length = max_trajectory_length
        self.latent_dim = latent_dim
        
        # Global variables for action logits and neighbor action logits
        self.action_logits = None
        self.neighbor_action_logits = None
        
        # Initialize replay buffer with our enhanced version
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            observation_dim=observation_dim,
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            device=device,
            sequence_length=max_trajectory_length
        )
        
        # Initialize all networks
        self.belief_processor = TransformerBeliefProcessor(
            hidden_dim=belief_dim,
            action_dim=action_dim,
            device=device,
            num_belief_states=num_states,
            nhead=4,  # Number of attention heads
            num_layers=2,  # Number of transformer layers
            dropout=0.1  # Dropout rate
        ).to(device)
        
        self.encoder = EncoderNetwork(
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device,
            num_belief_states=num_states
        ).to(device)
        
        self.decoder = DecoderNetwork(
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            num_belief_states=num_states,
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
            self.policy.parameters(),
            lr=learning_rate
        )
        
        # Separate Transformer optimizer - only for belief processor
        self.transformer_optimizer = torch.optim.Adam(
            self.belief_processor.parameters(),
            lr=learning_rate
        )
        
        self.q_optimizer = torch.optim.Adam(
            list(self.q_network1.parameters()) + 
            list(self.q_network2.parameters()),
            lr=learning_rate
        )
        self.inference_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
        )
        self.gain_optimizer = torch.optim.Adam([self.gain_parameter], lr=learning_rate)
        
        # Initialize belief and latent states with correct shapes
        self.current_belief = torch.ones(1, 1, belief_dim, device=device) / self.belief_processor.hidden_dim  # [1, batch_size=1, hidden_dim]
        self.current_latent = torch.ones(1, latent_dim, device=device) / self.encoder.fc_mean.out_features  # [1, latent_dim]
        self.current_mean = torch.zeros(1, latent_dim, device=device)
        self.current_logvar = torch.zeros(1, latent_dim, device=device)
        
        # Initialize belief distribution
        self.current_belief_distribution = torch.ones(1, self.belief_processor.num_belief_states, device=device) / self.belief_processor.num_belief_states
        
        # Initialize opponent belief distribution
        self.current_opponent_belief_distribution = torch.ones(1, self.num_agents, device=device) / self.num_agents

        # For tracking learning metrics
        self.action_probs_history = []
        
        # Episode tracking
        self.episode_step = 0
    
    def observe(self, signal, neighbor_actions):
        """Update belief state based on new observation."""
        # Check if this is the first observation of the episode
        is_first_obs = (self.episode_step == 0)
        self.episode_step += 1

        # Pass observation and belief to the belief processor
        belief, belief_distribution = self.belief_processor(
            signal,
            neighbor_actions, 
            self.current_belief
        )
        
        # Store belief state with consistent shape [1, batch_size=1, hidden_dim]
        self.current_belief = belief
        
        # Store the belief distribution
        self.current_belief_distribution = belief_distribution
        
        return self.current_belief, self.current_belief_distribution
        
    def store_transition(self, observation, belief, latent, action, reward, 
                         next_observation, next_belief, next_latent, mean=None, logvar=None, neighbor_actions=None):
        """Store a transition in the replay buffer."""
        # Ensure belief states have consistent shape before storing
        belief = self.belief_processor.standardize_belief_state(belief)
        next_belief = self.belief_processor.standardize_belief_state(next_belief)
        
        self.replay_buffer.push(
            observation, belief, latent, action, reward,
            next_observation, next_belief, next_latent, mean, logvar, neighbor_actions
        )
        
    def reset_internal_state(self):
        """Reset the agent's internal state (belief and latent variables)."""
        # Use zeros for a complete reset with correct shapes
        self.current_belief = torch.zeros(1, 1, self.belief_processor.hidden_dim, device=self.device)  # [1, batch_size=1, hidden_dim]
        self.current_latent = torch.zeros(1, self.encoder.fc_mean.out_features, device=self.device)
        self.current_mean = torch.zeros(1, self.encoder.fc_mean.out_features, device=self.device)
        self.current_logvar = torch.zeros(1, self.encoder.fc_logvar.out_features, device=self.device)
        
        # Reset belief distribution 
        self.current_belief_distribution = torch.ones(1, self.belief_processor.num_belief_states, device=self.device) / self.belief_processor.num_belief_states
        
        # Reset opponent belief distribution 
        self.current_opponent_belief_distribution = torch.ones(1, self.num_agents, device=self.device) / self.num_agents
        
        # Reset cached values to force recalculation
        self.action_logits = None
        self.neighbor_action_logits = None
        
        # Reset episode step counter
        self.episode_step = 0
        
    def reset_cache(self):
        """Reset cached values to force recalculation."""
        self.action_logits = None
        self.neighbor_action_logits = None
        
        # Detach all tensors to ensure no gradient flow between episodes
        self.current_belief = self.current_belief.detach()
        self.current_latent = self.current_latent.detach()
        self.current_mean = self.current_mean.detach()
        self.current_logvar = self.current_logvar.detach()
        if self.current_belief_distribution is not None:
            self.current_belief_distribution = self.current_belief_distribution.detach()
        if hasattr(self, 'current_opponent_belief_distribution') and self.current_opponent_belief_distribution is not None:
            self.current_opponent_belief_distribution = self.current_opponent_belief_distribution.detach()
        
        # No need to reset specific parameters for the transformer
        # as it doesn't maintain hidden state like GRU does
    
    def infer_latent(self, signal, neighbor_actions, reward, next_signal):
        """Infer latent state of neighbors based on our observations which already contain neighbor actions."""

        # Convert reward to tensor
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32).to(self.device).squeeze(1)

        # Get latent distribution and opponent belief distribution
        mean, logvar, opponent_belief_distribution = self.encoder(
            signal,
            neighbor_actions,
            reward_tensor,
            next_signal,
            self.current_latent
        )
        
        # Sample based on reparameterized distribution 
        # Ref: https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
        # Add numerical stability safeguards
        # First, clamp logvar to prevent extreme values
        logvar = torch.clamp(logvar, min=-20.0, max=2.0)
        
        # Then calculate variance with better safety measures
        var = torch.exp(0.5 * logvar)
        epsilon = 1e-6
        var = torch.clamp(var, min=epsilon, max=1e6)  # Also add maximum bound
        distribution = torch.distributions.Normal(mean, var)
        new_latent = distribution.rsample()
            
        # Store the current latent, mean, logvar, and opponent belief distribution
        self.current_latent = new_latent.unsqueeze(0)
        self.current_mean = mean
        self.current_logvar = logvar
        self.current_opponent_belief_distribution = opponent_belief_distribution
        
        return new_latent
    
    def select_action(self):
        """Select action based on current belief and latent."""

        # Calculate fresh action logits for action selection
        action_logits = self.policy(self.current_belief, self.current_latent)

        # Store a detached copy for caching
        self.action_logits = action_logits.detach()

        # Convert to probabilities
        action_probs = F.softmax(action_logits, dim=-1)

        # Store probability of incorrect action for learning rate calculation
        self.action_probs_history.append(action_probs.squeeze(0).detach().cpu().numpy())
        
        # Sample action from distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample().item()
        
        # Alternatively, use argmax for deterministic action selection
        #action = action_probs.argmax(dim=-1).item()

        
        return action, action_probs.squeeze(0).detach().cpu().numpy()
    
    def train(self, batch_size=32, sequence_length=32):
        """Train the agent using sequential data from the replay buffer."""
        # Sample sequential data from the replay buffer
        batch_sequences = self.replay_buffer.sample(batch_size, sequence_length, mode="sequence")
        
        # Update networks using sequential data
        return self.update(batch_sequences)
    
    def update(self, batch_sequences):
        """Update networks using batched data."""
        # Initialize losses
        total_inference_loss = 0
        total_critic_loss = 0
        total_policy_loss = 0
        total_transformer_loss = 0
        
        # Check if we have a single batch or a list of sequences
        if isinstance(batch_sequences, tuple):
            # Single batch case
            batch_sequences = [batch_sequences]
        
        # Process each time step in the sequence
        for t, batch in enumerate(batch_sequences):
            # Unpack the batch
            (signals, neighbor_actions, beliefs, latents, actions, rewards, next_signals, next_beliefs, next_latents, means, logvars) = batch
            
            # Update encoder-decoder (inference module)
            inference_loss = self._update_inference(
                signals,
                neighbor_actions, 
                next_signals, 
                next_latents,
                means, 
                logvars
            )
            total_inference_loss += inference_loss
            
            # Update policy (with advantage for GRU)
            policy_result = self._update_policy(beliefs, latents, actions, neighbor_actions)
            policy_loss, _ = policy_result
            total_policy_loss += policy_loss
            
            # Update Transformer with advantage
            transformer_loss = self._update_transformer(
                signals, 
                neighbor_actions, 
                beliefs,
                next_signals)
            total_transformer_loss += transformer_loss
            
            # Update Q-networks
            critic_loss = self._update_critics(
                signals, 
                neighbor_actions, 
                beliefs, 
                latents, 
                actions, 
                neighbor_actions, 
                rewards, 
                next_signals, 
                next_beliefs, 
                next_latents
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
            'transformer_loss': total_transformer_loss / sequence_length
        }
    
    def _update_inference(self, signals, neighbor_actions, next_signals, next_latents, means, logvars):
        """Update inference module with FURTHER-style temporal KL."""
        # Generate fresh neighbor action logits for the batch
        # Use the decoder directly on the batch
        batch_neighbor_logits = self.decoder(signals, next_latents)
        
        # Reshape if needed for cross entropy
        if batch_neighbor_logits.dim() == 3:
            batch_size, seq_len, action_dim = batch_neighbor_logits.shape
            batch_neighbor_logits = batch_neighbor_logits.view(batch_size * seq_len, action_dim)
            neighbor_actions = neighbor_actions.view(-1)
        
        # Calculate reconstruction loss
        recon_loss = F.cross_entropy(batch_neighbor_logits, neighbor_actions)
        
        # Calculate temporal KL divergence (FURTHER-style)
        kl_loss = self._calculate_temporal_kl_divergence(means, logvars)
        
        # Total loss
        inference_loss = recon_loss + kl_loss
        
        # Update networks
        self.inference_optimizer.zero_grad()
        inference_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), 
            max_norm=1.0
        )
        self.inference_optimizer.step()
        
        return inference_loss.item()
    
    def _calculate_temporal_kl_divergence(self, means_seq, logvars_seq):
        """Calculate KL divergence between sequential latent states (temporal smoothing)."""

        # KL(N(mu,E), N(m, S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m - mu)^T S^-1 (m - mu)))
        # Ref: https://github.com/lmzintgraf/varibad/blob/master/vae.py
        # Ref: https://github.com/dkkim93/further/blob/main/algorithm/further/agent.py

        kl_first_term = torch.sum(logvars_seq[:-1, :], dim=-1) - torch.sum(logvars_seq[1:, :], dim=-1)
        kl_second_term = self.latent_dim
        kl_third_term = torch.sum(1. / torch.exp(logvars_seq[:-1, :]) * torch.exp(logvars_seq[1:, :]), dim=-1)
        kl_fourth_term = (means_seq[:-1, :] - means_seq[1:, :]) / torch.exp(logvars_seq[:-1, :]) * (means_seq[:-1, :] - means_seq[1:, :])
        kl_fourth_term = kl_fourth_term.sum(dim=-1)
        
        kl = 0.5 * (kl_first_term - kl_second_term + kl_third_term + kl_fourth_term)

        return self.kl_weight * torch.mean(kl)
    
    def _update_critics(self, signals, neighbor_actions, beliefs, latents, actions, next_neighbor_actions, rewards, next_signals, next_beliefs, next_latents):
        """Update Q-networks."""
        # Get current Q-values
        q1 = self.q_network1(beliefs, latents, neighbor_actions).gather(1, actions.unsqueeze(1))
        q2 = self.q_network2(beliefs, latents, neighbor_actions).gather(1, actions.unsqueeze(1))
        
        # Compute next action probabilities
        with torch.no_grad():
            # Calculate fresh action logits for critic update
            next_action_logits = self.policy(next_beliefs, next_latents)
            next_action_probs = F.softmax(next_action_logits, dim=1)
            next_log_probs = F.log_softmax(next_action_logits, dim=1)
            entropy = -torch.sum(next_action_probs * next_log_probs, dim=1, keepdim=True)
            
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
        torch.nn.utils.clip_grad_norm_(
            list(self.q_network1.parameters()) + list(self.q_network2.parameters()),
            max_norm=1.0
        )
        self.q_optimizer.step()
        
        if self.discount_factor == 0:
            self.gain_optimizer.step()
        
        return q_loss.item()
    
    def _update_policy(self, beliefs, latents, actions, neighbor_actions):
        """Update policy network and calculate advantage for Transformer training.
        
        Returns:
            Tuple of (policy_loss_value, advantage)
        """
        # Generate fresh action logits for the batch
        action_logits = self.policy(beliefs, latents)
        
        # Calculate probabilities from the logits
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
        
        # Calculate advantage for Transformer training
        # Advantage is Q-value of taken action minus expected Q-value (baseline)
        q_actions = q.gather(1, actions.unsqueeze(1))
        advantage = q_actions - expected_q.detach()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        
        return policy_loss.item(), advantage
    
    def _update_transformer(self, signals, neighbor_actions, beliefs, 
                next_signals):
        """
        Update belief processor (Transformer) by optimizing for next state prediction.
        
        Args:
            signals: Current signals/observations
            neighbor_actions: Current neighbor actions
            beliefs: Current belief states
            latents: Current latent states
            actions: Actions taken
            next_signals: Next signals/observations
        
        Returns:
            float: Loss value
        """
        # Process current signals and neighbor actions through Transformer to get next belief
        _, belief_distributions = self.belief_processor(
            signals, neighbor_actions, beliefs
        )
        
        # The belief distribution should predict the next observation (signal)
        # Log likelihood of next signal given current belief
        # Use belief distribution as prediction of next signal
        transformer_loss = F.binary_cross_entropy(
            belief_distributions, next_signals, reduction='none'
        ).sum(dim=1).mean()
                
        # Update Transformer parameters
        self.transformer_optimizer.zero_grad()
        transformer_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.belief_processor.parameters(), max_norm=1.0)
        self.transformer_optimizer.step()
        
        return transformer_loss.item()
    
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
        
        # Load components
        self.belief_processor.load_state_dict(checkpoint['belief_processor'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.policy.load_state_dict(checkpoint['policy'])
        
        # Handle Q-networks
        try:
            self.q_network1.load_state_dict(checkpoint['q_network1'])
            self.q_network2.load_state_dict(checkpoint['q_network2'])
            self.target_q_network1.load_state_dict(checkpoint['target_q_network1'])
            self.target_q_network2.load_state_dict(checkpoint['target_q_network2'])
        except RuntimeError as e:
            print(f"Warning: Could not load Q-networks due to architecture changes: {e}")
            print("Initializing new Q-networks. You may need to retrain the model.")
        
        self.gain_parameter.data = checkpoint['gain_parameter']
        
        # Reset internal state after loading
        self.reset_internal_state()

    def get_belief_state(self):
        """Return the current belief state.
        
        Returns:
            belief: Current belief state tensor with shape [1, batch_size=1, hidden_dim]
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
    
    def get_opponent_belief_distribution(self):
        """Return the current opponent belief distribution.
        
        Returns:
            opponent_belief_distribution: Current opponent belief distribution tensor or None if not available
        """
        return self.current_opponent_belief_distribution if hasattr(self, 'current_opponent_belief_distribution') else None
        
    def end_episode(self):
        """
        Backward compatibility method - does nothing in the continuous version.
        The internal state is maintained across what would have been episode boundaries.
        """
        pass
    