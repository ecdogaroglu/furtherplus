import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.networks import EncoderNetwork, DecoderNetwork, PolicyNetwork, QNetwork, GRUBeliefProcessor
from modules.replay_buffer import ReplayBuffer
from modules.utils import get_best_device, encode_observation

class FURTHERPlusAgent:
    """FURTHER+ agent for social learning with additional advantage-based GRU training."""
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
        # Initialize same as before but with additional GRU optimizer
        
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
        
        # Initialize all networks as before
        self.belief_processor = GRUBeliefProcessor(
            input_dim=observation_dim,
            hidden_dim=belief_dim,
            action_dim=action_dim,
            device=device,
            num_belief_states=num_states  
        ).to(device)
        
        self.encoder = EncoderNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device,
            num_belief_states=num_states
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
            self.policy.parameters(),
            lr=learning_rate
        )
        
        # Separate GRU optimizer - only for belief processor
        self.gru_optimizer = torch.optim.Adam(
            self.belief_processor.parameters(),
            lr=learning_rate * 0.5  # Slower learning rate for stability
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
        
        # Initialize belief and latent states
        self.current_belief = torch.zeros(1, belief_dim, device=device)
        self.current_latent = torch.zeros(1, latent_dim, device=device)
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
            self.current_belief_distribution = belief_distribution
        else:
            self.current_belief_distribution = None
        
        # Reset cached values since belief state has changed
        self.reset_cache()
        
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
        # Use zeros for a complete reset
        self.current_belief = torch.zeros(1, self.belief_processor.hidden_dim, device=self.device)
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
        
        # Also reset any cached states in the GRU
        for name, param in self.belief_processor.gru.named_parameters():
            if 'bias_hh' in name:  # Reset the bias related to hidden state
                nn.init.constant_(param, 0)
    
    def infer_latent(self, observation, actions, reward, next_observation):
        """Infer latent state of neighbors based on their actions and our own observation."""
        # Convert to tensors
        observation_tensor = torch.FloatTensor(observation).to(self.device)
        if observation_tensor.dim() == 1:
            observation_tensor = observation_tensor.unsqueeze(0)
            
        next_observation_tensor = torch.FloatTensor(next_observation).to(self.device)
        if next_observation_tensor.dim() == 1:
            next_observation_tensor = next_observation_tensor.unsqueeze(0)
        
        # Convert dictionary of actions to a list in agent ID order
        actions_list = [actions.get(i, 0) for i in range(self.num_agents)]
        actions_tensor = torch.tensor([actions_list], dtype=torch.long).to(self.device)

        # Convert reward to tensor
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32).to(self.device)

        # Get latent distribution and opponent belief distribution
        mean, logvar, opponent_belief_distribution = self.encoder(
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
        
        # Store the current latent, mean, logvar, and opponent belief distribution
        self.current_latent = new_latent
        self.current_mean = mean
        self.current_logvar = logvar
        self.current_opponent_belief_distribution = opponent_belief_distribution
        
        # Compute neighbor action logits using the decoder
        with torch.no_grad():
            self.neighbor_action_logits = self.decoder(next_observation_tensor, new_latent)

        return new_latent
    
    def select_action(self):
        """Select action based on current belief and latent."""
        # Calculate fresh action logits for action selection
        action_logits = self.policy(self.current_belief, self.current_latent)
        
        # Store a detached copy for caching
        self.action_logits = action_logits.detach()
        
        # Convert to probabilities
        action_probs = F.softmax(action_logits, dim=1)
        
        # Store probability of incorrect action for learning rate calculation
        self.action_probs_history.append(action_probs.squeeze(0).detach().cpu().numpy())
        
        # Sample action from distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample().item()
        
        return action, action_probs.squeeze(0).detach().cpu().numpy()
    
    def train(self, batch_size=32, sequence_length=32):
        """Train the agent using sequential data from the replay buffer."""
        # Sample sequential data from the replay buffer
        batch_sequences = self.replay_buffer.sample(batch_size, sequence_length, mode="sequence")
        
        # Update networks using sequential data
        return self.update(batch_sequences)
    
    def update(self, batch_sequences):
        """Update all networks using sequential data."""
        if batch_sequences is None or len(batch_sequences) == 0:
            return
            
        # Initialize losses
        total_inference_loss = 0
        total_critic_loss = 0
        total_policy_loss = 0
        total_gru_loss = 0
        
        # Process each time step in the sequence
        for t, batch in enumerate(batch_sequences):
            # Unpack the batch
            (observations, beliefs, latents, actions, neighbor_actions, rewards, 
                next_observations, next_beliefs, next_latents, means, logvars) = batch
            
            # Update encoder-decoder (inference module)
            inference_loss = self._update_inference(
                neighbor_actions, 
                next_observations, next_latents,
                means, logvars
            )
            total_inference_loss += inference_loss
            
            # Update policy (with advantage for GRU)
            policy_result = self._update_policy(beliefs, latents, actions, neighbor_actions)
            policy_loss, advantage = policy_result
            total_policy_loss += policy_loss
            
            # Update GRU with advantage
            gru_loss = self._update_gru(observations, beliefs, latents, actions, advantage)
            total_gru_loss += gru_loss
            
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
            'gru_loss': total_gru_loss / sequence_length
        }
    
    def _update_inference(self, neighbor_actions, next_observations, next_latents, means, logvars):
        """Update inference module with FURTHER-style temporal KL."""
        # Generate fresh neighbor action logits for the batch
        batch_neighbor_logits = self.decoder(next_observations, next_latents)
        
        # Calculate reconstruction loss
        recon_loss = F.cross_entropy(batch_neighbor_logits, neighbor_actions)
        
        # Get sequential latent parameters for temporal KL calculation
        means_seq, logvars_seq = self.replay_buffer.get_sequential_latent_params()
        
        if means_seq is not None and logvars_seq is not None and means_seq.size(0) > 1:
            # Calculate temporal KL divergence (FURTHER-style)
            kl_loss = self._calculate_temporal_kl_divergence(means_seq, logvars_seq)
        else:
            # Fall back to standard KL if not enough sequential data
            kl_loss = -0.5 * torch.sum(1 + logvars - means.pow(2) - logvars.exp()) / means.size(0)
            kl_loss = self.kl_weight * kl_loss
        
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
        if means_seq.size(0) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # KL(N(mu_t, var_t), N(mu_{t+1}, var_{t+1}))
        kl_first_term = torch.sum(logvars_seq[:-1, :], dim=-1) - torch.sum(logvars_seq[1:, :], dim=-1)
        kl_second_term = self.latent_dim
        kl_third_term = torch.sum(1. / torch.exp(logvars_seq[:-1, :]) * torch.exp(logvars_seq[1:, :]), dim=-1)
        kl_fourth_term = (means_seq[:-1, :] - means_seq[1:, :]) / torch.exp(logvars_seq[:-1, :]) * (means_seq[:-1, :] - means_seq[1:, :])
        kl_fourth_term = kl_fourth_term.sum(dim=-1)
        
        kl = 0.5 * (kl_first_term - kl_second_term + kl_third_term + kl_fourth_term)
        return self.kl_weight * torch.mean(kl)
    
    def _update_critics(self, observations, beliefs, latents, actions, neighbor_actions, 
                        rewards, next_observations, next_beliefs, next_latents):
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
            
            # Calculate fresh neighbor action logits
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
        torch.nn.utils.clip_grad_norm_(
            list(self.q_network1.parameters()) + list(self.q_network2.parameters()),
            max_norm=1.0
        )
        self.q_optimizer.step()
        
        if self.discount_factor == 0:
            self.gain_optimizer.step()
        
        return q_loss.item()
    
    def _update_policy(self, beliefs, latents, actions, neighbor_actions):
        """Update policy network and calculate advantage for GRU training.
        
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
        
        # Calculate advantage for GRU training
        # Advantage is Q-value of taken action minus expected Q-value (baseline)
        q_actions = q.gather(1, actions.unsqueeze(1))
        advantage = q_actions - expected_q.detach()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        
        return policy_loss.item(), advantage
    
    def _update_gru(self, observations, beliefs, latents, actions, advantage):
        """Update belief processor (GRU) using advantage-based loss."""
        # Don't update if advantage is not valid
        if advantage is None or torch.isnan(advantage).any() or torch.isinf(advantage).any():
            return 0.0
        
        # Use the beliefs directly from the replay buffer
        # This is much simpler than trying to reprocess observations
        
        # Calculate policy logits using the beliefs from the buffer
        action_logits = self.policy(beliefs, latents)
        log_probs = F.log_softmax(action_logits, dim=1)
        
        # Get log probabilities of actual actions taken
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1))
        
        # Weight log probabilities by advantage (higher advantage = more weight)
        # This is similar to policy gradient with advantage weighting
        # Higher advantage means the action was better than expected, so reinforce that belief mapping
        gru_loss = -(action_log_probs * advantage.detach()).mean()
        
        # Update GRU parameters
        self.gru_optimizer.zero_grad()
        gru_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.belief_processor.parameters(), max_norm=1.0)
        self.gru_optimizer.step()
        
        return gru_loss.item()
    
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
    