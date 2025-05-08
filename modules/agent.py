import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.networks import EncoderNetwork, DecoderNetwork, PolicyNetwork, QNetwork, TransformerBeliefProcessor, TemporalGNN
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
        max_trajectory_length=50,
        use_gnn=True
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
        self.use_gnn = use_gnn
        
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
        
        # Initialize either the GNN or the traditional encoder-decoder
        if self.use_gnn:
            # Use the new TemporalGNN for inference learning
            self.inference_module = TemporalGNN(
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                num_agents=num_agents,
                device=device,
                num_belief_states=num_states,
                num_gnn_layers=2,  # Default value, will be updated later if needed
                num_attn_heads=4,  # Default value, will be updated later if needed
                dropout=0.1,
                temporal_window_size=5  # Default value, will be updated later if needed
            ).to(device)
        else:
            # Use the traditional encoder-decoder approach
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
        
        # Set up inference optimizer based on which inference module we're using
        if self.use_gnn:
            self.inference_optimizer = torch.optim.Adam(
                self.inference_module.parameters(),
                lr=learning_rate
            )
        else:
            self.inference_optimizer = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                lr=learning_rate
            )
            
        self.gain_optimizer = torch.optim.Adam([self.gain_parameter], lr=learning_rate)
        
        # Initialize belief and latent states with correct shapes
        self.current_belief = torch.ones(1, 1, belief_dim, device=device) / self.belief_processor.hidden_dim  # [1, batch_size=1, hidden_dim]
        self.current_latent = torch.ones(1, latent_dim, device=device) / latent_dim  # [1, latent_dim]
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
        
    def set_train_mode(self):
        """Set all networks to training mode."""
        self.belief_processor.train()
        if self.use_gnn:
            self.inference_module.train()
        else:
            self.encoder.train()
            self.decoder.train()
        self.policy.train()
        self.q_network1.train()
        self.q_network2.train()
        self.target_q_network1.train()
        self.target_q_network2.train()
        
    def set_eval_mode(self):
        """Set all networks to evaluation mode."""
        self.belief_processor.eval()
        if self.use_gnn:
            self.inference_module.eval()
        else:
            self.encoder.eval()
            self.decoder.eval()
        self.policy.eval()
        self.q_network1.eval()
        self.q_network2.eval()
        self.target_q_network1.eval()
        self.target_q_network2.eval()
        
    def reset_internal_state(self):
        """Reset the agent's internal state (belief and latent variables)."""
        # Use zeros for a complete reset with correct shapes
        self.current_belief = torch.zeros(1, 1, self.belief_processor.hidden_dim, device=self.device)  # [1, batch_size=1, hidden_dim]
        self.current_latent = torch.zeros(1, self.latent_dim, device=self.device)
        self.current_mean = torch.zeros(1, self.latent_dim, device=self.device)
        self.current_logvar = torch.zeros(1, self.latent_dim, device=self.device)
        
        # Reset belief distribution 
        self.current_belief_distribution = torch.ones(1, self.belief_processor.num_belief_states, device=self.device) / self.belief_processor.num_belief_states
        
        # Detach all tensors to ensure no gradient flow between episodes
        self.current_belief = self.current_belief.detach()
        self.current_latent = self.current_latent.detach()
        self.current_mean = self.current_mean.detach()
        self.current_logvar = self.current_logvar.detach()
        if self.current_belief_distribution is not None:
            self.current_belief_distribution = self.current_belief_distribution.detach()
        if hasattr(self, 'current_opponent_belief_distribution') and self.current_opponent_belief_distribution is not None:
            self.current_opponent_belief_distribution = self.current_opponent_belief_distribution.detach()
        
        # If using GNN, reset its temporal memory
        if self.use_gnn:
            self.inference_module.reset_memory()
    
    def infer_latent(self, signal, neighbor_actions, reward, next_signal):
        """Infer latent state of neighbors based on our observations which already contain neighbor actions."""

        # Convert reward to tensor
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32).to(self.device).squeeze(1)

        if self.use_gnn:
            # Use the GNN for inference
            mean, logvar, opponent_belief_distribution = self.inference_module(
                signal,
                neighbor_actions,
                reward_tensor,
                next_signal,
                self.current_latent
            )
        else:
            # Use the traditional encoder
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
        """Train agent using experiences from replay buffer."""
        # Only train if buffer has enough samples
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample sequences from replay buffer
        # For a normal sequence batch
        batch_sequences = self.replay_buffer.sample_sequence_batch(
            batch_size=batch_size, 
            sequence_length=sequence_length
        )
        
        # If we have a lot of data, also sample some transitions from previous episodes
        # This helps prevent overfitting to the current episode's state
        buffer_utilization = len(self.replay_buffer) / self.replay_buffer.capacity
        
        # If we have filled at least 30% of the buffer, start mixing in past episodes
        # This is crucial for preventing state overfitting across episodes
        if buffer_utilization > 0.3 and self.replay_buffer.episode_indices and len(self.replay_buffer.episode_indices) > 1:
            # Create a mixed batch with some sequences from previous episodes
            previous_episodes_batch = self.replay_buffer.sample_from_previous_episodes(
                batch_size=batch_size // 2,  # Half the batch from previous episodes
                sequence_length=sequence_length
            )
            
            # Combine with current episode data
            for key in batch_sequences:
                if key in previous_episodes_batch:
                    # For the first half, keep the current episode data
                    # For the second half, use data from previous episodes
                    current_half = batch_sequences[key][:batch_size//2]
                    previous_half = previous_episodes_batch[key]
                    batch_sequences[key] = torch.cat([current_half, previous_half], dim=0)
        
        # Update all network parameters
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
            
            # Update inference module
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
        
        if self.use_gnn:
            # For GNN inference module
            # We need dummy rewards for the GNN forward pass
            batch_size = signals.size(0)
            dummy_rewards = torch.zeros(batch_size, 1, device=self.device)
            
            # Forward pass through GNN to get new distribution parameters
            # Note: we detach next_latents to avoid gradients flowing back through the target network
            new_means, new_logvars, _ = self.inference_module(
                signals, 
                neighbor_actions, 
                dummy_rewards, 
                next_signals
            )
            
            # Generate action predictions using the current batch
            batch_neighbor_logits = self.inference_module.predict_actions(signals, next_latents.detach())
            
            # Reshape batch_neighbor_logits if needed for cross entropy
            if batch_neighbor_logits.dim() == 3:
                batch_size, seq_len, action_dim = batch_neighbor_logits.shape
                batch_neighbor_logits = batch_neighbor_logits.view(batch_size * seq_len, action_dim)
                neighbor_actions_reshaped = neighbor_actions.view(-1)
            else:
                neighbor_actions_reshaped = neighbor_actions
                
            # Calculate reconstruction loss
            recon_loss = F.cross_entropy(batch_neighbor_logits, neighbor_actions_reshaped)
            
            # Calculate temporal KL divergence with numerical stability
            kl_loss = self._calculate_temporal_kl_divergence(new_means, new_logvars)
        else:
            # For traditional encoder-decoder
            # Generate fresh neighbor action logits for the batch
            # Use the decoder directly on the batch
            batch_neighbor_logits = self.decoder(signals, next_latents)
        
            # Reshape if needed for cross entropy
            if batch_neighbor_logits.dim() == 3:
                batch_size, seq_len, action_dim = batch_neighbor_logits.shape
                batch_neighbor_logits = batch_neighbor_logits.view(batch_size * seq_len, action_dim)
                neighbor_actions_reshaped = neighbor_actions.view(-1)
            else:
                neighbor_actions_reshaped = neighbor_actions
            
            # Calculate reconstruction loss
            recon_loss = F.cross_entropy(batch_neighbor_logits, neighbor_actions_reshaped)
            
            # Calculate temporal KL divergence (FURTHER-style)
            kl_loss = self._calculate_temporal_kl_divergence(means, logvars)
        
        # Total loss
        inference_loss = recon_loss + kl_loss
        
        # Update networks
        self.inference_optimizer.zero_grad()
        inference_loss.backward()
        
        if self.use_gnn:
            torch.nn.utils.clip_grad_norm_(
                self.inference_module.parameters(), 
                max_norm=1.0
            )
        else:
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
        """Update policy network parameters."""
        # Compute action logits from policy
        logits = self.policy(beliefs, latents)
        
        # Compute log probabilities for actions
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for taken actions
        selected_log_probs = log_probs.gather(1, actions)
        
        # Compute entropy of the policy
        entropy = -(log_probs.exp() * log_probs).sum(dim=1, keepdim=True)
        
        # Get Q-values from both critics for the states and actions
        q1 = self.q_network1(beliefs, latents, neighbor_actions)
        q2 = self.q_network2(beliefs, latents, neighbor_actions)
        
        # Use minimum Q-value for robustness
        min_q = torch.min(q1, q2)
        
        # Get state-values by sampling actions from policy
        probs = F.softmax(logits, dim=-1)
        
        # Compute policy loss
        policy_loss = -(selected_log_probs * min_q).mean()
        
        # Add entropy for exploration
        policy_loss -= self.entropy_weight * entropy.mean()
        
        # Add regularization to prevent overfitting
        # L2 regularization on the policy network weights
        l2_reg = 0.0
        reg_weight = 0.0001  # Start with a small value
        for param in self.policy.parameters():
            l2_reg += torch.norm(param)**2
        policy_loss += reg_weight * l2_reg
        
        # Additional regularization: diversity loss to prevent converging to a single action
        # This encourages maintaining some probability of selecting different actions
        action_probs = F.softmax(logits, dim=-1)
        uniform_probs = torch.ones_like(action_probs) / self.action_dim
        diversity_loss = F.kl_div(action_probs.log(), uniform_probs, reduction='batchmean')
        
        diversity_weight = 0.01  # Weight for the diversity loss
        policy_loss -= diversity_weight * diversity_loss  # Subtract because we want to maximize diversity
        
        # Optimize policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.policy_optimizer.step()
        
        return policy_loss.item(), entropy.mean().item()
    
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
        """Save the agent's networks to a file."""
        checkpoint = {
            'belief_processor': self.belief_processor.state_dict(),
            'policy': self.policy.state_dict(),
            'q_network1': self.q_network1.state_dict(),
            'q_network2': self.q_network2.state_dict(),
            'target_q_network1': self.target_q_network1.state_dict(),
            'target_q_network2': self.target_q_network2.state_dict(),
            'gain_parameter': self.gain_parameter,
            'use_gnn': self.use_gnn,
            'agent_id': self.agent_id,
            'num_agents': self.num_agents,
            'num_states': self.num_states,
            'action_dim': self.action_dim,
            'latent_dim': self.latent_dim
        }
        
        # Save the appropriate inference module
        if self.use_gnn:
            checkpoint['inference_module'] = self.inference_module.state_dict()
            # Also save feature adapter if it exists
            if hasattr(self.inference_module, 'feature_adapter'):
                checkpoint['has_feature_adapter'] = True
        else:
            checkpoint['encoder'] = self.encoder.state_dict()
            checkpoint['decoder'] = self.decoder.state_dict()
            checkpoint['has_feature_adapter'] = False
        
        torch.save(checkpoint, path)
    
    def load(self, path, evaluation_mode=False):
        """
        Load agent model.
        
        Args:
            path: Path to the saved model
            evaluation_mode: If True, sets the model to evaluation mode after loading
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Check if the saved model uses GNN or not
        saved_model_uses_gnn = checkpoint.get('use_gnn', False)
        
        # Handle architecture mismatch between saved model and current instance
        if saved_model_uses_gnn != self.use_gnn:
            print(f"Warning: Architecture mismatch - Saved model uses GNN: {saved_model_uses_gnn}, Current model uses GNN: {self.use_gnn}")
            print("Adapting architecture to match saved model...")
            self.use_gnn = saved_model_uses_gnn
            
            # Re-initialize appropriate inference module based on saved model type
            if saved_model_uses_gnn:
                self.inference_module = TemporalGNN(
                    hidden_dim=self.latent_dim,  # Use latent_dim as hidden_dim for simplicity
                    action_dim=self.action_dim,
                    latent_dim=self.latent_dim,
                    num_agents=self.num_agents,
                    device=self.device,
                    num_belief_states=self.num_states,
                    num_gnn_layers=2,  # Default value
                    num_attn_heads=4,  # Default value
                    dropout=0.1,
                    temporal_window_size=5  # Default value
                ).to(self.device)
                
                # Update the optimizer
                self.inference_optimizer = torch.optim.Adam(
                    self.inference_module.parameters(),
                    lr=1e-3  # Default learning rate
                )
            else:
                self.encoder = EncoderNetwork(
                    action_dim=self.action_dim,
                    latent_dim=self.latent_dim,
                    hidden_dim=self.latent_dim,  # Use latent_dim as hidden_dim for simplicity
                    num_agents=self.num_agents,
                    device=self.device,
                    num_belief_states=self.num_states
                ).to(self.device)
                
                self.decoder = DecoderNetwork(
                    action_dim=self.action_dim,
                    latent_dim=self.latent_dim,
                    hidden_dim=self.latent_dim,  # Use latent_dim as hidden_dim for simplicity
                    num_agents=self.num_agents,
                    num_belief_states=self.num_states,
                    device=self.device
                ).to(self.device)
                
                # Update the optimizer
                self.inference_optimizer = torch.optim.Adam(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    lr=1e-3  # Default learning rate
                )
        
        # Check if we're loading a GRU model into a Transformer model
        is_gru_to_transformer = False
        try:
            # Try to load belief processor (may fail if architecture changed from GRU to Transformer)
            self.belief_processor.load_state_dict(checkpoint['belief_processor'])
        except RuntimeError as e:
            print(f"Warning: Could not load belief processor due to architecture change: {e}")
            print("Using the new Transformer belief processor with initialized weights.")
            print("Attempting to transfer knowledge from GRU to Transformer...")
            is_gru_to_transformer = True
            
            # Try to transfer knowledge from GRU to Transformer
            self._transfer_gru_to_transformer_knowledge(checkpoint)
            
        # Load other components that should be compatible
        try:
            self.policy.load_state_dict(checkpoint['policy'])
            print("Successfully loaded policy network.")
        except RuntimeError as e:
            print(f"Warning: Could not load policy network: {e}")
            print("Policy network will be randomly initialized.")
            
        # Handle inference module loading
        try:
            if self.use_gnn:
                if 'inference_module' in checkpoint:
                    self.inference_module.load_state_dict(checkpoint['inference_module'])
                    print("Successfully loaded GNN inference module.")
                else:
                    print("Warning: No GNN module found in checkpoint. Using initialized GNN.")
            else:
                if 'encoder' in checkpoint and 'decoder' in checkpoint:
                    self.encoder.load_state_dict(checkpoint['encoder'])
                    self.decoder.load_state_dict(checkpoint['decoder'])
                    print("Successfully loaded encoder-decoder modules.")
                else:
                    print("Warning: No encoder-decoder modules found in checkpoint.")
        except RuntimeError as e:
            print(f"Warning: Could not load inference components: {e}")
            print("Inference components will use initialized weights.")
            
        # Handle Q-networks
        try:
            self.q_network1.load_state_dict(checkpoint['q_network1'])
            self.q_network2.load_state_dict(checkpoint['q_network2'])
            self.target_q_network1.load_state_dict(checkpoint['target_q_network1'])
            self.target_q_network2.load_state_dict(checkpoint['target_q_network2'])
            print("Successfully loaded Q-networks.")
        except RuntimeError as e:
            print(f"Warning: Could not load Q-networks: {e}")
            print("Q-networks will use initialized weights.")
        
        # Load gain parameter
        try:
            if 'gain_parameter' in checkpoint:
                self.gain_parameter.data = checkpoint['gain_parameter']
                print("Successfully loaded gain parameter.")
        except Exception as e:
            print(f"Warning: Could not load gain parameter: {e}")
        
        # Reset internal state after loading
        self.reset_internal_state()
        
        # Set the model to evaluation mode if requested
        if evaluation_mode:
            self.set_eval_mode()
            print(f"Model set to evaluation mode.")
        else:
            self.set_train_mode()
            print(f"Model set to training mode.")
        
        # If we transferred from GRU to Transformer, recommend retraining
        if is_gru_to_transformer:
            print("Knowledge transfer from GRU to Transformer attempted.")
            print("For best performance, you should retrain the model for a few episodes.")
    
    def _transfer_gru_to_transformer_knowledge(self, checkpoint):
        """
        Transfer knowledge from a GRU model to a Transformer model.
        This helps preserve some of the learned knowledge when switching architectures.
        """
        try:
            # The most important part to transfer is the belief head weights
            # which map from hidden state to belief distribution
            if 'belief_processor' in checkpoint:
                gru_state_dict = checkpoint['belief_processor']
                
                # Transfer belief head weights if they have the same dimensions
                if 'belief_head.weight' in gru_state_dict and gru_state_dict['belief_head.weight'].size() == self.belief_processor.belief_head.weight.size():
                    self.belief_processor.belief_head.weight.data.copy_(gru_state_dict['belief_head.weight'])
                    self.belief_processor.belief_head.bias.data.copy_(gru_state_dict['belief_head.bias'])
                    print("Successfully transferred belief head weights from GRU to Transformer.")
                    
                # We can also try to initialize the input projection with GRU input weights
                if 'gru.weight_ih_l0' in gru_state_dict:
                    # The input weights of GRU can be used to initialize part of the input projection
                    gru_input_weights = gru_state_dict['gru.weight_ih_l0']
                    input_dim = min(gru_input_weights.size(1), self.belief_processor.input_projection.weight.size(1))
                    output_dim = min(gru_input_weights.size(0) // 3, self.belief_processor.input_projection.weight.size(0))
                    
                    # Copy the reset gate weights (first third of GRU weights)
                    self.belief_processor.input_projection.weight.data[:output_dim, :input_dim].copy_(
                        gru_input_weights[:output_dim, :input_dim]
                    )
                    print("Partially initialized Transformer input projection with GRU weights.")
                    
        except Exception as e:
            print(f"Error during knowledge transfer: {e}")
            print("Continuing with randomly initialized Transformer.")

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
    