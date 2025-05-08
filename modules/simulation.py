"""
Core simulation logic for FURTHER+ experiments.
"""

import numpy as np
import time
import torch
from tqdm import tqdm
from modules.utils import (
    create_output_directory,
    calculate_observation_dimension,
    encode_observation, 
    setup_random_seeds,
    calculate_theoretical_bounds,
    display_theoretical_bounds,
    write_config_file,
    flatten_episodic_metrics,
    save_final_models,
    save_checkpoint_models,
    set_metrics,
    reset_agent_internal_states,
    update_total_rewards,
    select_agent_actions,
    update_progress_display,
    store_transition_in_buffer,
    load_agent_models)
from modules.metrics import (
    initialize_metrics, 
    update_metrics, 
    calculate_agent_learning_rates_from_metrics,
    display_learning_rate_summary,
    prepare_serializable_metrics,
    save_metrics_to_file
)
from modules.plotting import generate_plots
from modules.agent import FURTHERPlusAgent
from modules.networks import TemporalGNN
from modules.replay_buffer import ReplayBuffer

def run_agents(env, args, training=True, model_path=None):
    """
    Run FURTHER+ agents in the social learning environment.
    
    Args:
        env: The social learning environment
        args: Command-line arguments
        training: Whether to train the agents (True) or just evaluate (False)
        model_path: Path to load models from (optional)
    
    Returns:
        learning_rates: Dictionary of learning rates for each agent
        serializable_metrics: Dictionary of metrics for JSON serialization
    """
    # Setup directory
    output_dir = create_output_directory(args, env, training)
    
    # Initialize agents and metrics
    obs_dim = calculate_observation_dimension(env)
    agents = initialize_agents(env, args, obs_dim)
    load_agent_models(agents, model_path, env.num_agents, training=training)
    metrics = initialize_metrics(env, args, training)

    # Calculate and display theoretical bounds
    theoretical_bounds = calculate_theoretical_bounds(env)
    display_theoretical_bounds(theoretical_bounds)
    
    replay_buffers = initialize_replay_buffers(agents, args, obs_dim)

    # Write configuration if training
    if training:
        write_config_file(args, env, theoretical_bounds, output_dir)


    print(f"Running {args.num_episodes} episode(s) with {args.horizon} steps per episode")
    
    # Initialize episodic metrics to store each episode separately
    episodic_metrics = {
        'episodes': []
    }
    
    # Episode loop
    for episode in range(args.num_episodes):
        # Set a different seed for each episode based on the base seed
        episode_seed = args.seed + episode
        setup_random_seeds(episode_seed, env)
        print(f"\nStarting episode {episode+1}/{args.num_episodes} with seed {episode_seed}")
        
        # Initialize fresh metrics for this episode
        metrics = initialize_metrics(env, args, training)
        
        # Run simulation for this episode
        observations, episode_metrics = run_simulation(
            env, agents, replay_buffers, metrics, args,
            output_dir, training
        )
        
        # Mark the end of this episode in the replay buffers
        if training:
            for agent_id, buffer in replay_buffers.items():
                buffer.end_trajectory()
                print(f"Marked end of episode {episode+1} in replay buffer for agent {agent_id}")
                print(f"Buffer now contains {len(buffer.episode_indices)} complete episodes")
        
        # Store this episode's metrics separately
        episodic_metrics['episodes'].append(episode_metrics)
    
    # Create a flattened version of metrics
    combined_metrics = flatten_episodic_metrics(episodic_metrics, env.num_agents)
    
    # Process results
    learning_rates = calculate_agent_learning_rates_from_metrics(combined_metrics)
    display_learning_rate_summary(learning_rates, theoretical_bounds['bound_rate'])
    
    # Save metrics and models
    serializable_metrics = prepare_serializable_metrics(
        combined_metrics, learning_rates, theoretical_bounds, args.horizon, training
    )
    
    # Also save the episodic metrics for more detailed analysis
    episodic_serializable_metrics = {
        'episodic_data': episodic_metrics,
        'learning_rates': learning_rates,
        'theoretical_bounds': theoretical_bounds,
        'episode_length': args.horizon,
        'num_episodes': args.num_episodes
    }
    
    save_metrics_to_file(serializable_metrics, output_dir, training)
    save_metrics_to_file(episodic_serializable_metrics, output_dir, training, filename='episodic_metrics.json')
    
    if training and args.save_model:
        save_final_models(agents, output_dir)
    
    # Generate plots with LaTeX style if requested
    generate_plots(
        combined_metrics, 
        env, 
        args, 
        output_dir, 
        training, 
        episodic_metrics,
        use_latex=args.use_tex if hasattr(args, 'use_tex') else False
    )
    
    return learning_rates, serializable_metrics

def run_simulation(env, agents, replay_buffers, metrics, args, output_dir, training):
    """Run the main simulation loop."""
    mode_str = "training" if training else "evaluation"
    
    print(f"Starting {mode_str} for {args.horizon} steps...")
    start_time = time.time()
    
    # Initialize environment and agents
    observations = env.initialize()
    total_rewards = np.zeros(env.num_agents) if training else None
    
    # Print the true state
    print(f"True state for this episode: {env.true_state}")
    
    # AUGMENTATION: If training, occasionally introduce synthetic state variations
    # This helps prevent overfitting to a specific state within an episode
    should_augment = training and getattr(args, 'use_state_augmentation', False)
    augmentation_frequency = getattr(args, 'augmentation_frequency', 10)
    synthetic_state_prob = getattr(args, 'synthetic_state_prob', 0.2)
    original_true_state = env.true_state
        
    # Set global metrics for access in other functions
    set_metrics(metrics)
    
    # Reset and initialize agent internal states
    reset_agent_internal_states(agents)
    
    # Set agents to appropriate mode
    for agent_id, agent in agents.items():
        if training:
            agent.set_train_mode()
        else:
            agent.set_eval_mode()
    
    # Main simulation loop
    steps_iterator = tqdm(range(args.horizon), desc="Training" if training else "Evaluating")
    for step in steps_iterator:
        # AUGMENTATION: Occasionally introduce synthetic state variations during training
        if should_augment and training and step > 0 and step % augmentation_frequency == 0:
            if np.random.random() < synthetic_state_prob:
                # Temporarily switch to a different state to introduce variability
                temp_state = (env.true_state + 1) % env.num_states
                env.true_state = temp_state
                print(f"[Augmentation] Temporarily using synthetic state: {temp_state} at step {step}")
            else:
                # Restore original state
                env.true_state = original_true_state
        
        # Get agent actions
        actions, action_probs = select_agent_actions(agents, metrics)
        
        # Take environment step
        next_observations, rewards, done, info = env.step(actions, action_probs)
        
        # Update rewards if training
        if training and rewards:
            update_total_rewards(total_rewards, rewards)
        
        # Update agent states and store transitions
        update_agent_states(
            agents, observations, next_observations, actions, rewards, 
            replay_buffers, metrics, env, args, training, step
        )
        
        # Update observations for next step
        observations = next_observations
        
        # Store and process metrics
        update_metrics(metrics, info, env, actions, action_probs, step, training)
        
        # Update progress display
        update_progress_display(steps_iterator, info, total_rewards, step, training)
        
        # Save models periodically if training
        if training and args.save_model and (step + 1) % max(1, args.horizon // 5) == 0:
            save_checkpoint_models(agents, output_dir, step)
        
        # AUGMENTATION: Reset to original state at the end of each augmentation cycle
        if should_augment and training and (step + 1) % augmentation_frequency == 0:
            env.true_state = original_true_state
        
        if done:
            break
    
    # Display completion time
    total_time = time.time() - start_time
    print(f"{mode_str.capitalize()} completed in {total_time:.2f} seconds")
    
    return observations, metrics

def update_agent_states(agents, observations, next_observations, actions, rewards, 
                        replay_buffers, metrics, env, args, training, step):
    """Update agent states and store transitions in replay buffer."""
    for agent_id, agent in agents.items():
        # Get current and next observations
        obs_data = observations[agent_id]
        next_obs_data = next_observations[agent_id]
        
        # Extract signals from observations
        signal = obs_data['signal']
        next_signal = next_obs_data['signal']
        
        # Get neighbor actions from the observation
        neighbor_actions = obs_data['neighbor_actions']
        next_neighbor_actions = next_obs_data['neighbor_actions']

        # Encode observations
        signal_encoded, actions_encoded = encode_observation(
            signal=signal,
            neighbor_actions=neighbor_actions,
            num_agents=env.num_agents,
            num_states=env.num_states
        )
        next_signal_encoded, _ = encode_observation(
            signal=next_signal,
            neighbor_actions=next_neighbor_actions,
            num_agents=env.num_agents,
            num_states=env.num_states
        )

        # Get current belief and latent states (before observation update)
        belief = agent.current_belief.detach().clone()  # Make a copy to ensure we have the pre-update state
        latent = agent.current_latent.detach().clone()

        # Update agent belief state
        next_belief, next_dstr = agent.observe(signal_encoded, actions_encoded)
        # Infer latent state for next observation
        # This ensures we're using the correct latent state for the next observation
        next_latent = agent.infer_latent(
            signal_encoded,
            actions_encoded,
            rewards[agent_id] if rewards else 0.0,
            next_signal_encoded
        )
            
        # Store internal states for visualization if requested (for both training and evaluation)

        if args.plot_internal_states and 'belief_states' in metrics:
            current_belief = agent.get_belief_state()
            current_latent = agent.get_latent_state()
            metrics['belief_states'][agent_id].append(current_belief.detach().cpu().numpy())
            metrics['latent_states'][agent_id].append(current_latent.detach().cpu().numpy())
            
            # Store belief distribution if available
            belief_distribution = agent.get_belief_distribution()
            metrics['belief_distributions'][agent_id].append(belief_distribution.detach().cpu().numpy())
                
            # Store opponent belief distribution if available
            opponent_belief_distribution = agent.get_opponent_belief_distribution()
            metrics['opponent_belief_distributions'][agent_id].append(opponent_belief_distribution.detach().cpu().numpy())
        
        # Store transition in replay buffer if training
        if training and agent_id in replay_buffers:
            
            # Get mean and logvar from inference
            mean, logvar = agent.get_latent_distribution_params()
            
            # Store transition
            store_transition_in_buffer(
                replay_buffers[agent_id],
                signal_encoded,
                actions_encoded,
                belief,
                latent,
                actions[agent_id],
                rewards[agent_id] if rewards else 0.0,
                next_signal_encoded,
                next_belief,
                next_latent,
                mean,
                logvar
            )
            
            # Update networks if enough samples
            if len(replay_buffers[agent_id]) > args.batch_size and step % args.update_interval == 0:
                # Sample a batch from the replay buffer
                batch = replay_buffers[agent_id].sample(args.batch_size)
                # Update network parameters
                agent.update(batch)

def initialize_agents(env, args, obs_dim):
    """Initialize FURTHER+ agents."""
    print(f"Initializing {env.num_agents} agents{'...' if args.eval_only else ' for evaluation...'}")
    
    # Log if using GNN
    if args.use_gnn:
        print(f"Using Graph Neural Network with {args.gnn_layers} layers, {args.attn_heads} attention heads, and temporal window of {args.temporal_window}")
    else:
        print("Using traditional encoder-decoder inference module")
        
    agents = {}
    
    for agent_id in range(env.num_agents):
        agent = FURTHERPlusAgent(
            agent_id=agent_id,
            num_agents=env.num_agents,
            num_states=env.num_states,
            observation_dim=obs_dim,
            action_dim=env.num_states,
            hidden_dim=args.hidden_dim,
            belief_dim=args.belief_dim,
            latent_dim=args.latent_dim,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            entropy_weight=args.entropy_weight,
            kl_weight=args.kl_weight,
            device=args.device,
            buffer_capacity=args.buffer_capacity,
            max_trajectory_length=args.horizon,
            use_gnn=args.use_gnn
        )
        
        # If using GNN, update the inference module with the specified parameters
        if args.use_gnn and hasattr(agent, 'inference_module'):
            agent.inference_module = TemporalGNN(
                hidden_dim=args.hidden_dim,
                action_dim=env.num_states,
                latent_dim=args.latent_dim,
                num_agents=env.num_agents,
                device=args.device,
                num_belief_states=env.num_states,
                num_gnn_layers=args.gnn_layers,
                num_attn_heads=args.attn_heads,
                dropout=0.1,
                temporal_window_size=args.temporal_window
            ).to(args.device)
            
            # Update the optimizer to use the new inference module
            agent.inference_optimizer = torch.optim.Adam(
                agent.inference_module.parameters(),
                lr=args.learning_rate
            )
            
        agents[agent_id] = agent
    
    return agents

def initialize_replay_buffers(agents, args, obs_dim):
    """Initialize replay buffers for training."""
    replay_buffers = {}
    
    for agent_id in agents:
        replay_buffers[agent_id] = ReplayBuffer(
            capacity=args.buffer_capacity,
            observation_dim=obs_dim,
            belief_dim=args.belief_dim,
            latent_dim=args.latent_dim,
            device=args.device,
            sequence_length=8  # Default sequence length for sampling
        )
    return replay_buffers