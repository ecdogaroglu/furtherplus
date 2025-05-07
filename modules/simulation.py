"""
Core simulation logic for FURTHER+ experiments.
"""

import numpy as np
import time
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
    load_agent_models(agents, model_path, env.num_agents)
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
    
    # Generate plots
    generate_plots(combined_metrics, env, args, output_dir, training, episodic_metrics)
    
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
        
    # Set global metrics for access in other functions
    set_metrics(metrics)
    
    # Reset and initialize agent internal states
    reset_agent_internal_states(agents)
    
    # Main simulation loop
    steps_iterator = tqdm(range(args.horizon), desc="Training" if training else "Evaluating")
    for step in steps_iterator:
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
        print(f"Step {step} - Observations: {observations}, next Observations: {next_observations}, Actions: {actions}, Rewards: {rewards}")
        
        # Update observations for next step
        observations = next_observations
        
        # Store and process metrics
        update_metrics(metrics, info, env, actions, action_probs, step, training)
        
        # Update progress display
        update_progress_display(steps_iterator, info, total_rewards, step, training)
        
        # Save models periodically if training
        if training and args.save_model and (step + 1) % max(1, args.horizon // 5) == 0:
            save_checkpoint_models(agents, output_dir, step)
        
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
        encoded_obs = encode_observation(
            signal=signal,
            neighbor_actions=neighbor_actions,
            num_agents=env.num_agents,
            num_states=env.num_states
        )
        encoded_next_obs = encode_observation(
            signal=next_signal,
            neighbor_actions=next_neighbor_actions,
            num_agents=env.num_agents,
            num_states=env.num_states
        )
        signal_one_hot = np.zeros(env.num_states)
        signal_one_hot[signal] = 1.0
        # Update agent belief state
        _, dstr = agent.observe(encoded_obs)
        print(f"Step {step}: Agent {agent_id} observed signal {signal} resulting in belief distribution {dstr.tolist()}")
        
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
            # Get current belief and latent states (before observation update)
            belief = agent.current_belief.detach().clone()  # Make a copy to ensure we have the pre-update state
            latent = agent.current_latent.detach().clone()
            
            # Infer latent state for next observation
            # This ensures we're using the correct latent state for the next observation
            next_latent = agent.infer_latent(
                encoded_obs,  
                {n_id: actions[n_id] for n_id in env.get_neighbors(agent_id) if n_id in actions},
                rewards[agent_id] if rewards else 0.0,
                encoded_next_obs 
            )
            
            # Get the updated belief state after processing the next observation
            next_belief = agent.current_belief.detach()  # This is now the updated belief after observe() was called
            
            # Get mean and logvar from inference
            mean, logvar = agent.get_latent_distribution_params()
            
            # Store transition
            store_transition_in_buffer(
                replay_buffers[agent_id],
                encoded_obs,
                belief,
                latent,
                actions[agent_id],
                rewards[agent_id] if rewards else 0.0,
                encoded_next_obs,
                next_belief,
                next_latent,
                mean,
                logvar,
                neighbor_actions
            )
            
            # Update networks if enough samples
            if len(replay_buffers[agent_id]) > args.batch_size and step % args.update_interval == 0:
                batch = replay_buffers[agent_id].sample(args.batch_size)
                agent.update(batch)

def initialize_agents(env, args, obs_dim):
    """Initialize FURTHER+ agents."""
    print(f"Initializing {env.num_agents} agents{'...' if args.eval_only else ' for evaluation...'}")
    agents = {}
    
    for agent_id in range(env.num_agents):
        agents[agent_id] = FURTHERPlusAgent(
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
            device=args.device
        )
    
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