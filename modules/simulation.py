"""
Core simulation logic for FURTHER+ experiments.
"""

import torch
import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm
from modules.agent import FURTHERPlusAgent, ReplayBuffer
from modules.utils import encode_observation
from modules.metrics import (
    initialize_metrics, 
    update_metrics, 
    calculate_agent_learning_rates_from_metrics,
    display_learning_rate_summary,
    prepare_serializable_metrics,
    save_metrics_to_file
)
from modules.plotting import generate_plots


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
    # Setup environment and directories
    setup_random_seeds(args.seed, env)
    obs_dim = calculate_observation_dimension(env)
    output_dir = create_output_directory(args, env, training)
    
    # Initialize agents and metrics
    agents = initialize_agents(env, args, obs_dim)
    replay_buffers = initialize_replay_buffers(agents, args, obs_dim, training)
    load_agent_models(agents, model_path, env.num_agents)
    metrics = initialize_metrics(env, args, training)
    
    # Calculate and display theoretical bounds
    theoretical_bounds = calculate_theoretical_bounds(env)
    display_theoretical_bounds(theoretical_bounds)
    
    # Write configuration if training
    if training:
        write_config_file(args, env, theoretical_bounds, output_dir)
    
    # Run simulation
    observations, metrics = run_simulation(
        env, agents, replay_buffers, metrics, args, 
        theoretical_bounds, output_dir, training
    )
    
    # Process results
    learning_rates = calculate_agent_learning_rates_from_metrics(metrics)
    display_learning_rate_summary(learning_rates, theoretical_bounds['bound_rate'])
    
    # Save metrics and models
    serializable_metrics = prepare_serializable_metrics(
        metrics, learning_rates, theoretical_bounds, args.total_steps, training
    )
    save_metrics_to_file(serializable_metrics, output_dir, training)
    
    if training and args.save_model:
        save_final_models(agents, output_dir)
    
    # Generate plots
    generate_plots(metrics, env, args, output_dir, training)
    
    return learning_rates, serializable_metrics


def setup_random_seeds(seed, env):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)


def calculate_observation_dimension(env):
    """Calculate the observation dimension based on environment properties."""
    return env.num_states + env.num_agents * env.num_states


def create_output_directory(args, env, training):
    """Create and return the output directory for experiment results."""
    dir_prefix = "" if training else "eval_"
    output_dir = Path(args.output_dir) / args.exp_name / f"{dir_prefix}network_{args.network_type}_agents_{env.num_agents}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def initialize_agents(env, args, obs_dim):
    """Initialize FURTHER+ agents."""
    print(f"Initializing {env.num_agents} agents{'...' if args.eval_only else ' for evaluation...'}")
    agents = {}
    
    for agent_id in range(env.num_agents):
        agents[agent_id] = FURTHERPlusAgent(
            agent_id=agent_id,
            num_agents=env.num_agents,
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


def initialize_replay_buffers(agents, args, obs_dim, training):
    """Initialize replay buffers for training."""
    replay_buffers = {}
    
    if training:
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


def load_agent_models(agents, model_path, num_agents):
    """Load pre-trained models if a path is provided."""
    if model_path is None:
        return
    
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"Warning: Model directory {model_dir} does not exist")
        return
    
    models_loaded = 0
    for agent_id in range(num_agents):
        model_file = model_dir / f"agent_{agent_id}.pt"
        if model_file.exists():
            print(f"Loading model for agent {agent_id} from {model_file}")
            agents[agent_id].load(str(model_file))
            agents[agent_id].reset_internal_state()
            models_loaded += 1
        else:
            print(f"Warning: Model file {model_file} not found")
    
    if models_loaded == 0:
        print(f"No model files found in directory {model_dir} for any of the {num_agents} agents")


def calculate_theoretical_bounds(env):
    """Calculate theoretical performance bounds."""
    return {
        'autarky_rate': env.get_autarky_rate(),
        'bound_rate': env.get_bound_rate(),
        'coordination_rate': env.get_coordination_rate()
    }


def display_theoretical_bounds(bounds):
    """Display theoretical bounds information."""
    print(f"Theoretical bounds:")
    print(f"  Autarky rate: {bounds['autarky_rate']:.4f}")
    print(f"  Coordination rate: {bounds['coordination_rate']:.4f}")
    print(f"  Upper bound rate: {bounds['bound_rate']:.4f}")


def write_config_file(args, env, bounds, output_dir):
    """Write configuration to a JSON file."""
    with open(output_dir / 'config.json', 'w') as f:
        config = {
            'args': vars(args),
            'theoretical_bounds': bounds,
            'environment': {
                'num_agents': env.num_agents,
                'num_states': env.num_states,
                'signal_accuracy': env.signal_accuracy,
                'network_type': args.network_type,
                'network_density': args.network_density if args.network_type == 'random' else None
            }
        }
        json.dump(config, f, indent=2)


def run_simulation(env, agents, replay_buffers, metrics, args, theoretical_bounds, output_dir, training):
    """Run the main simulation loop."""
    num_steps = args.total_steps
    mode_str = "training" if training else "evaluation"
    
    print(f"Starting {mode_str} for {num_steps} steps...")
    start_time = time.time()
    
    # Initialize environment and agents
    observations = env.initialize()
    total_rewards = np.zeros(env.num_agents) if training else None
    metrics['true_states'].append(env.true_state)
    
    # Reset and initialize agent internal states
    reset_agent_internal_states(agents)
    initialize_agent_belief_states(agents, observations, env)
    
    # Main simulation loop
    steps_iterator = tqdm(range(num_steps), desc="Training" if training else "Evaluating")
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
        
        # Update observations for next step
        observations = next_observations
        
        # Store and process metrics
        update_metrics(metrics, info, env, actions, action_probs, step, training)
        
        # Update progress display
        update_progress_display(steps_iterator, info, total_rewards, step, training)
        
        # Save models periodically if training
        if training and args.save_model and (step + 1) % max(1, args.total_steps // 5) == 0:
            save_checkpoint_models(agents, output_dir, step)
        
        if done:
            break
    
    # Display completion time
    total_time = time.time() - start_time
    print(f"{mode_str.capitalize()} completed in {total_time:.2f} seconds")
    
    return observations, metrics


def reset_agent_internal_states(agents):
    """Reset all agents' internal states to ensure fresh start."""
    for agent in agents.values():
        agent.reset_internal_state()


def initialize_agent_belief_states(agents, observations, env):
    """Initialize agent belief states based on initial observations."""
    for agent_id, agent in agents.items():
        # Encode observation for the agent
        obs_data = observations[agent_id]
        print(f"Signal type: {type(obs_data)}, value: {obs_data}")
        
        # Extract signal from observation
        if isinstance(obs_data, dict) and 'signal' in obs_data:
            signal = obs_data['signal']
            if hasattr(signal, 'item'):  # Handle numpy scalar
                signal = signal.item()
            neighbor_actions = obs_data['neighbor_actions']
        else:
            signal = obs_data
            if not isinstance(signal, int) and hasattr(signal, 'item'):
                signal = signal.item()
            neighbor_actions = {}  # No neighbor actions initially
            
        encoded_obs = encode_observation(
            signal=signal,
            neighbor_actions=neighbor_actions if neighbor_actions is not None else {},
            num_agents=env.num_agents,
            num_states=env.num_states
        )
        
        # Initialize belief state
        agent.observe(encoded_obs, neighbor_actions=neighbor_actions if neighbor_actions is not None else {})


def select_agent_actions(agents, metrics):
    """Select actions for all agents and return with probabilities."""
    actions = {}
    action_probs = {}
    
    for agent_id, agent in agents.items():
        action, probs = agent.select_action()
        actions[agent_id] = action
        
        # Convert to numpy if it's a tensor
        if hasattr(probs, 'cpu'):
            probs_np = probs.cpu().numpy()
        else:
            probs_np = probs
            
        action_probs[agent_id] = probs_np
        
        # Store full action probability distribution
        if 'full_action_probs' in metrics:
            metrics['full_action_probs'][agent_id].append(probs_np)
    
    return actions, action_probs


def update_total_rewards(total_rewards, rewards):
    """Update total rewards for each agent."""
    for agent_id, reward in rewards.items():
        total_rewards[agent_id] += reward


def update_agent_states(agents, observations, next_observations, actions, rewards, 
                        replay_buffers, metrics, env, args, training, step):
    """Update agent states and store transitions in replay buffer."""
    for agent_id, agent in agents.items():
        # Get current and next observations
        obs_data = observations[agent_id]
        next_obs_data = next_observations[agent_id]
        
        # Extract signals from observations
        if isinstance(obs_data, dict) and 'signal' in obs_data:
            obs = obs_data['signal']
            if hasattr(obs, 'item'):  # Handle numpy scalar
                obs = obs.item()
        else:
            obs = obs_data
            if not isinstance(obs, int) and hasattr(obs, 'item'):
                obs = obs.item()
                
        if isinstance(next_obs_data, dict) and 'signal' in next_obs_data:
            next_obs = next_obs_data['signal']
            if hasattr(next_obs, 'item'):  # Handle numpy scalar
                next_obs = next_obs.item()
        else:
            next_obs = next_obs_data
            if not isinstance(next_obs, int) and hasattr(next_obs, 'item'):
                next_obs = next_obs.item()
        
        # Get neighbor actions from the observation
        if isinstance(next_obs_data, dict) and 'neighbor_actions' in next_obs_data:
            neighbor_actions = next_obs_data['neighbor_actions']
            if neighbor_actions is None:
                neighbor_actions = {}
        else:
            # Fallback to computing neighbor actions from all actions
            neighbor_actions = get_neighbor_actions(actions, agent_id, env)
        
        # Encode observations
        encoded_obs = encode_observation(
            signal=obs,
            neighbor_actions={},  # No neighbor actions for the current observation
            num_agents=env.num_agents,
            num_states=env.num_states
        )
        encoded_next_obs = encode_observation(
            signal=next_obs,
            neighbor_actions={n_id: actions[n_id] for n_id in env.get_neighbors(agent_id) if n_id in actions},
            num_agents=env.num_agents,
            num_states=env.num_states
        )
        
        # Update agent belief state
        agent.observe(encoded_next_obs, neighbor_actions=neighbor_actions)
        
        # Store internal states for visualization if evaluating
        if not training and args.plot_internal_states and 'belief_states' in metrics:
            current_belief = agent.get_belief_state()
            current_latent = agent.get_latent_state()
            if current_belief is not None:
                metrics['belief_states'][agent_id].append(current_belief.cpu().numpy())
            if current_latent is not None:
                metrics['latent_states'][agent_id].append(current_latent.cpu().numpy())
        
        # Store transition in replay buffer if training
        if training and agent_id in replay_buffers:
            # Get current belief and latent states
            belief = agent.current_belief.detach()
            latent = agent.current_latent.detach()
            next_belief = agent.current_belief.detach()  # Current belief is next belief after update
            next_latent = agent.current_latent.detach()  # Current latent is next latent after update
            
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
                if batch is not None:
                    agent.update(batch)


def store_transition_in_buffer(buffer, obs, belief, latent, action, reward, next_obs, 
                              next_belief, next_latent, mean, logvar, neighbor_actions):
    """Store a transition in the replay buffer."""
    buffer.push(
        observation=obs,
        belief=belief,
        latent=latent,
        action=action,
        peer_action=None,  # For backward compatibility
        reward=reward,
        next_observation=next_obs,
        next_belief=next_belief,
        next_latent=next_latent,
        mean=mean,
        logvar=logvar,
        neighbor_actions=neighbor_actions
    )


def get_neighbor_actions(actions, agent_id, env):
    """Get actions of neighboring agents."""
    # In the original code, neighbor_actions are directly provided in the observation
    # This is a fallback implementation that doesn't rely on env.get_neighbors()
    neighbor_actions = {}
    
    for other_id, action in actions.items():
        if other_id != agent_id:  # Don't include the agent's own action
            neighbor_actions[other_id] = action
    
    return neighbor_actions


def update_progress_display(steps_iterator, info, total_rewards, step, training):
    """Update the progress bar with current information."""
    if 'mistake_rate' in info:
        steps_iterator.set_postfix(mistake_rate=info['mistake_rate'])
    
    if training and step > 0 and step % 1000 == 0:
        avg_rewards = total_rewards / (step + 1)
        print(f"\nStep {step}: Average rewards: {avg_rewards}")


def save_checkpoint_models(agents, output_dir, step):
    """Save checkpoint models during training."""
    checkpoint_dir = output_dir / 'models' / f'checkpoint_{step+1}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for agent_id, agent in agents.items():
        agent.save(str(checkpoint_dir / f'agent_{agent_id}.pt'))
    
    print(f"Saved checkpoint models at step {step+1} to {checkpoint_dir}")


def save_final_models(agents, output_dir):
    """Save final agent models."""
    final_model_dir = output_dir / 'models' / 'final'
    final_model_dir.mkdir(parents=True, exist_ok=True)
    
    for agent_id, agent in agents.items():
        agent.save(str(final_model_dir / f'agent_{agent_id}.pt'))
    
    print(f"Saved final models to {final_model_dir}")