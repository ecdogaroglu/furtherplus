import numpy as np
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import json
from pathlib import Path

def encode_observation(
    signal: int,
    neighbor_actions: Dict[int, int],
    num_agents: int,
    num_states: int
) -> np.ndarray:
    """
    Encode the observation (signal + neighbor actions) into a fixed-size vector.
    
    Args:
        signal: The private signal (an integer)
        neighbor_actions: Dictionary of neighbor IDs to their actions
        num_agents: Total number of agents in the environment
        num_states: Number of possible states
        
    Returns:
        encoded_obs: Encoded observation as a fixed-size vector
    """
    # One-hot encode the signal
    signal_one_hot = torch.zeros(num_states)
    signal_one_hot[signal] = 1.0
    # Encode neighbor actions (one-hot per neighbor)
    action_encoding = torch.zeros(num_agents * num_states)
    
    if neighbor_actions is not None:  # First step has no neighbor actions
        for neighbor_id, action in neighbor_actions.items():
            # Calculate the starting index for this neighbor's action encoding
            start_idx = neighbor_id * num_states
            # One-hot encode the action
            action_encoding[start_idx + action] = 1.0


    return signal_one_hot, action_encoding

def calculate_learning_rate(mistake_history: List[float]) -> float:
    """
    Calculate the learning rate (rate of decay of mistakes) using log-linear regression.
    
    Args:
        mistake_history: List of mistake rates over time
        
    Returns:
        learning_rate: Estimated learning rate
    """
    if len(mistake_history) < 10:  # Need sufficient points for regression
        return 0.0
    
    mistake_history = np.array(mistake_history)

    # Time steps
    t = np.arange(len(mistake_history))
    
    # Log of mistake probability, avoiding log(0)
    log_mistakes = np.log(np.clip(mistake_history, 1e-10, 1.0))
    
    # Simple linear regression on log-transformed data
    # log(P(mistake)) = -rt + c
    A = np.vstack([t, np.ones_like(t)]).T
    result = np.linalg.lstsq(A, log_mistakes, rcond=None)
    minus_r, c = result[0]
    
    # Negate slope to get positive learning rate
    learning_rate = -minus_r
    
    return learning_rate

def calculate_agent_learning_rates(
    incorrect_probs: Dict[int, List[List[float]]], 
    min_length: int = 100
) -> Dict[int, float]:
    """
    Calculate learning rates for each agent from their incorrect action probabilities.
    
    Args:
        incorrect_probs: Dictionary mapping agent IDs to lists of incorrect action 
                        probability histories (one list per episode)
        min_length: Minimum number of steps required for calculation
        
    Returns:
        learning_rates: Dictionary mapping agent IDs to their learning rates
    """
    learning_rates = {}
    
    for agent_id, prob_histories in incorrect_probs.items():
        # Truncate histories to a common length
        common_length = min(len(hist) for hist in prob_histories)
        if common_length < min_length:
            learning_rates[agent_id] = 0.0
            continue
            
        # Average across episodes for each time step
        avg_probs = []
        for t in range(common_length):
            avg_prob = np.mean([hist[t] for hist in prob_histories])
            avg_probs.append(avg_prob)
        
        # Calculate learning rate
        learning_rates[agent_id] = calculate_learning_rate(avg_probs)
        
    return learning_rates

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


def load_agent_models(agents, model_path, num_agents, training=True):
    """
    Load pre-trained models if a path is provided.
    
    Args:
        agents: Dictionary of agent objects
        model_path: Path to the directory containing model files
        num_agents: Number of agents to load models for
        training: Whether the models will be used for training (True) or evaluation (False)
    """
    # If no model path is provided, skip loading
    if model_path is None:
        print("No model path provided. Starting with fresh models.")
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
            # Set evaluation_mode=True if we're loading for evaluation
            agents[agent_id].load(str(model_file), evaluation_mode=not training)
            models_loaded += 1
        else:
            print(f"Warning: Model file {model_file} not found")
    
    if models_loaded == 0:
        print(f"No model files found in directory {model_dir} for any of the {num_agents} agents")
    else:
        print(f"Successfully loaded {models_loaded} models in {'training' if training else 'evaluation'} mode")


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


def reset_agent_internal_states(agents):
    """Reset all agents' internal states to ensure fresh start."""
    for agent in agents.values():
        agent.reset_internal_state()


# Global metrics dictionary for tracking
_metrics = None

def get_metrics():
    """Get the global metrics dictionary."""
    global _metrics
    return _metrics

def set_metrics(metrics):
    """Set the global metrics dictionary."""
    global _metrics
    _metrics = metrics

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



def store_transition_in_buffer(buffer, signal, neighbor_actions, belief, latent, action, reward, next_signal, 
                              next_belief, next_latent, mean, logvar):
    """Store a transition in the replay buffer."""
    buffer.push(
        signal=signal,
        neighbor_actions=neighbor_actions,
        belief=belief,
        latent=latent,
        action=action,
        reward=reward,
        next_signal=next_signal,
        next_belief=next_belief,
        next_latent=next_latent,
        mean=mean,
        logvar=logvar
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


def flatten_episodic_metrics(episodic_metrics, num_agents):
    """
    Flatten episodic metrics into a single combined metrics dictionary for backward compatibility.
    
    Args:
        episodic_metrics: Dictionary with episodes list containing metrics for each episode
        num_agents: Number of agents in the environment
        
    Returns:
        combined_metrics: Flattened metrics dictionary
    """
    if not episodic_metrics['episodes']:
        return {}  # No episodes to flatten
        
    # Initialize combined metrics based on the structure of the first episode
    first_episode = episodic_metrics['episodes'][0]
    combined_metrics = {}
    
    # Initialize each key in combined_metrics with the appropriate structure
    for key, value in first_episode.items():
        if isinstance(value, list):
            combined_metrics[key] = []
        elif isinstance(value, dict):
            combined_metrics[key] = {}
            for sub_key in value:
                if isinstance(value[sub_key], list):
                    combined_metrics[key][sub_key] = []
                else:
                    combined_metrics[key][sub_key] = value[sub_key]
        else:
            combined_metrics[key] = value
    
    # Combine metrics from all episodes
    for episode_metrics in episodic_metrics['episodes']:
        for key, value in episode_metrics.items():
            if isinstance(value, list):
                combined_metrics[key].extend(value)
            elif isinstance(value, dict):
                for sub_key in value:
                    if isinstance(value[sub_key], list):
                        if sub_key not in combined_metrics[key]:
                            combined_metrics[key][sub_key] = []
                        combined_metrics[key][sub_key].extend(value[sub_key])
                    elif sub_key not in combined_metrics[key]:
                        combined_metrics[key][sub_key] = value[sub_key]
    
    return combined_metrics