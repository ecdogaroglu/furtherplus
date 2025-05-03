"""
Metrics collection and processing for FURTHER+ experiments.
"""

import json
import numpy as np
from pathlib import Path
from modules.utils import calculate_learning_rate


def initialize_metrics(env, args, training):
    """Initialize metrics dictionary for tracking experiment results."""
    metrics = {
        'mistake_rates': [],
        'incorrect_probs': [],
        'action_probs': {agent_id: [] for agent_id in range(env.num_agents)},
        'full_action_probs': {agent_id: [] for agent_id in range(env.num_agents)},
        'true_states': []
    }
    
    # Add training-specific or evaluation-specific metrics
    if training:
        metrics['training_loss'] = []
        # Add belief and latent state tracking for training if plot_internal_states is enabled
        if hasattr(args, 'plot_internal_states') and args.plot_internal_states:
            metrics['belief_states'] = {agent_id: [] for agent_id in range(env.num_agents)}
            metrics['latent_states'] = {agent_id: [] for agent_id in range(env.num_agents)}
            metrics['belief_distributions'] = {agent_id: [] for agent_id in range(env.num_agents)}
    else:
        metrics['correct_actions'] = {agent_id: 0 for agent_id in range(env.num_agents)}
        # Add belief and latent state tracking for evaluation
        metrics['belief_states'] = {agent_id: [] for agent_id in range(env.num_agents)}
        metrics['latent_states'] = {agent_id: [] for agent_id in range(env.num_agents)}
        # Add belief distribution tracking
        metrics['belief_distributions'] = {agent_id: [] for agent_id in range(env.num_agents)}
    
    print(f"Initialized metrics dictionary with {len(metrics['action_probs'])} agent entries")
    return metrics


def update_metrics(metrics, info, env, actions, action_probs, step, training):
    """Update metrics with current step information."""
    # Store mistake rate
    metrics['mistake_rates'].append(info['mistake_rate'])
    
    # Debug info about incorrect_prob on first step
    if step == 0:
        print(f"First step incorrect_prob type: {type(info['incorrect_prob'])}")
        print(f"First step incorrect_prob value: {info['incorrect_prob']}")
        if isinstance(info['incorrect_prob'], list):
            print(f"First step incorrect_prob length: {len(info['incorrect_prob'])}")
    
    # Store incorrect probabilities
    store_incorrect_probabilities(metrics, info, env.num_agents)
    
    # Debug info about action_probs on first step
    if step == 0:
        print(f"First step action_probs keys: {list(metrics['action_probs'].keys())}")
        for agent_id in metrics['action_probs']:
            print(f"Agent {agent_id} action_probs length: {len(metrics['action_probs'][agent_id])}")
    
    # Store full action probability distributions
    for agent_id, probs in action_probs.items():
        if 'full_action_probs' in metrics:
            metrics['full_action_probs'][agent_id].append(probs)
    
    # Store true state
    metrics['true_states'].append(env.true_state)
    
    # Update correct action counts for evaluation
    if not training and 'correct_actions' in metrics:
        for agent_id, action in actions.items():
            if action == env.true_state:
                metrics['correct_actions'][agent_id] += 1


def store_incorrect_probabilities(metrics, info, num_agents):
    """Store incorrect action probabilities in metrics."""
    if 'incorrect_prob' in info:
        incorrect_prob = info['incorrect_prob']
        
        # Handle both list and scalar incorrect probabilities
        if isinstance(incorrect_prob, list):
            metrics['incorrect_probs'].append(incorrect_prob)
            
            # Also store in per-agent metrics
            for agent_id, prob in enumerate(incorrect_prob):
                if agent_id < num_agents:
                    metrics['action_probs'][agent_id].append(prob)
        else:
            # If we only have a scalar, store it and duplicate for all agents
            metrics['incorrect_probs'].append(incorrect_prob)
            for agent_id in range(num_agents):
                metrics['action_probs'][agent_id].append(incorrect_prob)


def calculate_agent_learning_rates_from_metrics(metrics):
    """Calculate learning rates for each agent from metrics."""
    learning_rates = {}
    
    # Calculate learning rate for each agent
    for agent_id, probs in metrics['action_probs'].items():
        if len(probs) > 0:
            learning_rates[agent_id] = calculate_learning_rate(probs)
        else:
            learning_rates[agent_id] = 0.0
    return learning_rates


def display_learning_rate_summary(learning_rates, bound_rate):
    """Display a summary of agent learning rates."""
    fastest_agent = max(learning_rates.items(), key=lambda x: x[1])
    slowest_agent = min(learning_rates.items(), key=lambda x: x[1])
    avg_learning_rate = np.mean(list(learning_rates.values()))
    
    print("\nLearning Rate Summary:")
    print(f"  Average Rate: {avg_learning_rate:.4f}")
    print(f"  Fastest Agent: {fastest_agent[0]} (Rate: {fastest_agent[1]:.4f})")
    print(f"  Slowest Agent: {slowest_agent[0]} (Rate: {slowest_agent[1]:.4f})")
    print(f"  Bound Rate: {bound_rate:.4f}")
    
    return {
        'fastest_agent': fastest_agent,
        'slowest_agent': slowest_agent,
        'avg_learning_rate': avg_learning_rate
    }


def prepare_serializable_metrics(metrics, learning_rates, theoretical_bounds, num_steps, training):
    """Prepare metrics for JSON serialization."""
    # Find fastest and slowest learning agents
    fastest_agent = max(learning_rates.items(), key=lambda x: x[1])
    slowest_agent = min(learning_rates.items(), key=lambda x: x[1])
    avg_learning_rate = np.mean(list(learning_rates.values()))
    
    serializable_metrics = {
        'total_steps': num_steps,
        'mistake_rates': [float(m) for m in metrics['mistake_rates']],
        'incorrect_probs': [[float(p) for p in agent_probs] if isinstance(agent_probs, list) else float(agent_probs) 
                           for agent_probs in metrics['incorrect_probs']],
        'action_probs': {str(agent_id): [float(p) for p in probs] 
                         for agent_id, probs in metrics['action_probs'].items()},
        'full_action_probs': {str(agent_id): [[float(p) for p in dist] for dist in probs] 
                              for agent_id, probs in metrics['full_action_probs'].items()},
        'true_states': metrics['true_states'],
        'learning_rates': {str(k): float(v) for k, v in learning_rates.items()},
        
        # Add belief and latent states if they exist
        'has_belief_states': 'belief_states' in metrics,
        'has_latent_states': 'latent_states' in metrics,
        'has_belief_distributions': 'belief_distributions' in metrics,
        'fastest_agent': {
            'id': int(fastest_agent[0]),
            'rate': float(fastest_agent[1])
        },
        'slowest_agent': {
            'id': int(slowest_agent[0]),
            'rate': float(slowest_agent[1])
        },
        'avg_learning_rate': float(avg_learning_rate),
        'theoretical_bounds': {
            'autarky_rate': float(theoretical_bounds['autarky_rate']),
            'coordination_rate': float(theoretical_bounds['coordination_rate']),
            'bound_rate': float(theoretical_bounds['bound_rate'])
        }
    }
    
    return serializable_metrics


def save_metrics_to_file(metrics, output_dir, training):
    """Save metrics to a JSON file."""
    metrics_file = output_dir / (f'training_metrics.json' if training else 'evaluation_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)


def process_incorrect_probabilities(metrics, num_agents):
    """Process incorrect probabilities for plotting."""
    agent_incorrect_probs = metrics['action_probs']
    
    # If action_probs is empty, try to process from incorrect_probs as fallback
    if not agent_incorrect_probs:
        print("Warning: Using fallback method to process incorrect probabilities")
        agent_incorrect_probs = {}
        for step_idx, step_probs in enumerate(metrics['incorrect_probs']):
            if isinstance(step_probs, list):
                # If we have per-agent probabilities
                for agent_id, prob in enumerate(step_probs):
                    if agent_id not in agent_incorrect_probs:
                        agent_incorrect_probs[agent_id] = []
                    agent_incorrect_probs[agent_id].append(prob)
            else:
                # If we only have an average probability
                for agent_id in range(num_agents):
                    if agent_id not in agent_incorrect_probs:
                        agent_incorrect_probs[agent_id] = []
                    agent_incorrect_probs[agent_id].append(step_probs)
    
    return agent_incorrect_probs


def print_debug_info_for_plotting(agent_incorrect_probs):
    """Print debug information about the data to be plotted."""
    print(f"Number of agents with data: {len(agent_incorrect_probs)}")
    for agent_id, probs in agent_incorrect_probs.items():
        print(f"Agent {agent_id}: {len(probs)} data points")
        if len(probs) > 0:
            print(f"  First few values: {probs[:5]}")
            print(f"  Last few values: {probs[-5:]}")
        else:
            print(f"  No data points for agent {agent_id}")