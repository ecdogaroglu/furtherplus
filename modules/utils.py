import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
import networkx as nx
import torch
import os
 
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
    signal_one_hot = np.zeros(num_states)
    signal_one_hot[signal] = 1.0
    
    # Encode neighbor actions (one-hot per neighbor)
    action_encoding = np.zeros(num_agents * num_states)
    
    if neighbor_actions is not None:  # First step has no neighbor actions
        for neighbor_id, action in neighbor_actions.items():
            # Calculate the starting index for this neighbor's action encoding
            start_idx = neighbor_id * num_states
            # One-hot encode the action
            action_encoding[start_idx + action] = 1.0
    
    # Concatenate signal and action encodings
    encoded_obs = np.concatenate([signal_one_hot, action_encoding])
    
    return encoded_obs

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

def plot_incorrect_action_probabilities(
    incorrect_probs: Dict[int, List[float]],
    title: str = "Incorrect Action Probabilities Over Time",
    save_path: Optional[str] = None,
    max_steps: Optional[int] = None,
    log_scale: bool = False,
    show_learning_rates: bool = True,
    episode_length: Optional[int] = None
) -> None:
    """
    Plot incorrect action probabilities for all agents with separate subplots for each episode.
    
    Args:
        incorrect_probs: Dictionary mapping agent IDs to lists of incorrect action probabilities over time
        title: Title of the plot
        save_path: Path to save the figure (if None, figure is displayed)
        max_steps: Maximum number of steps to plot (if None, plots all steps)
        log_scale: Whether to use logarithmic scale for y-axis
        show_learning_rates: Whether to calculate and display learning rates in the legend
        episode_length: Length of each episode (if None, treats all data as a single episode)
    """
    # Determine total steps and number of episodes
    total_steps = max(len(probs) for probs in incorrect_probs.values())
    
    if episode_length is None:
        # If episode_length is not provided, treat all data as a single episode
        num_episodes = 1
        episode_length = total_steps
    else:
        # Calculate number of episodes based on total steps and episode length
        num_episodes = (total_steps + episode_length - 1) // episode_length  # Ceiling division
    
    # Create a figure with subplots for each episode (2 per row)
    num_rows = (num_episodes + 1) // 2  # Ceiling division to get number of rows
    num_cols = min(2, num_episodes)  # At most 2 columns
    
    fig_width = 12  # Fixed width for 2 columns
    fig_height = 5 * num_rows  # Height scales with number of rows
    
    # Create figure and subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_width, fig_height), 
                            sharey=True, squeeze=False)
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Define a colormap for different agents
    colors = plt.cm.tab10(np.linspace(0, 1, len(incorrect_probs)))
    
    # Store learning rates for each agent across episodes
    learning_rates = {agent_id: [] for agent_id in incorrect_probs.keys()}
    
    # Plot each episode in a separate subplot
    for episode in range(num_episodes):
        ax = axes[episode]
        
        # Set subplot title
        ax.set_title(f"Episode {episode+1}")
        
        # Calculate start and end indices for this episode
        start_idx = episode * episode_length
        end_idx = min(start_idx + episode_length, total_steps)
        
        # Plot each agent's data for this episode
        for i, (agent_id, probs) in enumerate(sorted(incorrect_probs.items())):
            agent_color = colors[i]
            
            # Skip if we're out of data for this agent
            if start_idx >= len(probs):
                continue
                
            # Extract data for this episode
            episode_probs = probs[start_idx:min(end_idx, len(probs))]
            time_steps = np.arange(len(episode_probs))
            
            # Create label for this agent
            label = f"Agent {agent_id}"
            
            # Plot with appropriate scale
            if log_scale:
                line, = ax.semilogy(time_steps, episode_probs, 
                                  label=label,
                                  color=agent_color,
                                  linewidth=2)
            else:
                line, = ax.plot(time_steps, episode_probs, 
                              label=label,
                              color=agent_color,
                              linewidth=2)
            
            # Calculate and display learning rate if requested
            if show_learning_rates and len(episode_probs) >= 10:
                learning_rate = calculate_learning_rate(episode_probs)
                learning_rates[agent_id].append(learning_rate)
                
                # Update the label with learning rate
                line.set_label(f"{label} (r = {learning_rate:.4f})")
                
                # Plot fitted exponential decay
                x = np.arange(len(episode_probs))
                initial_value = episode_probs[0]
                y = np.exp(-learning_rate * x) * initial_value
                
                if log_scale:
                    ax.semilogy(x, y, '--', alpha=0.3, color=line.get_color())
                else:
                    ax.plot(x, y, '--', alpha=0.3, color=line.get_color())
        
        # Set labels and grid
        ax.set_xlabel("Time Steps")
        if episode == 0:  # Only set y-label on the first subplot
            ax.set_ylabel("Incorrect Action Probability" + (" (log scale)" if log_scale else ""))
        
        ax.grid(True, which="both" if log_scale else "major", ls="--", alpha=0.7)
        
        # Set y-axis limits for better visualization
        if log_scale:
            ax.set_ylim(0.001, 1.0)
        else:
            ax.set_ylim(0, 1.0)
        
        # Add legend to each subplot
        ax.legend(loc='best', fontsize='small')
    
    # Set a common title for the entire figure
    fig.suptitle(title, fontsize=16)
    
    # Add a text box with average learning rates across episodes
    if show_learning_rates and num_episodes > 1:
        avg_rates_text = "Average Learning Rates:\n"
        for agent_id, rates in learning_rates.items():
            if rates:  # Only include if we have rates
                avg_rate = np.mean(rates)
                avg_rates_text += f"Agent {agent_id}: {avg_rate:.4f}\n"
        
        # Add text box to the figure
        fig.text(0.01, 0.01, avg_rates_text, fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
    
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save the figure
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved incorrect action probabilities plot to {save_path}")
            plt.close()
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")
            plt.close()
    else:
        plt.show()
    

def plot_belief_distributions(
    belief_distributions: Dict[int, List[torch.Tensor]],
    true_states: List[int] = None,
    title: str = "Agent Belief Distributions Over Time",
    save_path: Optional[str] = None,
    max_steps: Optional[int] = None,
    episode_length: Optional[int] = None,
    num_episodes: Optional[int] = 1
) -> None:
    """
    Simplified version of the plot_belief_distributions function.
    """
    num_agents = len(belief_distributions)
    
    # Determine total steps
    total_steps = max(len(beliefs) for beliefs in belief_distributions.values())
    
    # If episode_length is not provided, use total_steps / num_episodes
    if episode_length is None:
        episode_length = total_steps // num_episodes if num_episodes > 0 else total_steps
    
    # Ensure we don't try to show more episodes than we have data for
    if num_episodes > 1 and episode_length * num_episodes > total_steps:
        num_episodes = max(1, total_steps // episode_length)
    
    # Create a figure with subplots arranged in a grid
    fig, all_axes = plt.subplots(
        num_agents,  # One row per agent
        num_episodes,  # One column per episode
        figsize=(8 * num_episodes, 5 * num_agents),
        sharex='col',  # Share x-axis within columns
        sharey='row'   # Share y-axis within rows
    )
    
    # Make sure all_axes is a 2D array
    if num_agents == 1 and num_episodes == 1:
        all_axes = np.array([[all_axes]])
    elif num_agents == 1:
        all_axes = np.array([all_axes])
    elif num_episodes == 1:
        all_axes = np.array([[ax] for ax in all_axes])
    
    # Plot each agent and episode
    for j, agent_id in enumerate(sorted(belief_distributions.keys())):
        for ep in range(num_episodes):
            # Each agent is a row, each episode is a column
            grid_row = j
            grid_col = ep
            
            # Skip if we're out of bounds
            if grid_row >= num_agents or grid_col >= num_episodes:
                continue
                
            ax = all_axes[grid_row, grid_col]
            
            # Calculate start and end indices for this episode
            start_idx = ep * episode_length
            end_idx = min(start_idx + episode_length, len(belief_distributions[agent_id]))
            
            # Set the title for this subplot
            ax.set_title(f"Agent {agent_id} - Episode {ep+1}")
            
            # Skip if we're out of data for this agent
            if start_idx >= len(belief_distributions[agent_id]) or start_idx == end_idx:
                continue
            
            # Get belief distributions for this agent and episode
            agent_beliefs = belief_distributions[agent_id][start_idx:end_idx]
            
            # Create time steps array (relative to episode start)
            time_steps = np.arange(len(agent_beliefs))
            
            # Skip if no data
            if len(agent_beliefs) == 0:
                continue
            
            # Get the number of belief states
            num_belief_states = agent_beliefs[0].shape[-1]
            
            # Create a line plot for each belief state
            belief_values = np.zeros((len(agent_beliefs), num_belief_states))
            for t, belief in enumerate(agent_beliefs):
                # Convert to numpy
                if isinstance(belief, torch.Tensor):
                    belief_np = belief.detach().cpu().numpy()
                    if belief_np.ndim > 1:
                        belief_np = belief_np.flatten()
                else:
                    belief_np = np.array(belief)
                    if belief_np.ndim > 1:
                        belief_np = belief_np.flatten()
                
                belief_values[t] = belief_np
            
            # Plot lines for each belief state
            colors = plt.cm.viridis(np.linspace(0, 1, num_belief_states))
            for state in range(num_belief_states):
                ax.plot(
                    time_steps, 
                    belief_values[:, state], 
                    label=f'State {state}',
                    color=colors[state],
                    linewidth=2
                )
            
            # Add legend
            if ep == 0 and j == 0:  # Only add legend to first subplot to avoid clutter
                ax.legend(loc='upper right', fontsize='small')
            
            # Set y-axis limits
            ax.set_ylim(0, 1.05)  # Probabilities range from 0 to 1
            
            # Add horizontal line at y=0.5 for reference
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
            
            # Highlight true state changes if provided
            if true_states is not None:
                # Get the true states for this episode
                episode_true_states = true_states[start_idx:min(end_idx, len(true_states))]
                
                # Add vertical lines at state changes
                prev_state = episode_true_states[0] if episode_true_states else None
                for t, state in enumerate(episode_true_states):
                    if state != prev_state:
                        ax.axvline(x=t, color='red', linestyle='--', alpha=0.5)
                        prev_state = state
    
    # Set common x-axis label for the bottom row
    for col in range(all_axes.shape[1]):
        all_axes[-1, col].set_xlabel("Time Steps")
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save the figure
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved belief distributions plot to {save_path}")
            plt.close()
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")
            plt.close()
    else:
        plt.show()

