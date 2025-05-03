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

def visualize_network(
    adjacency_matrix: np.ndarray,
    node_values: Optional[Dict[int, float]] = None,
    title: str = "Agent Network with Learning Rates",
    save_path: Optional[str] = None,
    theoretical_bounds: Optional[Dict[str, float]] = None
) -> None:
    """
    Visualize the network structure with node colors representing learning rates.
    
    Args:
        adjacency_matrix: Binary adjacency matrix
        node_values: Dictionary mapping node IDs to values (e.g., learning rates)
        title: Title of the plot
        save_path: Path to save the figure (if None, figure is displayed)
        theoretical_bounds: Dictionary of theoretical bounds to include in the legend
    """
    G = nx.DiGraph(adjacency_matrix)
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(12, 8))
    
    # Draw nodes with colors based on values
    if node_values:
        nodes = list(G.nodes())
        values = [node_values.get(n, 0) for n in nodes]
        
        # Normalize values for colormap
        vmin = min(values)
        vmax = max(values)
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            G, pos, 
            node_color=values, 
            cmap=plt.cm.viridis, 
            node_size=500, 
            alpha=0.9,
            vmin=vmin, 
            vmax=vmax
        )
        
        # Add colorbar
        plt.colorbar(nodes, label="Learning Rate")
    else:
        nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.8)
    
    # Draw edges with arrows
    nx.draw_networkx_edges(
        G, pos, 
        width=1.0, 
        alpha=0.6, 
        arrowsize=15,
        arrowstyle='->'
    )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add theoretical bounds if provided
    if theoretical_bounds:
        y_pos = 0.95
        for name, value in theoretical_bounds.items():
            plt.axhline(y=value, color='r', linestyle='--', alpha=0.7)
            plt.text(0.02, y_pos, f"{name}: {value:.4f}", 
                     transform=plt.gca().transAxes, fontsize=10)
            y_pos -= 0.05
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_mistake_rates(
    mistake_histories: Dict[str, List[float]],
    incorrect_prob_histories: Dict[str, List[float]] = None,
    theoretical_rates: Dict[str, float] = None,
    title: str = "Learning Performance Comparison",
    save_path: Optional[str] = None,
    log_scale: bool = True,
    show_incorrect_probs: bool = True
) -> None:
    """
    Plot mistake rates and/or incorrect probability assignments over time.
    
    Args:
        mistake_histories: Dictionary mapping strategy names to their mistake rate histories
        incorrect_prob_histories: Dictionary mapping strategy names to their incorrect probability histories
        theoretical_rates: Dictionary of theoretical rate lines to add
        title: Title of the plot
        save_path: Path to save the figure (if None, figure is displayed)
        log_scale: Whether to use logarithmic scale for y-axis
        show_incorrect_probs: Whether to show incorrect probability assignments instead of binary mistake rates
    """
    plt.figure(figsize=(12, 8))
    
    # Determine which histories to plot
    if show_incorrect_probs and incorrect_prob_histories:
        plot_histories = incorrect_prob_histories
        y_label = "Incorrect Probability Assignment"
    else:
        plot_histories = mistake_histories
        y_label = "Mistake Probability"
    
    # Plot histories
    for label, history in plot_histories.items():
        if log_scale:
            plt.semilogy(history, label=label, linewidth=2)
        else:
            plt.plot(history, label=label, linewidth=2)
            
        # Calculate and display learning rate
        if len(history) >= 10:
            learning_rate = calculate_learning_rate(history)
            
            # Add learning rate to label in legend
            handles, labels = plt.gca().get_legend_handles_labels()
            # Make sure learning_rate is a scalar
            try:
                lr_value = float(learning_rate)
            except (TypeError, ValueError):
                # If conversion fails, use a default value
                lr_value = 0.0
            labels[-1] = f"{label} (r = {lr_value:.4f})"
            plt.legend(handles, labels)
            
            # Also plot fitted exponential decay
            x = np.arange(len(history))
            # Use the converted value for the exponential decay
            # Make sure history[0] is a scalar
            try:
                if isinstance(history[0], (list, np.ndarray)):
                    # If it's a list or array, use the first element or average
                    if len(history[0]) > 0:
                        initial_value = float(np.mean(history[0]))
                    else:
                        initial_value = 0.5  # Default value
                else:
                    initial_value = float(history[0])
            except (TypeError, ValueError):
                # If conversion fails, use a default value
                initial_value = 0.5
            y = np.exp(-lr_value * x) * initial_value
            if log_scale:
                plt.semilogy(x, y, '--', alpha=0.5, color=plt.gca().lines[-1].get_color())
            else:
                plt.plot(x, y, '--', alpha=0.5, color=plt.gca().lines[-1].get_color())
    
    # Add theoretical rates as horizontal lines
    if theoretical_rates:
        for label, rate in theoretical_rates.items():
            # Compute exponential decay with the theoretical rate
            x = np.arange(max(len(h) for h in plot_histories.values()))
            y = np.exp(-rate * x) * 0.5  # Start at 0.5 as a reasonable initial mistake rate
            
            if log_scale:
                plt.semilogy(x, y, '-.', label=f"{label} ({rate:.4f})")
            else:
                plt.plot(x, y, '-.', label=f"{label} ({rate:.4f})")
    
    plt.xlabel("Time Steps")
    plt.ylabel(y_label + (" (log scale)" if log_scale else ""))
    plt.title(title)
    plt.grid(True, which="both" if log_scale else "major", ls="--", alpha=0.7)
    plt.legend(loc='best')
    
    # Set y-axis limits for better visualization
    if log_scale:
        plt.ylim(0.001, 1.0)
    else:
        plt.ylim(0, 1.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
        
def plot_agent_action_probabilities(
    full_action_probs: Dict[int, List[List[float]]],
    true_states: List[int] = None,
    num_states: int = 2,
    title: str = "Agent Action Probability Assignments",
    save_path: Optional[str] = None,
    max_steps: Optional[int] = None
) -> None:
    """
    Plot the action probability assignments of agents over time steps.
    
    Args:
        full_action_probs: Dictionary mapping agent IDs to lists of action probability distributions
                          [time_steps][state_probabilities]
        true_states: List of true states for each time step (to highlight correct state)
        num_states: Number of possible states
        title: Title of the plot
        save_path: Path to save the figure (if None, figure is displayed)
        max_steps: Maximum number of steps to plot (if None, plots all steps)
    """
    num_agents = len(full_action_probs)
    
    # Determine how many steps to plot
    if max_steps is None:
        # Find the agent with the most steps
        max_steps = max(len(probs) for probs in full_action_probs.values())
    else:
        # Limit to the specified number of steps
        max_steps = min(max_steps, max(len(probs) for probs in full_action_probs.values()))
    
    # Create a figure with subplots for each agent
    fig, axes = plt.subplots(
        1, 
        num_agents, 
        figsize=(5 * num_agents, 4),
        sharex=True
    )
    
    # If there's only one agent, make sure axes is an array
    if num_agents == 1:
        axes = np.array([axes])
    
    # Define colors for each state
    colors = plt.cm.tab10(np.linspace(0, 1, num_states))
    
    # Plot each agent
    for j, agent_id in enumerate(sorted(full_action_probs.keys())):
        ax = axes[j]
        
        # Get action probabilities for this agent
        agent_probs = full_action_probs[agent_id][:max_steps]
        
        # Create time steps array
        time_steps = np.arange(len(agent_probs))
        
        # Transpose the data to get state probabilities over time
        state_probs = [[] for _ in range(num_states)]
        for step_probs in agent_probs:
            for state in range(num_states):
                if state < len(step_probs):
                    state_probs[state].append(step_probs[state])
                else:
                    state_probs[state].append(0.0)
        
        # Plot probability for each state
        for state in range(num_states):
            ax.plot(
                time_steps, 
                state_probs[state], 
                label=f"State {state}", 
                color=colors[state],
                linewidth=2
            )
        
        # Highlight true state if provided
        if true_states:
            # Limit true states to the number of steps we're plotting
            plot_true_states = true_states[:len(time_steps)]
            
            # Create a step function for the true state
            prev_state = plot_true_states[0]
            state_changes = [(0, prev_state)]
            
            for t, state in enumerate(plot_true_states[1:], 1):
                if state != prev_state:
                    state_changes.append((t, state))
                    prev_state = state
            
            # Plot each segment of the true state
            for i in range(len(state_changes)):
                start_t, state = state_changes[i]
                end_t = time_steps[-1] if i == len(state_changes) - 1 else state_changes[i+1][0]
                
                ax.axhspan(
                    1.0, 1.05,
                    xmin=start_t/time_steps[-1], 
                    xmax=end_t/time_steps[-1],
                    color=colors[state], 
                    alpha=0.3,
                    label=f"True State ({state})" if i == 0 else None
                )
        
        # Set title and labels
        ax.set_title(f"Agent {agent_id}")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Action Probability")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Only add legend to the first subplot
        if j == 0:
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    else:
        plt.show()

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
        
def plot_latent_states(
    latent_states: Dict[int, List[torch.Tensor]],
    true_states: List[int] = None,
    num_states: int = 2,
    title: str = "Agent Latent States Over Time",
    save_path: Optional[str] = None,
    max_steps: Optional[int] = None,
    max_dims: int = 10,
    episode_length: Optional[int] = None
) -> None:
    """
    Plot the latent state evolution of agents over time with separate subplots for each episode.
    
    Args:
        latent_states: Dictionary mapping agent IDs to lists of latent state tensors
        true_states: List of true states for each time step (to highlight state changes)
        num_states: Number of possible states
        title: Title of the plot
        save_path: Path to save the figure (if None, figure is displayed)
        max_steps: Maximum number of steps to plot (if None, plots all steps)
        max_dims: Maximum number of latent dimensions to plot
        episode_length: Length of each episode (if None, treats all data as a single episode)
    """
    num_agents = len(latent_states)
    
    # Determine total steps and number of episodes
    total_steps = max(len(latents) for latents in latent_states.values())
    
    if episode_length is None:
        # If episode_length is not provided, treat all data as a single episode
        num_episodes = 1
        episode_length = total_steps
    else:
        # Calculate number of episodes based on total steps and episode length
        num_episodes = (total_steps + episode_length - 1) // episode_length  # Ceiling division
    
    # Limit to max_steps if specified
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
        # Recalculate number of episodes
        num_episodes = min(num_episodes, (total_steps + episode_length - 1) // episode_length)
    
    # Arrange episodes in a grid with 2 columns
    num_episode_cols = min(2, num_episodes)  # At most 2 columns
    num_episode_rows = (num_episodes + num_episode_cols - 1) // num_episode_cols  # Ceiling division
    
    # Create a figure with subplots arranged in a grid
    # Each agent gets a row for each set of episodes
    total_rows = num_agents * num_episode_rows
    fig_width = 12  # Fixed width for 2 columns
    fig_height = 4 * total_rows
    
    fig, all_axes = plt.subplots(
        total_rows,
        num_episode_cols, 
        figsize=(fig_width, fig_height),
        sharex='col',  # Share x-axis within columns
        sharey='row'   # Share y-axis within rows
    )
    
    # Make sure all_axes is a 2D array
    if total_rows == 1 and num_episode_cols == 1:
        all_axes = np.array([[all_axes]])
    elif total_rows == 1:
        all_axes = np.array([all_axes])
    elif num_episode_cols == 1:
        all_axes = np.array([[ax] for ax in all_axes])
    
    # Define colors for latent dimensions
    colors = plt.cm.tab20(np.linspace(0, 1, max_dims))
    
    # Plot each agent and episode
    for j, agent_id in enumerate(sorted(latent_states.keys())):
        for ep in range(num_episodes):
            # Calculate the row and column in the grid
            grid_row = (j * num_episode_rows) + (ep // num_episode_cols)
            grid_col = ep % num_episode_cols
            
            # Skip if we're out of bounds (can happen with odd number of episodes)
            if grid_row >= total_rows or grid_col >= num_episode_cols:
                continue
                
            ax = all_axes[grid_row, grid_col]
            
            # Calculate start and end indices for this episode
            start_idx = ep * episode_length
            end_idx = min(start_idx + episode_length, total_steps)
            
            # Skip if we're out of data for this agent
            if start_idx >= len(latent_states[agent_id]):
                continue
            
            # Get latent states for this agent and episode
            agent_latents = latent_states[agent_id][start_idx:min(end_idx, len(latent_states[agent_id]))]
            
            # Create time steps array (relative to episode start)
            time_steps = np.arange(len(agent_latents))
            
            # Skip if no data
            if len(agent_latents) == 0:
                continue
            
            # Extract latent dimensions (up to max_dims)
            latent_dim = agent_latents[0].shape[-1]
            plot_dims = min(latent_dim, max_dims)
            
            # Transpose the data to get latent dimensions over time
            latent_values = [[] for _ in range(plot_dims)]
            for latent in agent_latents:
                # Convert to numpy and flatten if needed
                if isinstance(latent, torch.Tensor):
                    latent_np = latent.detach().cpu().numpy()
                    if latent_np.ndim > 1:
                        latent_np = latent_np.flatten()
                else:
                    latent_np = np.array(latent)
                    if latent_np.ndim > 1:
                        latent_np = latent_np.flatten()
                
                # Store values for each dimension
                for dim in range(plot_dims):
                    if dim < len(latent_np):
                        latent_values[dim].append(latent_np[dim])
                    else:
                        latent_values[dim].append(0.0)
            
            # Plot each latent dimension
            for dim in range(plot_dims):
                ax.plot(
                    time_steps, 
                    latent_values[dim], 
                    label=f"Dim {dim}" if dim < 10 else None,  # Only label first 10 dimensions
                    color=colors[dim],
                    linewidth=1.5,
                    alpha=0.8
                )
            
            # Highlight true state changes if provided
            if true_states and start_idx < len(true_states):
                # Get true states for this episode
                episode_true_states = true_states[start_idx:min(end_idx, len(true_states))]
                
                # Create a step function for the true state
                if len(episode_true_states) > 0:
                    prev_state = episode_true_states[0]
                    state_changes = [(0, prev_state)]
                    
                    for t, state in enumerate(episode_true_states[1:], 1):
                        if state != prev_state:
                            state_changes.append((t, state))
                            prev_state = state
                    
                    # Plot vertical lines at state changes
                    for t, state in state_changes[1:]:  # Skip the first one (initial state)
                        ax.axvline(
                            x=t,
                            color='red',
                            linestyle='--',
                            alpha=0.5,
                            label="State Change" if t == state_changes[1][0] else None  # Only label the first one
                        )
                        ax.text(
                            t, 
                            ax.get_ylim()[1] * 0.9,
                            f"→ State {state}",
                            horizontalalignment='left',
                            verticalalignment='top',
                            fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
                        )
                    
                    # Add initial state label
                    ax.text(
                        0, 
                        ax.get_ylim()[1] * 0.9,
                        f"State {episode_true_states[0]}",
                        horizontalalignment='left',
                        verticalalignment='top',
                        fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
                    )
            
            # Set title and labels
            # Add agent ID to title
            ax.set_title(f"Agent {agent_id} - Episode {ep+1}")
            
            # Set y-label for leftmost plots
            if grid_col == 0:
                ax.set_ylabel("Latent Value")
            
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend only to the first subplot to avoid clutter
            if j == 0 and ep == 0:
                if plot_dims > 10:
                    ax.legend(loc='upper right', ncol=5, fontsize='small')
                else:
                    ax.legend(loc='upper right', ncol=min(5, plot_dims), fontsize='small')
    
    # Set common x-axis label for the bottom row
    for col in range(num_episode_cols):
        if col < all_axes.shape[1]:  # Make sure column exists
            all_axes[-1, col].set_xlabel("Time Steps")
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save the figure
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved latent states plot to {save_path}")
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
    episode_length: Optional[int] = None
) -> None:
    """
    Plot the belief distribution evolution of agents over time with separate subplots for each episode.
    
    Args:
        belief_distributions: Dictionary mapping agent IDs to lists of belief distribution tensors
        true_states: List of true states for each time step (to highlight state changes)
        title: Title of the plot
        save_path: Path to save the figure (if None, figure is displayed)
        max_steps: Maximum number of steps to plot (if None, plots all steps)
        episode_length: Length of each episode (if None, treats all data as a single episode)
    """
    
    num_agents = len(belief_distributions)
    
    # Determine total steps and number of episodes
    total_steps = max(len(beliefs) for beliefs in belief_distributions.values())
    
    if episode_length is None:
        # If episode_length is not provided, treat all data as a single episode
        num_episodes = 1
        episode_length = total_steps
    else:
        # Calculate number of episodes based on total steps and episode length
        num_episodes = (total_steps + episode_length - 1) // episode_length  # Ceiling division
    
    # Limit to max_steps if specified
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
        # Recalculate number of episodes
        num_episodes = min(num_episodes, (total_steps + episode_length - 1) // episode_length)
    
    # Arrange episodes in a grid with 2 columns
    num_episode_cols = min(2, num_episodes)  # At most 2 columns
    num_episode_rows = (num_episodes + num_episode_cols - 1) // num_episode_cols  # Ceiling division
    
    # Create a figure with subplots arranged in a grid
    # Each agent gets a row for each set of episodes
    total_rows = num_agents * num_episode_rows
    fig_width = 14  # Fixed width for 2 columns
    fig_height = 4 * total_rows
    
    fig, all_axes = plt.subplots(
        total_rows,
        num_episode_cols, 
        figsize=(fig_width, fig_height),
        sharex='col',  # Share x-axis within columns
        sharey='row'   # Share y-axis within rows
    )
    
    # Make sure all_axes is a 2D array
    if total_rows == 1 and num_episode_cols == 1:
        all_axes = np.array([[all_axes]])
    elif total_rows == 1:
        all_axes = np.array([all_axes])
    elif num_episode_cols == 1:
        all_axes = np.array([[ax] for ax in all_axes])
    
    # Plot each agent and episode
    for j, agent_id in enumerate(sorted(belief_distributions.keys())):
        for ep in range(num_episodes):
            # Calculate the row and column in the grid
            grid_row = (j * num_episode_rows) + (ep // num_episode_cols)
            grid_col = ep % num_episode_cols
            
            # Skip if we're out of bounds (can happen with odd number of episodes)
            if grid_row >= total_rows or grid_col >= num_episode_cols:
                continue
                
            ax = all_axes[grid_row, grid_col]
            
            # Calculate start and end indices for this episode
            start_idx = ep * episode_length
            end_idx = min(start_idx + episode_length, total_steps)
            
            # Skip if we're out of data for this agent
            if start_idx >= len(belief_distributions[agent_id]):
                continue
            
            # Get belief distributions for this agent and episode
            agent_beliefs = belief_distributions[agent_id][start_idx:min(end_idx, len(belief_distributions[agent_id]))]
            
            # Create time steps array (relative to episode start)
            time_steps = np.arange(len(agent_beliefs))
            
            # Skip if no data
            if len(agent_beliefs) == 0:
                continue
            
            # Get the number of belief states
            num_belief_states = agent_beliefs[0].shape[-1]
            
            # Create a heatmap-style plot
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
            
            # Plot heatmap
            im = ax.imshow(
                belief_values.T,  # Transpose to have belief states on y-axis
                aspect='auto',
                cmap='viridis',
                interpolation='nearest',
                origin='lower',
                extent=[0, len(agent_beliefs), 0, num_belief_states]
            )
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Probability')
            
            # Highlight true state changes if provided
            if true_states and start_idx < len(true_states):
                # Get true states for this episode
                episode_true_states = true_states[start_idx:min(end_idx, len(true_states))]
                
                # Create a step function for the true state
                if len(episode_true_states) > 0:
                    prev_state = episode_true_states[0]
                    state_changes = [(0, prev_state)]
                    
                    for t, state in enumerate(episode_true_states[1:], 1):
                        if state != prev_state:
                            state_changes.append((t, state))
                            prev_state = state
                    
                    # Plot vertical lines at state changes
                    for t, state in state_changes[1:]:  # Skip the first one (initial state)
                        ax.axvline(
                            x=t,
                            color='red',
                            linestyle='--',
                            alpha=0.7,
                            label="State Change" if t == state_changes[1][0] else None  # Only label the first one
                        )
                        ax.text(
                            t, 
                            num_belief_states * 0.9,
                            f"→ State {state}",
                            horizontalalignment='left',
                            verticalalignment='top',
                            fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
                        )
                    
                    # Add initial state label
                    ax.text(
                        0, 
                        num_belief_states * 0.9,
                        f"State {episode_true_states[0]}",
                        horizontalalignment='left',
                        verticalalignment='top',
                        fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
                    )
            
            # Set title and labels
            ax.set_title(f"Agent {agent_id} - Episode {ep+1}")
            ax.set_ylabel("Belief State")
            
            # Add y-ticks for each belief state
            ax.set_yticks(np.arange(num_belief_states) + 0.5)
            ax.set_yticklabels([f"State {i}" for i in range(num_belief_states)])
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3, which='both')
    
    # Set common x-axis label for the bottom row
    for col in range(num_episode_cols):
        if col < all_axes.shape[1]:  # Make sure column exists
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

def plot_belief_states(
    belief_states: Dict[int, List[torch.Tensor]],
    true_states: List[int] = None,
    num_states: int = 2,
    title: str = "Agent Belief States Over Time",
    save_path: Optional[str] = None,
    max_steps: Optional[int] = None,
    max_dims: int = 10,
    episode_length: Optional[int] = None
) -> None:
    """
    Plot the belief state evolution of agents over time with separate subplots for each episode.
    
    Args:
        belief_states: Dictionary mapping agent IDs to lists of belief state tensors
        true_states: List of true states for each time step (to highlight state changes)
        num_states: Number of possible states
        title: Title of the plot
        save_path: Path to save the figure (if None, figure is displayed)
        max_steps: Maximum number of steps to plot (if None, plots all steps)
        max_dims: Maximum number of belief dimensions to plot
        episode_length: Length of each episode (if None, treats all data as a single episode)
    """
    num_agents = len(belief_states)
    
    # Determine total steps and number of episodes
    total_steps = max(len(beliefs) for beliefs in belief_states.values())
    
    if episode_length is None:
        # If episode_length is not provided, treat all data as a single episode
        num_episodes = 1
        episode_length = total_steps
    else:
        # Calculate number of episodes based on total steps and episode length
        num_episodes = (total_steps + episode_length - 1) // episode_length  # Ceiling division
    
    # Limit to max_steps if specified
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
        # Recalculate number of episodes
        num_episodes = min(num_episodes, (total_steps + episode_length - 1) // episode_length)
    
    # Arrange episodes in a grid with 2 columns
    num_episode_cols = min(2, num_episodes)  # At most 2 columns
    num_episode_rows = (num_episodes + num_episode_cols - 1) // num_episode_cols  # Ceiling division
    
    # Create a figure with subplots arranged in a grid
    # Each agent gets a row for each set of episodes
    total_rows = num_agents * num_episode_rows
    fig_width = 12  # Fixed width for 2 columns
    fig_height = 4 * total_rows
    
    fig, all_axes = plt.subplots(
        total_rows,
        num_episode_cols, 
        figsize=(fig_width, fig_height),
        sharex='col',  # Share x-axis within columns
        sharey='row'   # Share y-axis within rows
    )
    
    # Make sure all_axes is a 2D array
    if total_rows == 1 and num_episode_cols == 1:
        all_axes = np.array([[all_axes]])
    elif total_rows == 1:
        all_axes = np.array([all_axes])
    elif num_episode_cols == 1:
        all_axes = np.array([[ax] for ax in all_axes])
    
    # Define colors for belief dimensions
    colors = plt.cm.tab20(np.linspace(0, 1, max_dims))
    
    # Plot each agent and episode
    for j, agent_id in enumerate(sorted(belief_states.keys())):
        for ep in range(num_episodes):
            # Calculate the row and column in the grid
            grid_row = (j * num_episode_rows) + (ep // num_episode_cols)
            grid_col = ep % num_episode_cols
            
            # Skip if we're out of bounds (can happen with odd number of episodes)
            if grid_row >= total_rows or grid_col >= num_episode_cols:
                continue
                
            ax = all_axes[grid_row, grid_col]
            
            # Calculate start and end indices for this episode
            start_idx = ep * episode_length
            end_idx = min(start_idx + episode_length, total_steps)
            
            # Skip if we're out of data for this agent
            if start_idx >= len(belief_states[agent_id]):
                continue
            
            # Get belief states for this agent and episode
            agent_beliefs = belief_states[agent_id][start_idx:min(end_idx, len(belief_states[agent_id]))]
            
            # Create time steps array (relative to episode start)
            time_steps = np.arange(len(agent_beliefs))
            
            # Skip if no data
            if len(agent_beliefs) == 0:
                continue
            
            # Extract belief dimensions (up to max_dims)
            belief_dim = agent_beliefs[0].shape[-1]
            plot_dims = min(belief_dim, max_dims)
            
            # Transpose the data to get belief dimensions over time
            belief_values = [[] for _ in range(plot_dims)]
            for belief in agent_beliefs:
                # Convert to numpy and flatten if needed
                if isinstance(belief, torch.Tensor):
                    belief_np = belief.detach().cpu().numpy()
                    if belief_np.ndim > 1:
                        belief_np = belief_np.flatten()
                else:
                    belief_np = np.array(belief)
                    if belief_np.ndim > 1:
                        belief_np = belief_np.flatten()
                
                # Store values for each dimension
                for dim in range(plot_dims):
                    if dim < len(belief_np):
                        belief_values[dim].append(belief_np[dim])
                    else:
                        belief_values[dim].append(0.0)
            
            # Plot each belief dimension
            for dim in range(plot_dims):
                ax.plot(
                    time_steps, 
                    belief_values[dim], 
                    label=f"Dim {dim}" if dim < 10 else None,  # Only label first 10 dimensions
                    color=colors[dim],
                    linewidth=1.5,
                    alpha=0.8
                )
            
            # Highlight true state changes if provided
            if true_states and start_idx < len(true_states):
                # Get true states for this episode
                episode_true_states = true_states[start_idx:min(end_idx, len(true_states))]
                
                # Create a step function for the true state
                if len(episode_true_states) > 0:
                    prev_state = episode_true_states[0]
                    state_changes = [(0, prev_state)]
                    
                    for t, state in enumerate(episode_true_states[1:], 1):
                        if state != prev_state:
                            state_changes.append((t, state))
                            prev_state = state
                    
                    # Plot vertical lines at state changes
                    for t, state in state_changes[1:]:  # Skip the first one (initial state)
                        ax.axvline(
                            x=t,
                            color='red',
                            linestyle='--',
                            alpha=0.5,
                            label="State Change" if t == state_changes[1][0] else None  # Only label the first one
                        )
                        ax.text(
                            t, 
                            ax.get_ylim()[1] * 0.9,
                            f"→ State {state}",
                            horizontalalignment='left',
                            verticalalignment='top',
                            fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
                        )
                    
                    # Add initial state label
                    ax.text(
                        0, 
                        ax.get_ylim()[1] * 0.9,
                        f"State {episode_true_states[0]}",
                        horizontalalignment='left',
                        verticalalignment='top',
                        fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
                    )
            
            # Set title and labels
            # Add agent ID to title
            ax.set_title(f"Agent {agent_id} - Episode {ep+1}")
            
            # Set y-label for leftmost plots
            if grid_col == 0:
                ax.set_ylabel("Belief Value")
            
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend only to the first subplot to avoid clutter
            if j == 0 and ep == 0:
                if plot_dims > 10:
                    ax.legend(loc='upper right', ncol=5, fontsize='small')
                else:
                    ax.legend(loc='upper right', ncol=min(5, plot_dims), fontsize='small')
    
    # Set common x-axis label for the bottom row
    for col in range(num_episode_cols):
        if col < all_axes.shape[1]:  # Make sure column exists
            all_axes[-1, col].set_xlabel("Time Steps")
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save the figure
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved belief states plot to {save_path}")
            plt.close()
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")
            plt.close()
    else:
        plt.show()
