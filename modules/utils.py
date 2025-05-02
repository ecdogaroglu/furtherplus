import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
import networkx as nx
import torch
 
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
        
def plot_learning_rate_by_network_size(
    network_sizes: List[int],
    learning_rates: Dict[str, List[float]],
    theoretical_bounds: Dict[str, float] = None,
    title: str = "Learning Rate vs Network Size",
    save_path: Optional[str] = None
) -> None:
    """
    Plot how learning rate scales with network size.
    
    Args:
        network_sizes: List of network sizes
        learning_rates: Dictionary mapping rate type (e.g., 'fastest', 'slowest', 'average') 
                       to list of learning rates for each network size
        theoretical_bounds: Dictionary of theoretical bounds to include as horizontal lines
        title: Title of the plot
        save_path: Path to save the figure (if None, figure is displayed)
    """
    plt.figure(figsize=(10, 6))
    
    # Plot learning rates vs network size
    for label, rates in learning_rates.items():
        plt.plot(network_sizes, rates, 'o-', label=label, linewidth=2, markersize=8)
    
    # Add theoretical bounds as horizontal lines
    if theoretical_bounds:
        for label, value in theoretical_bounds.items():
            plt.axhline(y=value, linestyle='--', label=f"{label} ({value:.4f})")
    
    plt.xlabel("Network Size (n)")
    plt.ylabel("Learning Rate (r)")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # Use log scale for x-axis if there's a wide range of network sizes
    if max(network_sizes) / min(network_sizes) > 10:
        plt.xscale('log')
        # Add minor grid lines
        plt.grid(True, which='minor', linestyle=':', alpha=0.4)
    
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
    show_learning_rates: bool = True
) -> None:
    """
    Plot incorrect action probabilities for all agents in a single plot.
    
    Args:
        incorrect_probs: Dictionary mapping agent IDs to lists of incorrect action probabilities over time
        title: Title of the plot
        save_path: Path to save the figure (if None, figure is displayed)
        max_steps: Maximum number of steps to plot (if None, plots all steps)
        log_scale: Whether to use logarithmic scale for y-axis
        show_learning_rates: Whether to calculate and display learning rates in the legend
    """
    plt.figure(figsize=(12, 8))
    
    # Determine how many steps to plot
    if max_steps is None:
        # Find the agent with the most steps
        max_steps = max(len(probs) for probs in incorrect_probs.values())
    else:
        # Limit to the specified number of steps
        max_steps = min(max_steps, max(len(probs) for probs in incorrect_probs.values()))
    
    # Define a colormap for different agents
    colors = plt.cm.tab10(np.linspace(0, 1, len(incorrect_probs)))
    
    # Plot each agent's incorrect probabilities
    for i, (agent_id, probs) in enumerate(sorted(incorrect_probs.items())):
        # Limit to max_steps
        agent_probs = probs[:max_steps]
        time_steps = np.arange(len(agent_probs))
        
        # Plot with appropriate scale
        if log_scale:
            line, = plt.semilogy(time_steps, agent_probs, label=f"Agent {agent_id}", 
                               color=colors[i], linewidth=2)
        else:
            line, = plt.plot(time_steps, agent_probs, label=f"Agent {agent_id}", 
                           color=colors[i], linewidth=2)
        
        # Calculate and display learning rate if requested
        if show_learning_rates and len(agent_probs) >= 10:
            learning_rate = calculate_learning_rate(agent_probs)
            
            # Update the label with learning rate
            line.set_label(f"Agent {agent_id} (r = {learning_rate:.4f})")
            
            # Plot fitted exponential decay
            x = np.arange(len(agent_probs))
            initial_value = agent_probs[0]
            y = np.exp(-learning_rate * x) * initial_value
            
            if log_scale:
                plt.semilogy(x, y, '--', alpha=0.5, color=line.get_color())
            else:
                plt.plot(x, y, '--', alpha=0.5, color=line.get_color())
    
    # Set labels and title
    plt.xlabel("Time Steps")
    plt.ylabel("Incorrect Action Probability" + (" (log scale)" if log_scale else ""))
    plt.title(title)
    plt.grid(True, which="both" if log_scale else "major", ls="--", alpha=0.7)
    plt.legend(loc='best')
    
    # Set y-axis limits for better visualization
    if log_scale:
        plt.ylim(0.001, 1.0)
    else:
        plt.ylim(0, 1.0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    else:
        plt.show()
        
def plot_policy_vs_internal_states(
    belief_states: Dict[int, List[torch.Tensor]],
    latent_states: Dict[int, List[torch.Tensor]],
    action_probs: Dict[int, List[List[float]]],
    true_states: List[int] = None,
    num_states: int = 2,
    title: str = "Policy Stability vs Internal State Fluctuations",
    save_path: Optional[str] = None,
    max_steps: Optional[int] = None
) -> None:
    """
    Plot policy outputs alongside belief and latent states to compare stability.
    
    Args:
        belief_states: Dictionary mapping agent IDs to lists of belief state tensors
        latent_states: Dictionary mapping agent IDs to lists of latent state tensors
        action_probs: Dictionary mapping agent IDs to lists of action probability distributions
        true_states: List of true states for each time step (to highlight state changes)
        num_states: Number of possible states
        title: Title of the plot
        save_path: Path to save the figure (if None, figure is displayed)
        max_steps: Maximum number of steps to plot (if None, plots all steps)
    """
    num_agents = len(belief_states)
    
    # Determine how many steps to plot
    if max_steps is None:
        # Find the agent with the most steps
        max_steps = min(
            max(len(beliefs) for beliefs in belief_states.values()),
            max(len(latents) for latents in latent_states.values()),
            max(len(probs) for probs in action_probs.values())
        )
    else:
        # Limit to the specified number of steps
        max_steps = min(
            max_steps,
            max(len(beliefs) for beliefs in belief_states.values()),
            max(len(latents) for latents in latent_states.values()),
            max(len(probs) for probs in action_probs.values())
        )
    
    # Create a figure with subplots for each agent
    fig, axes = plt.subplots(
        num_agents, 
        3,  # Three columns: belief variance, latent variance, policy outputs
        figsize=(18, 5 * num_agents),
        gridspec_kw={'width_ratios': [1, 1, 1.5]},
        sharex=True
    )
    
    # If there's only one agent, make sure axes is a 2D array
    if num_agents == 1:
        axes = np.array([axes])
    
    # Plot each agent
    for j, agent_id in enumerate(sorted(belief_states.keys())):
        # Get data for this agent
        agent_beliefs = belief_states[agent_id][:max_steps]
        agent_latents = latent_states[agent_id][:max_steps]
        agent_action_probs = action_probs[agent_id][:max_steps]
        
        # Create time steps array
        time_steps = np.arange(len(agent_beliefs))
        
        # 1. Plot belief state variance over time
        ax_belief = axes[j, 0]
        
        # Calculate variance across belief dimensions at each time step
        belief_variances = []
        for belief in agent_beliefs:
            if isinstance(belief, torch.Tensor):
                belief_np = belief.detach().cpu().numpy().flatten()
            else:
                belief_np = np.array(belief).flatten()
            belief_variances.append(np.var(belief_np))
        
        ax_belief.plot(time_steps, belief_variances, 'b-', linewidth=2, label='Belief Variance')
        ax_belief.set_title(f"Agent {agent_id} Belief State Variance")
        ax_belief.set_ylabel("Variance")
        ax_belief.grid(True, linestyle='--', alpha=0.7)
        
        # 2. Plot latent state variance over time
        ax_latent = axes[j, 1]
        
        # Calculate variance across latent dimensions at each time step
        latent_variances = []
        for latent in agent_latents:
            if isinstance(latent, torch.Tensor):
                latent_np = latent.detach().cpu().numpy().flatten()
            else:
                latent_np = np.array(latent).flatten()
            latent_variances.append(np.var(latent_np))
        
        ax_latent.plot(time_steps, latent_variances, 'g-', linewidth=2, label='Latent Variance')
        ax_latent.set_title(f"Agent {agent_id} Latent State Variance")
        ax_latent.set_ylabel("Variance")
        ax_latent.grid(True, linestyle='--', alpha=0.7)
        
        # 3. Plot policy outputs (action probabilities) over time
        ax_policy = axes[j, 2]
        
        # Convert action probabilities to numpy array
        action_probs_np = np.array(agent_action_probs)
        
        # Plot each action probability
        for action in range(num_states):
            ax_policy.plot(
                time_steps, 
                action_probs_np[:, action], 
                label=f"Action {action} Prob",
                linewidth=2
            )
        
        ax_policy.set_title(f"Agent {agent_id} Action Probabilities")
        ax_policy.set_ylabel("Probability")
        ax_policy.set_ylim(0, 1)
        ax_policy.grid(True, linestyle='--', alpha=0.7)
        ax_policy.legend(loc='upper right')
        
        # Highlight true state changes if provided
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
            
            # Plot vertical lines at state changes on all subplots
            for t, state in state_changes[1:]:  # Skip the first one (initial state)
                for ax in axes[j]:
                    ax.axvline(
                        x=t,
                        color='red',
                        linestyle='--',
                        alpha=0.5,
                        label="State Change" if t == state_changes[1][0] and ax == axes[j, 0] else None
                    )
                    
                # Add state change label only on the policy plot
                axes[j, 2].text(
                    t, 
                    0.9,
                    f"→ State {state}",
                    horizontalalignment='left',
                    verticalalignment='top',
                    fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
                )
            
            # Add initial state label
            axes[j, 2].text(
                0, 
                0.9,
                f"State {plot_true_states[0]}",
                horizontalalignment='left',
                verticalalignment='top',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
            )
    
    # Set common x-axis label
    for ax in axes[-1]:
        ax.set_xlabel("Time Steps")
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
    max_dims: int = 10
) -> None:
    """
    Plot the latent state evolution of agents over time.
    
    Args:
        latent_states: Dictionary mapping agent IDs to lists of latent state tensors
        true_states: List of true states for each time step (to highlight state changes)
        num_states: Number of possible states
        title: Title of the plot
        save_path: Path to save the figure (if None, figure is displayed)
        max_steps: Maximum number of steps to plot (if None, plots all steps)
        max_dims: Maximum number of latent dimensions to plot
    """
    num_agents = len(latent_states)
    
    # Determine how many steps to plot
    if max_steps is None:
        # Find the agent with the most steps
        max_steps = max(len(latents) for latents in latent_states.values())
    else:
        # Limit to the specified number of steps
        max_steps = min(max_steps, max(len(latents) for latents in latent_states.values()))
    
    # Create a figure with subplots for each agent
    fig, axes = plt.subplots(
        num_agents, 
        1, 
        figsize=(12, 4 * num_agents),
        sharex=True
    )
    
    # If there's only one agent, make sure axes is an array
    if num_agents == 1:
        axes = np.array([axes])
    
    # Define colors for latent dimensions
    colors = plt.cm.tab20(np.linspace(0, 1, max_dims))
    
    # Plot each agent
    for j, agent_id in enumerate(sorted(latent_states.keys())):
        ax = axes[j]
        
        # Get latent states for this agent
        agent_latents = latent_states[agent_id][:max_steps]
        
        # Create time steps array
        time_steps = np.arange(len(agent_latents))
        
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
                f"State {plot_true_states[0]}",
                horizontalalignment='left',
                verticalalignment='top',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
            )
        
        # Set title and labels
        ax.set_title(f"Agent {agent_id} Latent State Evolution")
        ax.set_ylabel("Latent Value")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend with limited entries to avoid overcrowding
        if plot_dims > 10:
            ax.legend(loc='upper right', ncol=5, fontsize='small')
        else:
            ax.legend(loc='upper right', ncol=min(5, plot_dims), fontsize='small')
    
    # Set common x-axis label
    axes[-1].set_xlabel("Time Steps")
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
    max_dims: int = 10
) -> None:
    """
    Plot the belief state evolution of agents over time.
    
    Args:
        belief_states: Dictionary mapping agent IDs to lists of belief state tensors
        true_states: List of true states for each time step (to highlight state changes)
        num_states: Number of possible states
        title: Title of the plot
        save_path: Path to save the figure (if None, figure is displayed)
        max_steps: Maximum number of steps to plot (if None, plots all steps)
        max_dims: Maximum number of belief dimensions to plot
    """
    num_agents = len(belief_states)
    
    # Determine how many steps to plot
    if max_steps is None:
        # Find the agent with the most steps
        max_steps = max(len(beliefs) for beliefs in belief_states.values())
    else:
        # Limit to the specified number of steps
        max_steps = min(max_steps, max(len(beliefs) for beliefs in belief_states.values()))
    
    # Create a figure with subplots for each agent
    fig, axes = plt.subplots(
        num_agents, 
        1, 
        figsize=(12, 4 * num_agents),
        sharex=True
    )
    
    # If there's only one agent, make sure axes is an array
    if num_agents == 1:
        axes = np.array([axes])
    
    # Define colors for belief dimensions
    colors = plt.cm.tab20(np.linspace(0, 1, max_dims))
    
    # Plot each agent
    for j, agent_id in enumerate(sorted(belief_states.keys())):
        ax = axes[j]
        
        # Get belief states for this agent
        agent_beliefs = belief_states[agent_id][:max_steps]
        
        # Create time steps array
        time_steps = np.arange(len(agent_beliefs))
        
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
                f"State {plot_true_states[0]}",
                horizontalalignment='left',
                verticalalignment='top',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
            )
        
        # Set title and labels
        ax.set_title(f"Agent {agent_id} Belief State Evolution")
        ax.set_ylabel("Belief Value")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend with limited entries to avoid overcrowding
        if plot_dims > 10:
            ax.legend(loc='upper right', ncol=5, fontsize='small')
        else:
            ax.legend(loc='upper right', ncol=min(5, plot_dims), fontsize='small')
    
    # Set common x-axis label
    axes[-1].set_xlabel("Time Steps")
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
def plot_decision_boundaries(
    belief_states: Dict[int, List[torch.Tensor]],
    latent_states: Dict[int, List[torch.Tensor]],
    action_probs: Dict[int, List[List[float]]],
    true_states: List[int] = None,
    title: str = "Decision Boundaries in Internal State Space",
    save_path: Optional[str] = None,
    n_components: int = 2,
    plot_type: str = "both"  # "belief", "latent", or "both"
) -> None:
    """
    Visualize decision boundaries by projecting belief and latent states to 2D/3D using PCA
    and coloring by action probabilities.
    
    Args:
        belief_states: Dictionary mapping agent IDs to lists of belief state tensors
        latent_states: Dictionary mapping agent IDs to lists of latent state tensors
        action_probs: Dictionary mapping agent IDs to lists of action probability distributions
        true_states: List of true states for each time step
        title: Title of the plot
        save_path: Path to save the figure (if None, figure is displayed)
        n_components: Number of PCA components (2 or 3)
        plot_type: Which state type to plot ("belief", "latent", or "both")
    """
    from sklearn.decomposition import PCA
    import numpy as np
    
    if n_components not in [2, 3]:
        raise ValueError("n_components must be either 2 or 3")
    
    # Determine which plots to create
    plot_belief = plot_type in ["belief", "both"]
    plot_latent = plot_type in ["latent", "both"]
    
    # Set up the figure
    if plot_belief and plot_latent:
        fig = plt.figure(figsize=(18, 8))
        if n_components == 3:
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
        else:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
    else:
        fig = plt.figure(figsize=(10, 8))
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
    
    # Process each agent
    for agent_id, belief_sequence in belief_states.items():
        # Convert tensors to numpy arrays
        belief_data = np.array([b.cpu().numpy().flatten() for b in belief_sequence])
        latent_data = np.array([l.cpu().numpy().flatten() for l in latent_states[agent_id]])
        
        # Get action probabilities for state 1 (assuming binary state)
        # This will be used for coloring the points
        # Make sure we're getting the correct format - action_probs might be a list of lists or numpy arrays
        probs_state_1 = []
        for probs in action_probs[agent_id]:
            if isinstance(probs, (list, np.ndarray)) and len(probs) > 1:
                # If it's a list/array with multiple elements, take the second one (index 1)
                probs_state_1.append(float(probs[1]))
            elif isinstance(probs, (float, np.float32, np.float64)):
                # If it's already a single value, use it directly
                probs_state_1.append(float(probs))
            else:
                # Default case
                probs_state_1.append(0.5)
        
        # Make sure the length matches the belief/latent data
        if len(probs_state_1) > len(belief_data):
            probs_state_1 = probs_state_1[:len(belief_data)]
        elif len(probs_state_1) < len(belief_data):
            # Pad with the last value if needed
            last_val = probs_state_1[-1] if probs_state_1 else 0.5
            probs_state_1.extend([last_val] * (len(belief_data) - len(probs_state_1)))
            
        probs_state_1 = np.array(probs_state_1)
        
        # Apply PCA to reduce dimensionality
        if plot_belief and len(belief_data) > 0:
            belief_pca = PCA(n_components=n_components)
            belief_reduced = belief_pca.fit_transform(belief_data)
            
            # Create scatter plot with color based on action probability
            if plot_belief and plot_latent:
                ax_belief = ax1
            else:
                ax_belief = ax
                
            if n_components == 3:
                scatter = ax_belief.scatter(
                    belief_reduced[:, 0], 
                    belief_reduced[:, 1], 
                    belief_reduced[:, 2],
                    c=probs_state_1, 
                    cmap='coolwarm', 
                    s=30, 
                    alpha=0.7
                )
                ax_belief.set_xlabel('PC1')
                ax_belief.set_ylabel('PC2')
                ax_belief.set_zlabel('PC3')
                ax_belief.set_title(f'Agent {agent_id} Belief State Space\nColored by P(Action=1)')
            else:
                scatter = ax_belief.scatter(
                    belief_reduced[:, 0], 
                    belief_reduced[:, 1], 
                    c=probs_state_1, 
                    cmap='coolwarm', 
                    s=30, 
                    alpha=0.7
                )
                ax_belief.set_xlabel('PC1')
                ax_belief.set_ylabel('PC2')
                ax_belief.set_title(f'Agent {agent_id} Belief State Space\nColored by P(Action=1)')
                
                # Add colorbar
                plt.colorbar(scatter, ax=ax_belief, label='P(Action=1)')
                
                # Add decision boundary at 0.5 probability
                # Find the points closest to 0.5 probability
                boundary_points = []
                for i in range(len(probs_state_1)-1):
                    if (probs_state_1[i] < 0.5 and probs_state_1[i+1] >= 0.5) or \
                       (probs_state_1[i] >= 0.5 and probs_state_1[i+1] < 0.5):
                        boundary_points.append((belief_reduced[i], belief_reduced[i+1]))
                
                # Draw lines connecting boundary points
                for p1, p2 in boundary_points:
                    ax_belief.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.5)
        
        # Apply PCA to latent states
        if plot_latent and len(latent_data) > 0:
            latent_pca = PCA(n_components=n_components)
            latent_reduced = latent_pca.fit_transform(latent_data)
            
            # Make sure the length of probs_state_1 matches the latent data
            latent_probs_state_1 = probs_state_1
            if len(latent_probs_state_1) > len(latent_data):
                latent_probs_state_1 = latent_probs_state_1[:len(latent_data)]
            elif len(latent_probs_state_1) < len(latent_data):
                # Pad with the last value if needed
                last_val = latent_probs_state_1[-1] if latent_probs_state_1.size > 0 else 0.5
                latent_probs_state_1 = np.append(latent_probs_state_1, [last_val] * (len(latent_data) - len(latent_probs_state_1)))
            
            # Create scatter plot with color based on action probability
            if plot_belief and plot_latent:
                ax_latent = ax2
            else:
                ax_latent = ax
                
            if n_components == 3:
                scatter = ax_latent.scatter(
                    latent_reduced[:, 0], 
                    latent_reduced[:, 1], 
                    latent_reduced[:, 2],
                    c=latent_probs_state_1, 
                    cmap='coolwarm', 
                    s=30, 
                    alpha=0.7
                )
                ax_latent.set_xlabel('PC1')
                ax_latent.set_ylabel('PC2')
                ax_latent.set_zlabel('PC3')
                ax_latent.set_title(f'Agent {agent_id} Latent State Space\nColored by P(Action=1)')
            else:
                scatter = ax_latent.scatter(
                    latent_reduced[:, 0], 
                    latent_reduced[:, 1], 
                    c=latent_probs_state_1, 
                    cmap='coolwarm', 
                    s=30, 
                    alpha=0.7
                )
                ax_latent.set_xlabel('PC1')
                ax_latent.set_ylabel('PC2')
                ax_latent.set_title(f'Agent {agent_id} Latent State Space\nColored by P(Action=1)')
                
                # Add colorbar
                plt.colorbar(scatter, ax=ax_latent, label='P(Action=1)')
                
                # Add decision boundary at 0.5 probability
                # Find the points closest to 0.5 probability
                boundary_points = []
                for i in range(len(latent_probs_state_1)-1):
                    if (latent_probs_state_1[i] < 0.5 and latent_probs_state_1[i+1] >= 0.5) or \
                       (latent_probs_state_1[i] >= 0.5 and latent_probs_state_1[i+1] < 0.5):
                        boundary_points.append((latent_reduced[i], latent_reduced[i+1]))
                
                # Draw lines connecting boundary points
                for p1, p2 in boundary_points:
                    ax_latent.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.5)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()