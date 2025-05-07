"""
Visualization functions for FURTHER+ experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import torch
import os
from scipy import stats

from modules.metrics import process_incorrect_probabilities, print_debug_info_for_plotting
from modules.utils import calculate_learning_rate

def generate_plots(metrics, env, args, output_dir, training, episodic_metrics=None):
    """
    Generate plots from experiment results.
    
    Args:
        metrics: Combined metrics dictionary
        env: Environment object
        args: Command-line arguments
        output_dir: Directory to save plots
        training: Whether this is training or evaluation
        episodic_metrics: Optional dictionary with episode-separated metrics
    """
    # Process incorrect probabilities for plotting
    agent_incorrect_probs = process_incorrect_probabilities(metrics, env.num_agents)
    
    # Debug information about processed data
    print_debug_info_for_plotting(agent_incorrect_probs)
    
    # Check if we have data to plot
    has_data = any(len(probs) > 0 for probs in agent_incorrect_probs.values())
    
    if not has_data:
        print("WARNING: No data to plot! Skipping plot generation.")
        create_empty_plot(output_dir)
    else:
        # Plot incorrect action probabilities with episode separation
        plot_incorrect_action_probabilities(
            incorrect_probs=agent_incorrect_probs,
            title=f"Incorrect Action Probabilities ({args.network_type.capitalize()} Network, {env.num_agents} Agents)",
            save_path=str(output_dir / 'incorrect_action_probs.png'),
            log_scale=False,
            show_learning_rates=True,
            episode_length=args.horizon  # Use horizon directly
        )
        
        # Plot mean incorrect action probabilities with confidence intervals if we have multiple episodes
        if episodic_metrics and 'episodes' in episodic_metrics and len(episodic_metrics['episodes']) > 1:
            plot_mean_incorrect_action_probabilities_with_ci(
                episodic_metrics=episodic_metrics,
                title=f"Mean Incorrect Action Probabilities with 95% CI ({args.network_type.capitalize()} Network, {env.num_agents} Agents)",
                save_path=str(output_dir / 'mean_incorrect_action_probs_with_ci.png'),
                log_scale=False,
                episode_length=args.horizon
            )
        
        # Plot agent actions if available
        if 'agent_actions' in metrics and any(len(actions) > 0 for actions in metrics['agent_actions'].values()):
            plot_agent_actions(
                actions=metrics['agent_actions'],
                true_states=metrics['true_states'],
                title=f"Agent Actions Over Time ({args.network_type.capitalize()} Network, {env.num_agents} Agents)",
                save_path=str(output_dir / 'agent_actions.png'),
                episode_length=args.horizon,
                num_episodes=args.num_episodes if training else 1
            )
    
    # Plot internal states if requested (for both training and evaluation)
    if args.plot_internal_states:
        # Check if we have the necessary data
        has_belief_data = ('belief_states' in metrics and 
                          any(len(beliefs) > 0 for beliefs in metrics['belief_states'].values()))
        has_latent_data = ('latent_states' in metrics and 
                          any(len(latents) > 0 for latents in metrics['latent_states'].values()))
        
        if has_belief_data or has_latent_data:
            print(f"Generating internal state plots in {'training' if training else 'evaluation'} mode...")
            

            generate_internal_state_plots(metrics, env, args, output_dir)
        else:
            print("No internal state data available for plotting. Make sure you're running in evaluation mode or collecting internal states during training.")


def create_empty_plot(output_dir):
    """Create an empty plot with a message when no data is available."""
    plt.figure(figsize=(12, 8))
    plt.text(0.5, 0.5, "No data available for plotting", 
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=20)
    plt.savefig(str(output_dir / 'incorrect_action_probs.png'))
    plt.close()


def plot_mean_incorrect_action_probabilities_with_ci(
    episodic_metrics: Dict,
    title: str = "Mean Incorrect Action Probabilities with 95% CI",
    save_path: Optional[str] = None,
    log_scale: bool = False,
    episode_length: Optional[int] = None
) -> None:
    """
    Plot mean incorrect action probabilities across episodes with 95% confidence intervals.
    
    Args:
        episodic_metrics: Dictionary containing episode-separated metrics
        title: Title of the plot
        save_path: Path to save the figure (if None, figure is displayed)
        log_scale: Whether to use logarithmic scale for y-axis
        episode_length: Length of each episode
    """
    if not episodic_metrics or 'episodes' not in episodic_metrics or not episodic_metrics['episodes']:
        print("No episodic metrics available for plotting mean incorrect action probabilities with CI.")
        return
    
    # Extract episodes data
    episodes = episodic_metrics['episodes']
    num_episodes = len(episodes)
    
    if num_episodes < 2:
        print("Need at least 2 episodes to plot mean with confidence intervals.")
        return
    
    # Get the number of agents from the first episode
    if 'action_probs' not in episodes[0]:
        print("No action probabilities found in episodic metrics.")
        return
    
    agent_ids = list(episodes[0]['action_probs'].keys())
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Define a colormap for different agents
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_ids)))
    
    # For each agent, collect data across episodes and calculate mean and CI
    for i, agent_id in enumerate(sorted(agent_ids, key=int)):
        agent_color = colors[i]
        
        # Collect data for this agent across all episodes
        # We need to ensure all episodes have the same length for this agent
        min_length = min(len(episodes[ep_idx]['action_probs'][agent_id]) for ep_idx in range(num_episodes))
        
        if min_length == 0:
            print(f"Agent {agent_id} has no data in at least one episode. Skipping.")
            continue
        
        # Create a 2D array where each row is an episode and each column is a time step
        agent_data = np.zeros((num_episodes, min_length))
        for ep_idx in range(num_episodes):
            agent_data[ep_idx, :] = episodes[ep_idx]['action_probs'][agent_id][:min_length]
        
        # Calculate mean and standard deviation across episodes for each time step
        mean_probs = np.mean(agent_data, axis=0)
        std_probs = np.std(agent_data, axis=0)
        
        # Calculate 95% confidence interval
        # For small sample sizes, use t-distribution
        t_value = stats.t.ppf(0.975, num_episodes - 1)  # 95% CI (two-tailed)
        ci = t_value * std_probs / np.sqrt(num_episodes)
        
        # Create time steps array
        time_steps = np.arange(min_length)
        
        # Plot mean line
        if log_scale:
            line, = plt.semilogy(time_steps, mean_probs, 
                              label=f"Agent {agent_id}",
                              color=agent_color,
                              linewidth=2)
        else:
            line, = plt.plot(time_steps, mean_probs, 
                          label=f"Agent {agent_id}",
                          color=agent_color,
                          linewidth=2)
        
        # Plot confidence interval
        plt.fill_between(time_steps, 
                         mean_probs - ci, 
                         mean_probs + ci, 
                         color=agent_color, 
                         alpha=0.2)
        
        # Calculate and display learning rate for the mean curve
        if len(mean_probs) >= 10:
            learning_rate = calculate_learning_rate(mean_probs)
            
            # Update the label with learning rate
            line.set_label(f"Agent {agent_id} (r = {learning_rate:.4f})")
            
            # Plot fitted exponential decay
            x = np.arange(len(mean_probs))
            initial_value = mean_probs[0]
            y = np.exp(-learning_rate * x) * initial_value
            
            if log_scale:
                plt.semilogy(x, y, '--', alpha=0.5, color=line.get_color())
            else:
                plt.plot(x, y, '--', alpha=0.5, color=line.get_color())
    
    # Set labels and grid
    plt.xlabel("Time Steps")
    plt.ylabel("Incorrect Action Probability" + (" (log scale)" if log_scale else ""))
    plt.grid(True, which="both" if log_scale else "major", ls="--", alpha=0.7)
    
    # Set y-axis limits for better visualization
    if log_scale:
        plt.ylim(0.001, 1.0)
    else:
        plt.ylim(0, 1.0)
    
    # Add legend
    plt.legend(loc='best')
    
    # Set title
    plt.title(title)
    
    # Add text about number of episodes
    plt.figtext(0.01, 0.01, f"Based on {num_episodes} episodes", fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save the figure
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved mean incorrect action probabilities plot with CI to {save_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        plt.show()
    
    plt.close()


def generate_internal_state_plots(metrics, env, args, output_dir, episode_num=None):
    """
    Generate plots of internal agent states during evaluation.
    
    Args:
        metrics: Dictionary of metrics
        env: Environment object
        args: Command-line arguments
        output_dir: Directory to save plots
        episode_num: Optional episode number for episode-specific plots
    """
    
    # Plot belief distributions if available
    if ('belief_distributions' in metrics and 
        any(len(beliefs) > 0 for beliefs in metrics['belief_distributions'].values())):
        
        # Add episode number to title if provided
        episode_suffix = f" - Episode {episode_num}" if episode_num is not None else ""
        
        plot_belief_distributions(
            belief_distributions=metrics['belief_distributions'],
            true_states=metrics['true_states'],
            title=f"Belief Distributions Evolution ({args.network_type.capitalize()} Network, {env.num_agents} Agents){episode_suffix}",
            save_path=str(output_dir / 'belief_distributions.png'),
            episode_length=args.horizon,  # Use horizon directly
            num_episodes=1 if episode_num is not None else args.num_episodes  # Single episode if episode_num is provided
        )
    
    # Plot opponent belief distributions if available
    if ('opponent_belief_distributions' in metrics and 
        any(len(beliefs) > 0 for beliefs in metrics['opponent_belief_distributions'].values())):
        
        # Add episode number to title if provided
        episode_suffix = f" - Episode {episode_num}" if episode_num is not None else ""
        
        plot_belief_distributions(
            belief_distributions=metrics['opponent_belief_distributions'],
            true_states=metrics['true_states'],
            title=f"Opponent Belief Distributions Evolution ({args.network_type.capitalize()} Network, {env.num_agents} Agents){episode_suffix}",
            save_path=str(output_dir / 'opponent_belief_distributions.png'),
            episode_length=args.horizon,  # Use horizon directly
            num_episodes=1 if episode_num is not None else args.num_episodes  # Single episode if episode_num is provided
        )
    
    # Plot agent actions if available
    if ('agent_actions' in metrics and 
        any(len(actions) > 0 for actions in metrics['agent_actions'].values())):
        
        # Add episode number to title if provided
        episode_suffix = f" - Episode {episode_num}" if episode_num is not None else ""
        
        plot_agent_actions(
            actions=metrics['agent_actions'],
            true_states=metrics['true_states'],
            title=f"Agent Actions Over Time ({args.network_type.capitalize()} Network, {env.num_agents} Agents){episode_suffix}",
            save_path=str(output_dir / 'agent_actions.png'),
            episode_length=args.horizon,  # Use horizon directly
            num_episodes=1 if episode_num is not None else args.num_episodes  # Single episode if episode_num is provided
        )
    

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
    total_steps = episode_length
    
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


def plot_agent_actions(actions, true_states, title, save_path=None, episode_length=None, num_episodes=1):
    """
    Plot the actions taken by agents over time.
    
    Args:
        actions: Dictionary mapping agent IDs to lists of actions
        true_states: List of true states at each time step
        title: Plot title
        save_path: Path to save the plot (if None, displays the plot)
        episode_length: Length of each episode (for marking episode boundaries)
        num_episodes: Number of episodes in the data
    """
    # Determine number of agents to plot (limit to 8 for readability)
    agent_ids = list(actions.keys())
    num_agents_to_plot = min(8, len(agent_ids))
    selected_agents = agent_ids[:num_agents_to_plot]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot true state
    if true_states:
        plt.plot(true_states, 'k-', linewidth=2, label='True State')
    
    # Plot actions for each agent
    for agent_id in selected_agents:
        agent_actions = actions[agent_id]
        if agent_actions:
            plt.plot(agent_actions, 'o-', markersize=3, alpha=0.7, label=f'Agent {agent_id}')
    
    # Add episode boundaries if episode_length is provided
    if episode_length and num_episodes > 1:
        for ep in range(1, num_episodes):
            plt.axvline(x=ep*episode_length, color='gray', linestyle='--', alpha=0.7)
    
    # Set labels and title
    plt.xlabel('Time Step')
    plt.ylabel('Action / State')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Set y-ticks to match state indices if we can determine the number of states
    if true_states:
        num_states = max(true_states) + 1
        plt.yticks(range(num_states), [f'State {s}' for s in range(num_states)])
    
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save the figure
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved agent actions plot to {save_path}")
            plt.close()
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")
            plt.close()
    else:
        plt.show()
