"""
Visualization functions for FURTHER+ experiments.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from modules.utils import (
    plot_incorrect_action_probabilities,
    plot_belief_distributions
    )
from modules.metrics import process_incorrect_probabilities, print_debug_info_for_plotting


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
    
    # Plot internal states if requested (for both training and evaluation)
    if args.plot_internal_states:
        # Check if we have the necessary data
        has_belief_data = ('belief_states' in metrics and 
                          any(len(beliefs) > 0 for beliefs in metrics['belief_states'].values()))
        has_latent_data = ('latent_states' in metrics and 
                          any(len(latents) > 0 for latents in metrics['latent_states'].values()))
        
        if has_belief_data or has_latent_data:
            print(f"Generating internal state plots in {'training' if training else 'evaluation'} mode...")
            
            # Use episodic metrics if available, otherwise use combined metrics
            if episodic_metrics and 'episodes' in episodic_metrics and episodic_metrics['episodes']:
                print("Using episode-separated metrics for plotting...")
                for i, episode_metrics in enumerate(episodic_metrics['episodes']):
                    episode_dir = output_dir / f'episode_{i+1}'
                    episode_dir.mkdir(exist_ok=True)
                    generate_internal_state_plots(episode_metrics, env, args, episode_dir, episode_num=i+1)
                
                # Also generate combined plots for comparison
                generate_internal_state_plots(metrics, env, args, output_dir)
            else:
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
    

