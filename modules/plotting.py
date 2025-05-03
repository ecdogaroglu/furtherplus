"""
Visualization functions for FURTHER+ experiments.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from modules.utils import (
    plot_incorrect_action_probabilities,
    plot_belief_states,
    plot_belief_distributions,
    plot_latent_states)
from modules.metrics import process_incorrect_probabilities, print_debug_info_for_plotting


def generate_plots(metrics, env, args, output_dir, training):
    """Generate plots from experiment results."""
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
            episode_length=args.horizon  # Pass the episode length to separate episodes
        )
    
    # Plot internal states if requested ( if argfor both training and evaluation)
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


def generate_internal_state_plots(metrics, env, args, output_dir):
    """Generate plots of internal agent states during evaluation."""
    # Plot belief states
    if ('belief_states' in metrics and 
        any(len(beliefs) > 0 for beliefs in metrics['belief_states'].values()) and 
        (args.plot_type == 'belief' or args.plot_type == 'both')):
        
        plot_belief_states(
            belief_states=metrics['belief_states'],
            true_states=metrics['true_states'],
            num_states=env.num_states,
            title=f"Belief States Evolution ({args.network_type.capitalize()} Network, {env.num_agents} Agents)",
            save_path=str(output_dir / 'belief_states.png'),
            max_steps=min(1000, len(metrics['true_states'])),  # Limit to 1000 steps for readability
            episode_length=args.horizon  # Pass episode length to separate episodes
        )
        print(f"Belief states plot saved to {output_dir / 'belief_states.png'}")
    
    # Plot belief distributions if available
    if ('belief_distributions' in metrics and 
        any(len(beliefs) > 0 for beliefs in metrics['belief_distributions'].values())):
        
        plot_belief_distributions(
            belief_distributions=metrics['belief_distributions'],
            true_states=metrics['true_states'],
            title=f"Belief Distributions Evolution ({args.network_type.capitalize()} Network, {env.num_agents} Agents)",
            save_path=str(output_dir / 'belief_distributions.png'),
            max_steps=min(1000, len(metrics['true_states'])),  # Limit to 1000 steps for readability
            episode_length=args.horizon  # Pass episode length to separate episodes
        )
        print(f"Belief distributions plot saved to {output_dir / 'belief_distributions.png'}")
    
    # Plot latent states
    if ('latent_states' in metrics and 
        any(len(latents) > 0 for latents in metrics['latent_states'].values()) and 
        (args.plot_type == 'latent' or args.plot_type == 'both')):
        
        plot_latent_states(
            latent_states=metrics['latent_states'],
            true_states=metrics['true_states'],
            num_states=env.num_states,
            title=f"Latent States Evolution ({args.network_type.capitalize()} Network, {env.num_agents} Agents)",
            save_path=str(output_dir / 'latent_states.png'),
            max_steps=min(1000, len(metrics['true_states'])),  # Limit to 1000 steps for readability
            episode_length=args.horizon  # Pass episode length to separate episodes
        )
        print(f"Latent states plot saved to {output_dir / 'latent_states.png'}")
    
