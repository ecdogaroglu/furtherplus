"""
Visualization functions for FURTHER+ experiments.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from modules.utils import (
    plot_learning_rate_by_network_size,
    plot_incorrect_action_probabilities,
    plot_belief_states,
    plot_latent_states,
    plot_policy_vs_internal_states,
    plot_decision_boundaries
)
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
        # Plot incorrect action probabilities
        plot_incorrect_action_probabilities(
            incorrect_probs=agent_incorrect_probs,
            title=f"Incorrect Action Probabilities ({args.network_type.capitalize()} Network, {env.num_agents} Agents)",
            save_path=str(output_dir / 'incorrect_action_probs.png'),
            log_scale=False,
            show_learning_rates=True
        )
    
    # Plot internal states if requested (only for evaluation)
    if not training and args.plot_internal_states:
        generate_internal_state_plots(metrics, env, args, output_dir)


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
            max_steps=min(1000, len(metrics['true_states']))  # Limit to 1000 steps for readability
        )
        print(f"Belief states plot saved to {output_dir / 'belief_states.png'}")
    
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
            max_steps=min(1000, len(metrics['true_states']))  # Limit to 1000 steps for readability
        )
        print(f"Latent states plot saved to {output_dir / 'latent_states.png'}")
    
    # Plot policy vs internal states
    if ('belief_states' in metrics and 'latent_states' in metrics and 'full_action_probs' in metrics):
        generate_policy_and_decision_boundary_plots(metrics, env, args, output_dir)


def generate_policy_and_decision_boundary_plots(metrics, env, args, output_dir):
    """Generate policy stability and decision boundary plots."""
    # Plot policy vs internal states
    plot_policy_vs_internal_states(
        belief_states=metrics['belief_states'],
        latent_states=metrics['latent_states'],
        action_probs=metrics['full_action_probs'],
        true_states=metrics['true_states'],
        num_states=env.num_states,
        title=f"Policy Stability vs Internal State Fluctuations ({args.network_type.capitalize()} Network, {env.num_agents} Agents)",
        save_path=str(output_dir / 'policy_vs_internal_states.png'),
        max_steps=min(1000, len(metrics['true_states']))  # Limit to 1000 steps for readability
    )
    
    print(f"Policy vs internal states plot saved to {output_dir / 'policy_vs_internal_states.png'}")
    
    # Plot decision boundaries using PCA
    plot_decision_boundaries(
        belief_states=metrics['belief_states'],
        latent_states=metrics['latent_states'],
        action_probs=metrics['full_action_probs'],
        true_states=metrics['true_states'],
        title=f"Decision Boundaries in Internal State Space ({args.network_type.capitalize()} Network, {env.num_agents} Agents)",
        save_path=str(output_dir / 'decision_boundaries_2d.png'),
        n_components=2,
        plot_type=args.plot_type
    )
    
    print(f"Decision boundaries (2D) plot saved to {output_dir / 'decision_boundaries_2d.png'}")
    
    # Also create a 3D version if we have enough data points
    if any(len(beliefs) > 3 for beliefs in metrics['belief_states'].values()):
        plot_decision_boundaries(
            belief_states=metrics['belief_states'],
            latent_states=metrics['latent_states'],
            action_probs=metrics['full_action_probs'],
            true_states=metrics['true_states'],
            title=f"Decision Boundaries in Internal State Space - 3D ({args.network_type.capitalize()} Network, {env.num_agents} Agents)",
            save_path=str(output_dir / 'decision_boundaries_3d.png'),
            n_components=3,
            plot_type=args.plot_type
        )
        
        print(f"Decision boundaries (3D) plot saved to {output_dir / 'decision_boundaries_3d.png'}")