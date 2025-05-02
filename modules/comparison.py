"""
Network size comparison functionality for FURTHER+ experiments.
"""

import torch
import numpy as np
from pathlib import Path
from modules.environment import SocialLearningEnvironment
from modules.utils import plot_learning_rate_by_network_size
from modules.simulation import run_agents


def compare_network_sizes(args):
    """Compare learning rates across different network sizes."""
    print(f"Comparing network sizes: {args.network_sizes_list}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set up output directory
    output_dir = Path(args.output_dir) / args.exp_name / f"network_size_comparison_{args.network_type}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store results for each network size
    results = {}
    
    # Track learning rates
    avg_learning_rates = []
    fastest_learning_rates = []
    slowest_learning_rates = []
    
    # Get theoretical bounds (same for all sizes)
    sample_env = SocialLearningEnvironment(
        num_agents=args.network_sizes_list[0],
        signal_accuracy=args.signal_accuracy,
        network_type=args.network_type,
        network_params={'density': args.network_density} if args.network_type == 'random' else None,
        total_steps=100,  # Doesn't matter for bounds
        seed=args.seed
    )
    
    autarky_rate = sample_env.get_autarky_rate()
    bound_rate = sample_env.get_bound_rate()
    coordination_rate = sample_env.get_coordination_rate()
    
    theoretical_bounds = {
        'autarky_rate': autarky_rate,
        'coordination_rate': coordination_rate,
        'bound_rate': bound_rate
    }
    
    # Run for each network size
    for network_size in args.network_sizes_list:
        print(f"\nEvaluating network size {network_size}...")
        
        # Create environment
        env = SocialLearningEnvironment(
            num_agents=network_size,
            signal_accuracy=args.signal_accuracy,
            network_type=args.network_type,
            network_params={'density': args.network_density} if args.network_type == 'random' else None,
            total_steps=args.total_steps,
            seed=args.seed
        )
        
        # Run agents
        learning_rates, metrics = run_agents(
            env=env,
            args=args,
            training=True,
            model_path=None
        )
        
        # Store results
        results[network_size] = {
            'learning_rates': learning_rates,
            'metrics': metrics
        }
        
        # Track summary statistics
        avg_rate = np.mean(list(learning_rates.values()))
        fastest_rate = max(learning_rates.values())
        slowest_rate = min(learning_rates.values())
        
        avg_learning_rates.append((network_size, avg_rate))
        fastest_learning_rates.append((network_size, fastest_rate))
        slowest_learning_rates.append((network_size, slowest_rate))
        
        print(f"Network size {network_size}: Avg rate = {avg_rate:.4f}, "
              f"Fastest = {fastest_rate:.4f}, Slowest = {slowest_rate:.4f}")
    
    # Generate comparison plot
    plot_learning_rate_by_network_size(
        avg_rates=avg_learning_rates,
        fastest_rates=fastest_learning_rates,
        slowest_rates=slowest_learning_rates,
        autarky_rate=autarky_rate,
        bound_rate=bound_rate,
        coordination_rate=coordination_rate,
        title=f"Learning Rates by Network Size ({args.network_type.capitalize()} Network)",
        save_path=str(output_dir / 'learning_rates_by_size.png')
    )
    
    print(f"\nComparison complete. Results saved to {output_dir}")
    return results