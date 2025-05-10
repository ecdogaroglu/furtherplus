"""
Command-line argument parsing for POLARIS experiments.
"""

import argparse
from modules.utils import get_best_device


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate POLARIS in social learning")
    
    # Environment parameters
    parser.add_argument('--num-agents', type=int, default=2, help='Number of agents')
    parser.add_argument('--signal-accuracy', type=float, default=0.75, help='Accuracy of private signals')
    parser.add_argument('--network-type', type=str, default='complete', 
                        choices=['complete', 'ring', 'star', 'random'], help='Network structure')
    parser.add_argument('--network-density', type=float, default=0.5, 
                        help='Density for random networks')
    parser.add_argument('--horizon', type=int, default=1000, help='Total number of steps')
    parser.add_argument('--num-episodes', type=int, default=1, 
                        help='Number of episodes for training (true state is reset between episodes)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--buffer-capacity', type=int, default=1000, help='Replay buffer capacity')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--update-interval', type=int, default=10, help='Steps between updates')
    
    # Agent hyperparameters
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--belief-dim', type=int, default=256, help='Belief state dimension')
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent space dimension')
    parser.add_argument('--discount-factor', type=float, default=0.9, 
                        help='Discount factor (0 = average reward)')
    parser.add_argument('--entropy-weight', type=float, default=0.5, help='Entropy bonus weight')
    parser.add_argument('--kl-weight', type=float, default=10, help='KL weight for inference')
    
    # Model type selection
    parser.add_argument('--use-gnn', action='store_true', 
                        help='Use Graph Neural Network with temporal attention')
    parser.add_argument('--gnn-layers', type=int, default=2,
                        help='Number of GNN layers')
    parser.add_argument('--attn-heads', type=int, default=4,
                        help='Number of attention heads in GNN')
    parser.add_argument('--temporal-window', type=int, default=5,
                        help='Temporal window size for GNN memory')
                        
    # EWC parameters
    parser.add_argument('--use-ewc', action='store_true',
                        help='Use Elastic Weight Consolidation to prevent catastrophic forgetting')
    parser.add_argument('--ewc-importance', type=float, default=50.0,
                        help='Importance factor for EWC penalty (lambda)')
    parser.add_argument('--ewc-online', action='store_true',
                        help='Use online EWC instead of standard EWC')
    
    # Experiment settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default=get_best_device(), 
                        help='Device to use (cuda, mps, or cpu)')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--exp-name', type=str, default='brandl_validation', 
                        help='Experiment name')
    parser.add_argument('--save-model', action='store_true', help='Save agent models')
    parser.add_argument('--load-model', type=str, nargs='?', const='auto',
                        help='Load model (if specified without path: load final model, with path: load from specified path)')
    
    # Network size comparison
    parser.add_argument('--compare-sizes', action='store_true', 
                        help='Compare different network sizes')
    parser.add_argument('--network-sizes', type=str, default='2,4,6,10,20', 
                        help='Comma-separated list of network sizes')
    
    # Training vs evaluation
    parser.add_argument('--eval-only', action='store_true', 
                        help='Only run evaluation, no training')
    parser.add_argument('--train-then-evaluate', action='store_true', 
                        help='Train for 4 episodes with horizon 1000, then evaluate for 4 episodes with horizon 1000')
    # Visualization options
    parser.add_argument('--plot-internal-states', action='store_true',
                        help='Generate detailed visualizations of internal states (belief, latent, decision boundaries)')
    parser.add_argument('--plot-type', type=str, default='both', choices=['belief', 'latent', 'both'],
                        help='Type of internal state to plot (belief, latent, or both)')
    parser.add_argument('--latex-style', action='store_true',
                        help='Use LaTeX-style formatting for plots (publication quality)')
    parser.add_argument('--use-tex', action='store_true',
                        help='Use actual LaTeX rendering for text in plots (requires LaTeX installation)')
    
    args = parser.parse_args()
    
    # Process network sizes
    if hasattr(args, 'network_sizes'):
        args.network_sizes_list = [int(size) for size in args.network_sizes.split(',')]
    
    return args