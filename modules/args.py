"""
Command-line argument parsing for FURTHER+ experiments.
"""

import argparse
from modules.utils import get_best_device
import time


def parse_args():
    """Parse command-line arguments for the experiment."""
    parser = argparse.ArgumentParser(description="FURTHER+ Social Learning Experiment")
    
    # Environment parameters
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents")
    parser.add_argument("--network_type", type=str, default="complete", choices=["complete", "ring", "star", "random"], help="Type of social network")
    parser.add_argument("--network_density", type=float, default=0.5, help="Density of random network (if network_type is 'random')")
    parser.add_argument("--signal_accuracy", type=float, default=0.75, help="Accuracy of private signals")
    
    # Experiment parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--horizon", type=int, default=100, help="Number of time steps per episode")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for logs and models")
    parser.add_argument("--exp_name", type=str, default=f"brandl_validation_{time.strftime('%Y%m%d_%H%M%S')}", help="Experiment name")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only (no training)")
    
    # Agent parameters
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for neural networks")
    parser.add_argument("--belief_dim", type=int, default=64, help="Belief state dimension")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent state dimension")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--buffer_capacity", type=int, default=10000, help="Replay buffer capacity")
    parser.add_argument("--use_gnn", action="store_true", help="Use GNN for inference instead of encoder-decoder")
    parser.add_argument("--sequence_length", type=int, default=8, help="Sequence length for training")
    
    # Enhanced features to reduce state overfitting
    parser.add_argument("--use_state_augmentation", action="store_true", help="Occasionally switch states during training to increase diversity")
    parser.add_argument("--augmentation_frequency", type=int, default=10, help="How often to potentially switch states during training")
    parser.add_argument("--synthetic_state_prob", type=float, default=0.2, help="Probability of switching to a synthetic state during augmentation")
    
    # Training control
    parser.add_argument("--discount_factor", type=float, default=0.99, help="Discount factor for RL")
    parser.add_argument("--entropy_weight", type=float, default=0.01, help="Weight for entropy regularization")
    parser.add_argument("--train_interval", type=int, default=1, help="Steps between training updates")
    parser.add_argument("--target_update_rate", type=float, default=0.005, help="Soft update rate for target networks")
    parser.add_argument("--train_steps", type=int, default=1, help="Number of training steps per update")
    
    # Model saving/loading
    parser.add_argument("--save_model", action="store_true", help="Save trained models")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load models from, or 'auto' to load the final model")
    
    # Visualization
    parser.add_argument("--use_tex", action="store_true", help="Use LaTeX for plot rendering")
    
    args = parser.parse_args()
    return args