#!/usr/bin/env python3
"""
FURTHER+ Social Learning Experiment Script (Modular Version)

This script runs experiments with FURTHER+ agents in a social learning environment.
It has been refactored into a modular structure for better organization and maintainability.
"""

import os
import torch
import numpy as np
from pathlib import Path

from modules.environment import SocialLearningEnvironment
from modules.args import parse_args
from modules.simulation import run_agents


def main():
    """Main function to run the experiment."""
    # Parse command-line arguments
    args = parse_args()
    
    # Set initial random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.train_then_evaluate:

        # Train
        args.num_episodes = 4
        args.horizon = 1000
        args.eval_only = False
        args.save_model = True
        args.use_gnn = True
        run_experiment(args)

        # Evaluate
        args.num_episodes = 4
        args.horizon = 1000
        args.eval_only = True
        args.load_model = 'auto'
        run_experiment(args)

    else:
        run_experiment(args)


def run_experiment(args):
    """Run the experiment with the given arguments."""
        # Create environment
    env = SocialLearningEnvironment(
        num_agents=args.num_agents,
        signal_accuracy=args.signal_accuracy,
        network_type=args.network_type,
        network_params={'density': args.network_density} if args.network_type == 'random' else None,
        horizon=args.horizon,
        seed=args.seed
    )
    
    # Determine model path for loading
    model_path = None
    if args.load_model:
        if args.load_model == 'auto':
            # Automatically find the final model directory
            model_path = Path(args.output_dir) / args.exp_name / f"network_{args.network_type}_agents_{args.num_agents}" / 'models' / 'final'
            print(f"Automatically loading final models from: {model_path}")
        else:
            # Use the specified path
            model_path = Path(args.load_model)
            print(f"Loading models from specified path: {model_path}")
    
    # Print episode information if training
    if not args.eval_only and args.num_episodes > 1:
        print(f"Training with {args.num_episodes} episodes, {args.horizon} steps per episode")
        print(f"True state will be randomly selected at the beginning of each episode with different seeds")
    
    # Run agents in training or evaluation mode
    if args.eval_only:
        run_agents(env, args, training=False, model_path=model_path)
    else:
        run_agents(env, args, training=True, model_path=model_path)

if __name__ == "__main__":
    main()