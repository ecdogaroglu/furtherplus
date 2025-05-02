# FURTHER+ Social Learning Framework

This repository contains the implementation of FURTHER+ (Flexible Unsupervised Reinforcement learning with Transformative Hidden Embedding Representations), a deep reinforcement learning framework for social learning tasks.

## Overview

FURTHER+ is designed to study how agents learn from each other in social networks. The framework implements a multi-agent reinforcement learning environment where agents must learn to coordinate their actions based on partial observations and communication with neighbors in their social network.

The framework includes:

- A social learning environment with configurable network structures (complete, ring, star, random)
- A deep reinforcement learning agent architecture with belief and latent state representations
- Comprehensive visualization tools for analyzing agent behavior and learning dynamics
- Support for various experimental configurations and hyperparameter tuning

## Key Features

- **Flexible Network Topologies**: Configure different social network structures to study how information flows between agents
- **Internal State Representations**: Analyze how agents develop internal belief and latent state representations
- **Decision Boundary Visualization**: Understand how agents make decisions based on their internal states
- **Comparative Analysis**: Compare different model architectures and network configurations
- **Reproducible Experiments**: Save and load models, configurations, and results for reproducibility

## Installation

```bash
# Clone the repository
git clone https://github.com/ecdogaroglu/furtherplus.git
cd furtherplus

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python experiment.py --num-agents 2 --network-type complete --total-steps 10000
```

### Evaluation

```bash
python experiment.py --eval-only --load-model --num-agents 2 --total-steps 1000
```

### Visualization Options

The framework includes several visualization options for analyzing agent behavior:

```bash
# Enable internal state visualizations (belief states, latent states, decision boundaries)
python experiment.py --eval-only --load-model --plot-internal-states

# Visualize only belief states
python experiment.py --eval-only --load-model --plot-internal-states --plot-type belief

# Visualize only latent states
python experiment.py --eval-only --load-model --plot-internal-states --plot-type latent

# Visualize both belief and latent states (default)
python experiment.py --eval-only --load-model --plot-internal-states --plot-type both
```

### Visualization Types

1. **Belief States**: Visualizes the evolution of agent belief states over time
2. **Latent States**: Visualizes the evolution of agent latent states over time
3. **Policy vs. Internal States**: Shows how policy outputs relate to internal state fluctuations
4. **Decision Boundaries**: Uses PCA to project high-dimensional belief and latent states into 2D/3D space, coloring points by action probabilities to visualize decision boundaries

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--num-agents` | Number of agents in the network | 2 |
| `--network-type` | Network structure (complete, ring, star, random) | complete |
| `--total-steps` | Total number of steps for training/evaluation | 10000 |
| `--hidden-dim` | Hidden layer dimension | 64 |
| `--belief-dim` | Belief state dimension | 64 |
| `--latent-dim` | Latent space dimension | 16 |
| `--plot-internal-states` | Enable internal state visualizations | False |
| `--plot-type` | Type of internal state to plot (belief, latent, both) | both |
| `--eval-only` | Run evaluation only (no training) | False |
| `--load-model` | Load pre-trained model | False |
| `--save-model` | Save model after training | True |
| `--save-interval` | Steps between model saves | 1000 |
| `--exp-name` | Experiment name for saving results | "default" |

## Advanced Analysis

The decision boundary visualizations provide insights into how agents organize information internally and make decisions. By comparing models with different dimensions, you can analyze:

1. How decision boundaries form in the internal representation spaces
2. The relationship between model size and decision boundary clarity
3. The robustness of internal representations to noise and fluctuations
4. How information is encoded and transferred between agents in different network topologies

## Project Structure

```
furtherplus/
├── experiment.py          # Main experiment script
├── modules/
│   ├── agent.py           # Agent architecture implementation
│   ├── environment.py     # Environment implementation
│   ├── simulation.py      # Simulation logic
│   ├── metrics.py         # Evaluation metrics
│   ├── plotting.py        # Visualization tools
│   ├── comparison.py      # Comparative analysis tools
│   ├── args.py            # Command line argument parsing
│   └── utils.py           # Utility functions
├── results/               # Experiment results (ignored by git)
└── readme.md              # This file
```

## Requirements

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Matplotlib
- NetworkX
- scikit-learn (for PCA visualization)

## Citation

If you use FURTHER+ in your research, please cite:

```
@misc{dogaroglu2024furtherplus,
  author = {Dogaroglu, Ege},
  title = {FURTHER+: A Framework for Multi-Agent Social Learning with Internal Representations},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ecdogaroglu/furtherplus}}
}
```