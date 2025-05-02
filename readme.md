# FURTHER+ Social Learning Framework

This repository contains the implementation of FURTHER+ (Flexible Unsupervised Reinforcement learning with Transformative Hidden Embedding Representations), a deep reinforcement learning framework for social learning tasks.

## Overview

FURTHER+ is designed to study how agents learn from each other in social networks. The framework includes:

- A social learning environment with configurable network structures
- A deep reinforcement learning agent architecture with belief and latent state representations
- Visualization tools for analyzing agent behavior and learning dynamics

## Usage

### Basic Training

```bash
python experiment_script.py --num-agents 2 --network-type complete --total-steps 10000
```

### Evaluation

```bash
python experiment_script.py --eval-only --load-model --num-agents 2 --total-steps 1000
```

### Visualization Options

The framework includes several visualization options for analyzing agent behavior:

```bash
# Enable internal state visualizations (belief states, latent states, decision boundaries)
python experiment_script.py --eval-only --load-model --plot-internal-states

# Visualize only belief states
python experiment_script.py --eval-only --load-model --plot-internal-states --plot-type belief

# Visualize only latent states
python experiment_script.py --eval-only --load-model --plot-internal-states --plot-type latent

# Visualize both belief and latent states (default)
python experiment_script.py --eval-only --load-model --plot-internal-states --plot-type both
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

## Advanced Analysis

The decision boundary visualizations provide insights into how agents organize information internally and make decisions. By comparing models with different dimensions, you can analyze:

1. How decision boundaries form in the internal representation spaces
2. The relationship between model size and decision boundary clarity
3. The robustness of internal representations to noise and fluctuations

## Requirements

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Matplotlib
- NetworkX
- scikit-learn (for PCA visualization)