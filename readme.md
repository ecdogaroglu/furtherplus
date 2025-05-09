# POLARIS Social Learning Framework

A modular deep reinforcement learning framework for multi-agent social learning and coordination, extending the [FURTHER](https://github.com/dkkim93/further) framework by Dong-Ki Kim et al.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.8+](https://img.shields.io/badge/pytorch-1.8+-orange.svg)](https://pytorch.org/)

## Overview

POLARIS (Partially Observable Learning with Active Reinforcement In Social environments) is a framework designed to study how agents learn from each other in social networks. It implements a multi-agent reinforcement learning environment where agents must coordinate their actions based on partial observations and neighbor interactions within configurable network topologies.

This framework extends the original [FURTHER](https://arxiv.org/pdf/2203.03535.pdf) framework (Kim et al., NeurIPS 2022) with enhanced capabilities specifically for social learning scenarios. While the original FURTHER focused on influencing long-term behavior in multiagent reinforcement learning, POLARIS adds specialized components for belief modeling, information propagation through networks, and advantage-weighted training.

The framework enables researchers to investigate emergent social learning phenomena, information diffusion dynamics, and collective intelligence in artificial agent networks.

## Key Features

- **Modular Architecture**: Clean separation of environment, agent models, and simulation logic
- **Customizable Network Topologies**: Experiment with complete, ring, star, or random network structures
- **Sophisticated Agent Architecture**:
  - Transformer-based belief processor for tracking agent internal state
  - Latent variable model for inferring other agents' states
  - Advantage-weighted belief learning mechanism
  - Graph Neural Network option for network-aware representations
- **Neural Architecture Options**:
  - Transformer-based belief processing
  - Graph Neural Networks with temporal attention
  - Traditional encoder-decoder architectures
- **Comprehensive Analysis Tools**:
  - Real-time learning rate estimation
  - Belief and latent state visualization
  - Theoretical bounds comparison
  - Publication-quality plots

## Neural Architecture

POLARIS offers multiple neural architectures that can be selected based on research needs:

### Transformer-Based Belief Processor

The default belief processor uses a Transformer architecture with self-attention to better capture temporal dependencies in observed signals and neighbor actions. Key advantages include:

- Better handling of long-range dependencies in observation sequences
- Improved information extraction from neighbor behaviors
- Superior performance in complex social learning scenarios

### Graph Neural Networks with Temporal Attention

For network-aware representations, POLARIS implements a Graph Neural Network with temporal attention that can be enabled with `--use-gnn`:

- Represents agents and their interactions as nodes and edges in a graph
- Maintains a temporal memory of past states and applies attention
- Adapts to different network topologies automatically
- Configurable with `--gnn-layers`, `--attn-heads`, and `--temporal-window`

## Installation

```bash
# Clone the repository
git clone https://github.com/ecdogaroglu/polaris.git
cd polaris

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Training

```bash
python experiment.py --num-agents 4 --network-type complete --horizon 1000
```

### Evaluation with Pre-trained Models

```bash
python experiment.py --eval-only --load-model --num-agents 4 --horizon 500
```

## Usage Guide

### Training Configuration

POLARIS offers extensive customization for training agents:

```bash
python experiment.py \
  --num-agents 6 \
  --network-type random \
  --network-density 0.5 \
  --signal-accuracy 0.75 \
  --horizon 2000 \
  --num-episodes 5 \
  --hidden-dim 128 \
  --belief-dim 128 \
  --latent-dim 64 \
  --discount-factor 0.99 \
  --entropy-weight 0.01 \
  --kl-weight 0.1 \
  --learning-rate 0.001 \
  --batch-size 64 \
  --buffer-capacity 5000 \
  --update-interval 1 \
  --use-gnn \
  --gnn-layers 2 \
  --attn-heads 4 \
  --temporal-window 5 \
  --exp-name "custom_experiment" \
  --save-model
```

### Visualization Options

The framework provides rich visualization capabilities for analyzing agent behavior:

```bash
# Enable internal state visualizations
python experiment.py --eval-only --load-model --plot-internal-states

# Specify visualization type
python experiment.py --eval-only --load-model --plot-internal-states --plot-type belief

# Generate publication-quality plots
python experiment.py --eval-only --load-model --latex-style

# Use LaTeX rendering for text (requires LaTeX installation)
python experiment.py --eval-only --load-model --latex-style --use-tex
```

### Visualization Types

1. **Incorrect Action Probabilities**: Shows how quickly agents learn the correct state
2. **Belief Distributions**: Visualizes the evolution of agent belief states over time
3. **Agent Actions**: Plots the actions taken by agents compared to the true state
4. **Latent Representations**: Shows how agents model other agents in their latent space

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--num-agents` | Number of agents in the network | 2 |
| `--signal-accuracy` | Accuracy of private signals | 0.75 |
| `--network-type` | Network structure (complete, ring, star, random) | complete |
| `--network-density` | Density for random networks | 0.5 |
| `--horizon` | Total number of steps per episode | 1000 |
| `--num-episodes` | Number of episodes for training | 1 |
| `--batch-size` | Batch size | 64 |
| `--buffer-capacity` | Replay buffer capacity | 1000 |
| `--learning-rate` | Learning rate | 0.001 |
| `--update-interval` | Steps between updates | 1 |
| `--hidden-dim` | Hidden layer dimension | 128 |
| `--belief-dim` | Belief state dimension | 128 |
| `--latent-dim` | Latent space dimension | 128 |
| `--discount-factor` | Discount factor (0 = average reward) | 0.9 |
| `--entropy-weight` | Entropy bonus weight | 0.5 |
| `--kl-weight` | KL weight for inference | 10 |
| `--use-gnn` | Use Graph Neural Network with temporal attention | False |
| `--gnn-layers` | Number of GNN layers | 2 |
| `--attn-heads` | Number of attention heads in GNN | 4 |
| `--temporal-window` | Temporal window size for GNN memory | 5 |
| `--seed` | Random seed | 42 |
| `--output-dir` | Output directory | results |
| `--exp-name` | Experiment name | brandl_validation |
| `--save-model` | Save agent models | False |
| `--load-model` | Load model | None |
| `--eval-only` | Run evaluation only | False |
| `--train-then-evaluate` | Train for 4 episodes with horizon 1000, then evaluate | False |
| `--plot-internal-states` | Enable internal state visualizations | False |
| `--plot-type` | Visualization type (belief, latent, both) | both |
| `--latex-style` | Use LaTeX-style formatting for plots | False |
| `--use-tex` | Use LaTeX rendering for text in plots | False |

## Advanced Analysis

POLARIS enables sophisticated analysis of multi-agent learning dynamics:

- **Learning Rate Estimation**: Automatically calculates exponential learning rates for each agent
- **Theoretical Bounds Comparison**: Compares agent performance against theoretical bounds from [Brandl (2024)]
- **Belief Dynamics**: Visualize how agent beliefs evolve and converge with experience
- **Network Effects**: Study how different network topologies affect information flow and learning

## Project Structure

```
polaris/
├── experiment.py          # Main experiment script
├── modules/
│   ├── __init__.py        # Package initialization
│   ├── agent.py           # POLARIS agent implementation
│   ├── args.py            # Command line argument parsing
│   ├── environment.py     # Social learning environment
│   ├── metrics.py         # Evaluation metrics collection
│   ├── networks.py        # Neural network architectures
│   ├── plotting.py        # Visualization functions
│   ├── replay_buffer.py   # Sequential experience replay
│   ├── simulation.py      # Main simulation logic
│   └── utils.py           # Utility functions
├── results/               # Experiment results directory
├── requirements.txt       # Package dependencies
└── README.md              # This file
```

## Requirements

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Matplotlib
- NetworkX
- scikit-learn
- tqdm
- torch-geometric (for GNN functionality)

## Citation

If you use POLARIS in your research, please cite:

```bibtex
@misc{dogaroglu2025polaris,
  author = {Dogaroglu, Ege Can},
  title = {POLARIS: Partially Observable Learning with Active Reinforcement In Social environments},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ecdogaroglu/polaris}}
}
```

And also cite the original FURTHER framework:

```bibtex
@inproceedings{kim2022further,
  title={Influencing Long-Term Behavior in Multiagent Reinforcement Learning},
  author={Kim, Dong-Ki and Riemer, Matthew D and Liu, Miao and Foerster, Jakob Nicolaus and Everett, Michael and Sun, Chuangchuang and Tesauro, Gerald and How, Jonathan P},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Oh, Alice H. and Agarwal, Alekh and Belgrave, Danielle and Cho, Kyunghyun},
  year={2022},
  url={https://openreview.net/forum?id=_S9amb2-M-I}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

The original FURTHER framework is also licensed under the MIT License, Copyright (c) 2022 Dong-Ki Kim.
