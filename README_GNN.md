# Graph Neural Network with Temporal Attention for Multi-Agent Learning

This repository extends the FURTHER+ framework with a Graph Neural Network (GNN) that incorporates temporal attention mechanisms to better leverage neighbor actions for improved action selection.

## Key Features

### 1. Graph-Based Representation

The implementation represents the multi-agent environment as a graph where:
- Each agent is a node
- Interactions between agents are edges
- Node features combine belief states and actions
- The graph structure captures the communication network topology

### 2. Temporal Attention Mechanism

The model maintains a temporal memory of previous states and uses attention to focus on relevant historical information:

- Stores a sliding window of past node features and graph structures
- Uses multi-head attention to weigh the importance of past observations
- Applies causal masking to ensure agents only use information available at decision time
- Dynamically adjusts attention based on the current context

### 3. Neural Architecture

The model consists of several key components:

- **Graph Attention Networks (GAT)**: Process the spatial relationships between agents
- **Temporal Attention Layers**: Track and utilize historical information
- **Latent Space Encoder**: Maps graph representations to a latent distribution
- **Action Predictor**: Leverages latent representations to predict neighbor actions

## Usage

To use the GNN model, run the experiment with the `--use-gnn` flag:

```bash
python experiment.py --use-gnn --gnn-layers 2 --attn-heads 4 --temporal-window 5
```

### Parameters

- `--use-gnn`: Enable GNN with temporal attention (instead of traditional encoder-decoder)
- `--gnn-layers`: Number of graph neural network layers (default: 2)
- `--attn-heads`: Number of attention heads in each layer (default: 4)
- `--temporal-window`: Number of past time steps to consider (default: 5)

## Implementation Details

### Graph Construction

For each agent, we construct a graph where:
1. The agent itself is the central node with its belief state and action as features
2. Neighbor agents are connected nodes with their actions as features
3. The graph is fully connected to allow information flow between all agents

### Temporal Processing

The model processes temporal information by:
1. Storing a sliding window of past graph states
2. Applying GNN layers to each timestep independently
3. Using temporal attention to integrate information across time
4. Generating predictions based on the temporally-aware representation

### Advantages Over Previous Approach

- **Structured Representation**: Explicitly models the network structure of agent interactions
- **Temporal Context**: Maintains and utilizes historical information more effectively
- **Attention Mechanism**: Focuses on the most relevant neighbors and time periods
- **Scalability**: Better handles increasing numbers of agents

## Requirements

Additional PyTorch Geometric packages are required:
- torch-geometric
- torch-scatter
- torch-sparse
- torch-cluster
- torch-spline-conv 