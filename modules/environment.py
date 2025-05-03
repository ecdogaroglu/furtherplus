import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union

class SocialLearningEnvironment:
    """
    Multi-agent environment for social learning based on Brandl 2024.
    
    This environment models agents with partial observability learning 
    the true state through private signals and observations of other agents' actions.
    
    This implementation uses a continuous stream of data without episode structure.
    """
    
    def __init__(
        self,
        num_agents: int = 10,
        num_states: int = 2,
        signal_accuracy: float = 0.75,
        network_type: str = "complete",
        network_params: Dict = None,
        horizon: int = 1000,
        seed: Optional[int] = None
    ):
        """
        Initialize the social learning environment.
        
        Args:
            num_agents: Number of agents in the environment
            num_states: Number of possible states of the world
            signal_accuracy: Probability that a signal matches the true state
            network_type: Type of network structure ('complete', 'ring', 'star', 'random')
            network_params: Parameters for network generation if network_type is 'random'
            horizon: Total number of steps to run
            seed: Random seed for reproducibility
        """
        self.num_agents = num_agents
        self.num_states = num_states
        self.num_actions = num_states  # Actions correspond to states
        self.signal_accuracy = signal_accuracy
        self.horizon = horizon
        
        # Set random seed
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
            
        # Initialize the network (who observes whom)
        self.network = self._create_network(network_type, network_params)
        
        # Initialize state and time variables
        self.true_state = None
        self.current_step = 0
        self.actions = np.zeros(self.num_agents, dtype=np.int32)
        self.signals = np.zeros(self.num_agents, dtype=np.int32)
        
        # Track metrics
        self.correct_actions = np.zeros(self.num_agents, dtype=np.int32)
        self.mistake_history = []
        
    def _create_network(self, network_type: str, network_params: Dict = None) -> np.ndarray:
        """
        Create a network structure based on the specified type.
        """
        # Initialize an empty graph
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_agents))
        
        # Create the specified network structure
        if network_type == "complete":
            # Every agent observes every other agent
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if i != j:
                        G.add_edge(i, j)
        
        elif network_type == "ring":
            # Each agent observes only adjacent agents in a ring
            for i in range(self.num_agents):
                G.add_edge(i, (i + 1) % self.num_agents)
                G.add_edge(i, (i - 1) % self.num_agents)
        
        elif network_type == "star":
            # Central agent (0) observes all others, others observe only the central agent
            for i in range(1, self.num_agents):
                G.add_edge(0, i)  # Central agent observes periphery
                G.add_edge(i, 0)  # Periphery observes central agent
        
        elif network_type == "random":
            # Random network with specified density
            density = network_params.get("density", 0.5)
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if i != j and self.rng.random() < density:
                        G.add_edge(i, j)
                        
            # Ensure the graph is strongly connected
            if not nx.is_strongly_connected(G):
                components = list(nx.strongly_connected_components(G))
                # Connect all components
                for i in range(len(components) - 1):
                    u = self.rng.choice(list(components[i]))
                    v = self.rng.choice(list(components[i + 1]))
                    G.add_edge(u, v)
                    G.add_edge(v, u)
        
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        
        # Convert NetworkX graph to adjacency matrix
        adjacency_matrix = nx.to_numpy_array(G, dtype=np.int32)
        return adjacency_matrix
    
    def _generate_signal(self, agent_id: int) -> int:
        """
        Generate a private signal for an agent based on the true state.
        """
        if self.rng.random() < self.signal_accuracy:
            # Signal matches the true state with probability signal_accuracy
            signal = self.true_state
        else:
            # Generate a random incorrect signal
            incorrect_states = [s for s in range(self.num_states) if s != self.true_state]
            signal = self.rng.choice(incorrect_states)
        
        return signal
    
    def _compute_reward(self, agent_id: int, action: int) -> float:
        """
        Compute the reward for an agent's action using derived reward function.
        """
        signal = self.signals[agent_id]
        q = self.signal_accuracy
        
        # For binary case with signal accuracy q
        if action == signal:  # Action matches signal
            reward = q / (2*q - 1)
        else:  # Action doesn't match signal
            reward = -(1 - q) / (2*q - 1)
        
        return reward
    
    def initialize(self) -> Dict[int, Dict]:
        """
        Initialize the environment state.
        
        Returns:
            observations: Dictionary of initial observations for each agent
        """
        # Sample a new true state
        self.true_state = self.rng.randint(0, self.num_states)
        
        # Reset step counter
        self.current_step = 0
        
        # Reset actions
        self.actions = np.zeros(self.num_agents, dtype=np.int32)
        
        # Generate initial signals for all agents
        self.signals = np.array([self._generate_signal(i) for i in range(self.num_agents)])
        
        # Reset metrics
        self.correct_actions = np.zeros(self.num_agents, dtype=np.int32)
        self.mistake_history = []
        self.incorrect_prob_history = []  # Track incorrect probability assignments
        
        # Create initial observations for each agent
        observations = {}
        for agent_id in range(self.num_agents):
            # Initially, agents only observe their private signal
            observations[agent_id] = {
                'signal': self.signals[agent_id],
                'neighbor_actions': None,  # No actions observed yet
            }
        
        return observations
    
    def step(self, actions: Dict[int, int], action_probs: Dict[int, np.ndarray] = None) -> Tuple[Dict[int, Dict], Dict[int, float], bool, Dict]:
        """
        Take a step in the environment given the actions of all agents.
        
        Args:
            actions: Dictionary mapping agent IDs to their chosen actions
            action_probs: Dictionary mapping agent IDs to their action probability distributions
        """
        # Update step counter
        self.current_step += 1
        
        # Update actions
        for agent_id, action in actions.items():
            self.actions[agent_id] = action
            
        # Generate new signals for all agents
        self.signals = np.array([self._generate_signal(i) for i in range(self.num_agents)])
        
        # Compute rewards and track correct actions
        rewards = {}
        for agent_id in range(self.num_agents):
            rewards[agent_id] = self._compute_reward(agent_id, actions[agent_id])
            
            # Track correct actions for metrics
            if actions[agent_id] == self.true_state:
                self.correct_actions[agent_id] += 1
        
        # Calculate mistake rate for this step (binary)
        mistake_rate = 1.0 - np.mean([1.0 if a == self.true_state else 0.0 for a in self.actions])
        self.mistake_history.append(mistake_rate)
        
        # Calculate incorrect probability assignment rate
        # For each agent, get the probability they assigned to incorrect states
        incorrect_probs = []
        for agent_id in range(self.num_agents):
            if agent_id in action_probs:
                # Sum probabilities assigned to all incorrect states
                incorrect_prob = 1.0 - action_probs[agent_id][self.true_state]
                incorrect_probs.append(incorrect_prob)
            else:
                # If we don't have probabilities for this agent, use 0.5 as a default
                print(f"Warning: No action probabilities for agent {agent_id}")
                incorrect_probs.append(0.5)
        
        # Average incorrect probability across all agents
        avg_incorrect_prob = np.mean(incorrect_probs)
        self.incorrect_prob_history.append(incorrect_probs.copy())

        
        # Create observations for each agent
        observations = {}
        for agent_id in range(self.num_agents):
            # Get actions of neighbors that this agent can observe
            neighbor_actions = {}
            for neighbor_id in range(self.num_agents):
                if self.network[agent_id, neighbor_id] == 1:
                    neighbor_actions[neighbor_id] = self.actions[neighbor_id]
            
            observations[agent_id] = {
                'signal': self.signals[agent_id],
                'neighbor_actions': neighbor_actions,
            }
        
        # Check if we've reached the total number of steps
        done = self.current_step >= self.horizon
        
        # Additional information
        info = {
            'true_state': self.true_state,
            'mistake_rate': mistake_rate,
            'incorrect_prob': self.incorrect_prob_history[-1].copy() if isinstance(self.incorrect_prob_history[-1], list) else self.incorrect_prob_history[-1],
            'correct_actions': self.correct_actions.copy()
        }
        
        return observations, rewards, done, info
    
    def get_autarky_rate(self) -> float:
        """
        Calculate the theoretical autarky learning rate.
        
        Returns:
            autarky_rate: Theoretical learning rate for a single agent in autarky
        """
        # For binary state space with signal accuracy p
        p = self.signal_accuracy
        autarky_rate = (2*p - 1) * np.log(p/(1-p))
        return autarky_rate
    
    def get_bound_rate(self) -> float:
        """
        Calculate the theoretical upper bound on learning rate from Theorem 1.
        
        Returns:
            bound_rate: Theoretical upper bound on learning rate
        """
        # For binary state space with signal accuracy p
        p = self.signal_accuracy
        bound_rate = 2 * (2*p - 1) * np.log(p/(1-p))
        return bound_rate
    
    def get_coordination_rate(self) -> float:
        """
        Calculate the theoretical coordination learning rate from Theorem 2.
        
        Returns:
            coordination_rate: Theoretical coordination learning rate
        """
        # For binary state space with signal accuracy p
        p = self.signal_accuracy
        coordination_rate = 4 * (p - 0.5)**2 * np.log(p/(1-p)) / (2*p - 1)
        return coordination_rate
        
    def reset(self) -> Dict[int, Dict]:
        """
        Backward compatibility method that calls initialize().
        """
        return self.initialize()
        
    def seed(self, seed: Optional[int] = None) -> None:
        """
        Set the random seed for the environment.
        """
        self.rng = np.random.RandomState(seed)
        
    def get_neighbors(self, agent_id: int) -> List[int]:
        """
        Get the list of neighbors that an agent can observe based on the network structure.
        
        Args:
            agent_id: The ID of the agent
            
        Returns:
            List of agent IDs that the specified agent can observe
        """
        neighbors = []
        for neighbor_id in range(self.num_agents):
            if self.network[agent_id, neighbor_id] == 1:
                neighbors.append(neighbor_id)
        return neighbors