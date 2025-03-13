# MCTS MODULE

import collections
import math
import numpy as np
from typing import List, Optional

import torch
import torch.nn.functional as F

from configs import MuZeroConfig

from models import ActionHistory, KnownBounds, Node, Action, Player
from nn_manager.networks import Network, NetworkOutput

###################
## START HELPERS ##

MAXIMUM_FLOAT_VALUE = float('inf')

class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self, known_bounds: Optional[KnownBounds]):
    self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value
  
## END HELPERS ##
#################

###################
## START CLASSES ##

class Node(object):

  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}
    self.hidden_state = None
    self.reward = 0

  def expanded(self) -> bool:
    return len(self.children) > 0

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

## END CLASSES ##
#################


## CORE ALGORITHM ##
# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.

class MCTS(object):
  def __init__(self, config : MuZeroConfig):
    self.config = config 
    
  def run_mcts(self, root: Node, action_history: ActionHistory, action_space: List[Action],
              network: Network):
    min_max_stats = MinMaxStats(self.config.known_bounds)

    for _ in range(self.config.num_simulations):
      history = action_history.clone()
      node = root
      search_path = [node]

      while node.expanded():
        action, node = self.select_child(node, min_max_stats)
        history.add_action(action)
        search_path.append(node)

      # Inside the search tree we use the dynamics function to obtain the next
      # hidden state given an action and the previous hidden state.
      parent = search_path[-2]

      # One hot encode last action for recurrent inference TODO move into network?
      last_action_tensor = torch.tensor([history.last_action().index])
      one_hot_encoded_last_action = F.one_hot(last_action_tensor, len(action_space))

      # Run recurrent inference
      network_output = network.recurrent_inference(parent.hidden_state,
                                                   one_hot_encoded_last_action)
      self.expand_node(node, history.to_play(), history.action_space(), network_output)

      self.backpropagate(search_path, network_output.value, history.to_play(),
                    self.config.discount, min_max_stats)


  def select_action(self, num_moves: int, node: Node,
                    network: Network) -> Action:
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    # t = self.config.visit_softmax_temperature_fn(
    #     num_moves=num_moves, training_steps=network.training_steps())
    temp = 1 - num_moves * 0.01 # After 100 goes to argmax action based on probability
    _, action = softmax_sample(visit_counts, temp)
    return action


  # Select the child with the highest UCB score.
  def select_child(self, node: Node,
                  min_max_stats: MinMaxStats):
    _, action, child = max(
        (self.ucb_score(node, child, min_max_stats), action,
        child) for action, child in node.children.items())
    return action, child


  # The score for a node is based on its value, plus an exploration bonus based on
  # the prior.
  def ucb_score(self, parent: Node, child: Node,
                min_max_stats: MinMaxStats) -> float:
    pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) /
                    self.config.pb_c_base) + self.config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value())
    return prior_score + value_score


  # We expand a node using the value, xx and policy prediction obtained from
  # the neural network.
  def expand_node(self, node: Node, to_play: Player, actions : List[Action],
                  network_output: NetworkOutput):
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
      node.children[action] = Node(p / policy_sum)


  # At the end of a simulation, we propagate the evaluation all the way up the
  # tree to the root.
  def backpropagate(self, search_path: List[Node], value: float, to_play: Player,
                    discount: float, min_max_stats: MinMaxStats):
    for node in search_path:
      node.value_sum += value if node.to_play == to_play else -value
      node.visit_count += 1
      min_max_stats.update(node.value())

      value = node.reward + discount * value


  # At the start of each search, we add dirichlet noise to the prior of the root
  # to encourage the search to explore new actions.
  def add_exploration_noise(self, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(actions))
    frac = self.config.root_exploration_fraction
    for a, n in zip(actions, noise):
      node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def softmax_sample(distribution, temperature: float):
    """
    Sample an action from a distribution of visit counts using softmax with temperature.
    
    Args:
        distribution: List of (visit_count, action) tuples
        temperature: Controls exploration vs exploitation
                     (lower = more greedy, higher = more uniform)
    
    Returns:
        Tuple of (probability, selected_action)
    """
    # Extract visit counts from the distribution
    counts, actions = zip(*distribution) if distribution else ([], [])
    
    if not actions:
        return 0, None
    
    # Handle the edge case where temperature is very small
    if temperature < 0.01:
        # Just return the action with the highest visit count
        best_idx = np.argmax(counts)
        probs = np.zeros(len(actions))
        probs[best_idx] = 1.0
        return probs[best_idx], actions[best_idx]
    
    # Apply softmax with temperature
    counts_t = [count ** (1 / temperature) for count in counts]
    total = sum(counts_t)
    probs = [count / total for count in counts_t]
    
    # Sample an action according to the probabilities
    idx = np.random.choice(len(actions), p=probs)
    return probs[idx], actions[idx]