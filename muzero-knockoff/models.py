"""
Contains documented common classes used across modules for improved readability
"""
import collections
from typing import Dict, List, NamedTuple

import gym
import torch


### Start GAME ###
class Action(object):
  """
  Action class, just a wrapped index int.  
  Careful: Do not double wrap Action objects, they need to call .index before interacting with simulation.  
  Action is only used outside of simulation.

  Attributes:
    index (int): index of action in list of actions
  """
  def __init__(self, index: int):
    self.index = index

  def __hash__(self):
    return self.index

  def __eq__(self, other):
    return self.index == other.index

  def __gt__(self, other):
    return self.index > other.index

class Player(object):
  """
  NOT IMPLEMENTED Player class for mulitplayer games
  """
  pass

class Environment(gym.Env):
  """
  The environment MuZero is interacting with.  
  Extending gym.Env for the standard OpenAI gym functions
  """
  def __init__(self):
    super.__init__()

class ActionHistory(object):
  """
  Simple history container used inside the search.
  Only used to keep track of the actions executed.
  
  Attributes:
    history (List[Action]): action history  
    action_space_size (int): size of action space

  """

  def __init__(self, history: List[Action], action_space_size: int):
    self.history = list(history)
    self.action_space_size = action_space_size

  def clone(self):
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action: Action):
    self.history.append(action)

  def last_action(self) -> Action:
    return self.history[-1]

  def action_space(self) -> List[Action]:
    return [Action(i) for i in range(self.action_space_size)]

  def to_play(self) -> Player:
    return Player()


### END GAME ###

### START Networks ###

class NetworkOutput(NamedTuple):
  """
  Neural network output for uMCTS.
  
  Attributes:
      value (torch.Tensor): Estimated value of the current state
      reward (torch.Tensor): Immediate reward for the current state
      policy_logits (Dict[Action, torch.Tensor]): Action probabilities (logits)
      hidden_state (torch.Tensor): Internal representation state

  unpack like this:  
  value, reward, policy_logits, hidden_state = NetworkOutput()
  """
  value: torch.Tensor 
  reward: torch.Tensor
  policy_logits: Dict[Action, torch.Tensor]
  hidden_state: torch.Tensor # List[float]

### END Networks ###

### START TRAINING ###

class SampleTargets(NamedTuple):
  """
  Targets from batch sample. 
  Values are gathederd td_steps ahead.

  Attributes:
    value (float): Value from the prediction network
    reward (float):
    policy_proba (List[float]): 
  """
  value : float 
  reward : float
  policy_proba : List[float]

class SampleData(NamedTuple):
  """
  Data for single sample at position p from a batch.
  Action history contains one hot encoded action vectors for actions at positions p + num_unroll_steps 
  Values are gathederd td_steps ahead.
  Reward and policy_proba does not use td_steps

  Attributes:
    observation (torch.Tensor): observation at position p
    action_history (List[torch_Tensor]): one hot encoded action vectors
    targets (List[SampleTargets]): (value, reward, policy_proba)
  """

  observation : torch.Tensor
  action_history : List[torch.Tensor]
  targets : List[SampleTargets]

## END TRAINING


### START uMCTS ###

class Node(object):
  """
  Node class for mcts.
  prior : value of parent node.

  Attributes:
    visit_count (int): Number of times node has been visited
    to_play (int): Player to move at this node
    prior (float): Prior probability from parent
    value_sum (float): Accumulated value from simulations
    children (Dict[Action, Node]): Child nodes by action
    hidden_state (torch.Tensor): Network state representation
    reward (float): Immediate reward at this node
  """

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


KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


### END uMCTS ###