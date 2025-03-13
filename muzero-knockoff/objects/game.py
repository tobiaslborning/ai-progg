### Game object, used to interact with a simulated enviroment ###

from typing import List
import gym
import torch

from snake import SnakeEnv

class Action(object):

  def __init__(self, index: int):
    self.index = index

  def __hash__(self):
    return self.index

  def __eq__(self, other):
    return self.index == other.index

  def __gt__(self, other):
    return self.index > other.index

class Player(object):
  pass


class Environment(gym.Env):
  """The environment MuZero is interacting with."""

  def step(self, action):
    pass

class ActionHistory(object):
  """Simple history container used inside the search.

  Only used to keep track of the actions executed.
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


class Game(object):
  """A single episode of interaction with the environment."""

  def __init__(self, action_space_size: int, discount: float):
    self.environment = SnakeEnv()  # Game specific environment.
    self.game_terminal = False
    self.observations = [torch.Tensor(self.environment._get_observation())]
    self.actions = []
    self.rewards = []
    self.child_visits = []
    self.root_values = []
    self.action_space_size = action_space_size
    self.discount = discount

  def terminal(self) -> bool:
    # Game specific termination rules.
    return self.game_terminal

  def legal_actions(self) -> List[Action]:
    # Game specific calculation of legal actions.
    pass

  def action_space(self) -> List[Action]:
    # Return the entire action space
    return [Action(i) for i in self.environment._get_action_space()] 

  def apply(self, action: Action):
    obs , reward, done, _ = self.environment.step(action.index) # Converting action to int before intereacting with env.
    self.observations.append(obs)
    if done:
      self.game_terminal = True
    self.rewards.append(reward)
    self.actions.append(action)

  def store_search_statistics(self, root: "Node"): # "" to remove circular dependency
    sum_visits = sum(child.visit_count for child in root.children.values())
    action_space = (Action(index) for index in range(self.action_space_size))
    self.child_visits.append([
        root.children[a].visit_count / sum_visits if a in root.children else 0
        for a in action_space
    ])
    self.root_values.append(root.value())

  def make_image(self, state_index: int):
    # Game specific feature planes.
    return torch.Tensor(self.observations[state_index]) # TODO SUPPORT LONGER HISTORY

  def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int,
                  to_play: Player): # PLAYER NOT USED
    """
    Make target data from a simulated step
    
    state_index      : the observation at the chosen position
    num_unroll_steps : number of actions taken after the chosen position (if they exist)
    td_steps         : estimate value based on the position td_steps in the future
    to_play          : NOT USED but included in DeepMinds Pseudocode

    returns -> List[(value, reward, policy), ...]
    """
    # The value target is the discounted root value of the search tree N steps
    # into the future, plus the discounted sum of all rewards until then.
    targets = []
    for current_index in range(state_index, state_index + num_unroll_steps + 1):
      bootstrap_index = current_index + td_steps
      if bootstrap_index < len(self.root_values):
        value = self.root_values[bootstrap_index] * self.discount**td_steps
      else:
        value = 0

      for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
        value += reward * self.discount**i  # pytype: disable=unsupported-operands

      if current_index < len(self.root_values):
        targets.append((value, self.rewards[current_index],
                        self.child_visits[current_index]))
      else:
        # States past the end of games are treated as absorbing states.
        targets.append((0, 0, []))
    return targets

  def to_play(self) -> Player:
    return Player()

  def action_history(self) -> ActionHistory:
    return ActionHistory(self.actions, self.action_space_size)

