### Game object, used to interact with a simulated enviroment ###

import os
from typing import List
import gym
from matplotlib import animation, pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from environments.fruit_picker import FruitPickerEnv
from models import Action, ActionHistory, Node, Player, SampleTargets
from environments.snake import SnakeEnv

class Game(object):
  """A single episode of interaction with the environment."""
  
  def __init__(self, action_space_size: int, discount: float, env: gym.Env):
    self.environment = env  # Game specific environment.
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
    # dir = ["UP", "RIGHT", "DOWN", "LEFT"][action.index]
    # print(f"Applying: {dir} at obs number{len(self.observations)}:")
    # print(self.observations[-1])
    # print()
    obs , reward, done, _ = self.environment.step(action.index) # Converting action to int before intereacting with env.
    self.observations.append(torch.Tensor(obs))
    if done:
      self.game_terminal = True
    self.rewards.append(reward)
    self.actions.append(action)

  def store_search_statistics(self, root: Node):
    sum_visits = sum(child.visit_count for child in root.children.values())
    action_space = (Action(index) for index in range(self.action_space_size))
    self.child_visits.append([
        root.children[a].visit_count / sum_visits if a in root.children else 0
        for a in action_space
    ])
    self.root_values.append(root.value())

  def make_image(self, state_index: int):
    """
    Returns the two last frames, zeros if no previous frames exists
    """
    # Game specific feature planes.
    current = self.observations[state_index] # Get state index
    if state_index == 0 or len(self.observations) == 1:
      prev = torch.zeros_like(current)
    else: 
      prev = self.observations[state_index - 1]

    return torch.stack([current, prev], dim=2)
  
  def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int,
                to_play: Player) -> SampleTargets:
    """
    Make target data from a simulated step
    
    state_index      : the observation at the chosen position
    num_unroll_steps : number of actions taken after the chosen position (if they exist)
    td_steps         : estimate value based on the position td_steps in the future
    to_play          : NOT USED but included in DeepMinds Pseudocode

    returns -> List[(value, reward, policy), ...]
    """
    targets = []
    for current_index in range(state_index, state_index + num_unroll_steps):
      bootstrap_index = current_index + td_steps
      if bootstrap_index < len(self.root_values):
        value = self.root_values[bootstrap_index] * self.discount**td_steps
      else:
        # If we're looking beyond the episode, death has most likely occured, use last reward as value
        if len(self.rewards) > 0:
          value = self.rewards[-1]
        else:
          value = -1.0

      # Calculate the discounted rewards
      for i, reward in enumerate(self.rewards[current_index:min(bootstrap_index, len(self.rewards))]):
        value += reward * self.discount**i

      if not isinstance(value, torch.Tensor): 
        value = torch.Tensor([value])

      if current_index < len(self.root_values):
        targets.append(SampleTargets(value, self.rewards[current_index],
                      self.child_visits[current_index]))
      else:
        # Use the last reward (which might be negative if the game ended by leaving the board)
        last_reward = self.rewards[-1] if len(self.rewards) > 0 else 0
        # Uniform distribution (equal probability for all actions)
        num_actions = len(self.child_visits[0]) if len(self.child_visits) > 0 else 4
        terminal_policy = [1.0 / num_actions] * num_actions
        
        targets.append(SampleTargets(torch.Tensor([value]), last_reward, terminal_policy))

    return targets
  
  def to_play(self) -> Player:
    return Player()

  def action_history(self) -> ActionHistory:
    return ActionHistory(self.actions, self.action_space_size)
  
  def visualize_game(self, simulation, interval=500) -> None:


    fig, ax = plt.subplots(figsize=(5, 5))

    colors = ["#f0f0e8", "#4d4d4d", "#7df28c"]  # background, green, gray
    cmap = ListedColormap(colors)
    
    im = ax.imshow(self.observations[0], cmap=cmap, vmin=0, vmax=2)
    ax.set_title(f"FruitPicker - Survived {len(self.observations)} steps")

    rows, cols = self.observations[0].shape
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)

    # Remove tick labels and major ticks
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    def update(frame):
        im.set_array(self.observations[frame])
        return [im]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(self.observations),
        interval=interval,  # milliseconds per frame
        blit=False,
        repeat=False
    )

    replay_path = os.path.join("rl_assets", "replay")
    ani.save(os.path.join(replay_path, f"fruit_picker_{simulation}.gif"), writer='pillow')
    plt.close()


