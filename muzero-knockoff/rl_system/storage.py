### Main storage object ###
# Used to store both the replay buffer, and the trained neural networks 
from typing import List, NamedTuple
import numpy as np
from configs import MuZeroConfig
from models import Action, SampleData
from rl_system.game import Game
from nn_manager.networks import Network
import torch.nn.functional as F
import torch
import random

class ReplayBuffer(object):

  def __init__(self, config: MuZeroConfig):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self, num_unroll_steps: int, td_steps: int) -> List[SampleData]:
    games = [self.sample_game() for _ in range(self.batch_size)]
    game_pos = [(g, self.sample_position(g)) for g in games]
    
    batch : List[SampleData] = []

    for (g,i) in game_pos:
      image = g.make_image(i)
      actions : List[Action] = g.actions[i:i + num_unroll_steps]
      # TODO find out if the [] wraping of action is problematic
      one_hot_encoded_actions : List[torch.Tensor] = [F.one_hot(torch.tensor([action.index]), len(g.action_space())) 
                                                      for action in actions] # Convert actions to one hot vector
      targets = g.make_target(i, num_unroll_steps, td_steps, g.to_play())
      batch.append(SampleData(image, one_hot_encoded_actions, targets))

    return batch
      

  def sample_game(self) -> Game:
    """
    Sample random game from replay buffer
    -> random Game
    """
    # Sample game from buffer either uniformly or according to some priority.
    num_games = len(self.buffer)
    return self.buffer[random.randint(0, num_games - 1)] # Sample random game

  def sample_position(self, game : Game) -> int:
    """
    Sample random position index from game
    -> random postion index (int)
    """
    # Sample position from game either uniformly or according to some priority.
    num_positions = len(game.observations)
    return random.randint(0, num_positions - 1)


def print_sample_data(sample: SampleData, verbose: bool = False):
  """
  Prints a nicely formatted overview of a SampleData object.
  
  Args:
      sample: A SampleData object containing observation, action history, and targets
      verbose: If True, prints detailed tensor values; if False, just prints shapes and summaries
  """
  print("=" * 50)
  print("SAMPLE DATA OVERVIEW")
  print("=" * 50)
  
  # Print observation information
  print("\n📊 OBSERVATION:")
  print(f"  Shape: {sample.observation.shape}")
  
  if verbose:
      # Handle different observation dimensions
      if len(sample.observation.shape) == 3:  # Batch, channels, height, width
          print("  Grid representation:")
          for t in range(sample.observation.shape[0]):
              print(f"  Timestep {t}:")
              for row in sample.observation[t]:
                  print("    " + " ".join(f"{val.item():5.2f}" for val in row))
      else:
          print("  Values:")
          print(sample.observation.cpu().numpy())
  
  # Print action history
  print("\n🎮 ACTION HISTORY:")
  print(f"  Number of actions: {len(sample.action_history)}")
  
  for i, action in enumerate(sample.action_history):
      action_np = action.cpu().numpy()
      if verbose:
          print(f"  Action {i}: {action_np}")
      
      # Decode one-hot action
      if 1 in action_np:
          action_idx = np.where(action_np == 1)[0][0]
          action_name = ["Up", "Right", "Down", "Left"][action_idx] if action_idx < 4 else f"Action {action_idx}"
          print(f"  Step {i}: {action_name} (index {action_idx})")
      else:
          print(f"  Step {i}: No action (all zeros)")
  
  # Print targets (value, reward, policy_proba)
  print("\n🎯 TARGETS:")
  print(f"  Number of target steps: {len(sample.targets)}")
  
  for i, target in enumerate(sample.targets):
      print(f"  Step {i}:")
      print(f"    Value: {target.value.clone().item():.4f}")
      print(f"    Reward: {target.reward:.4f}")
      
      # Print policy probabilities
      if verbose:
          print("    Policy probabilities:")
          action_names = ["Up", "Right", "Down", "Left"]
          for j, prob in enumerate(target.policy_proba):
              action_name = action_names[j] if j < len(action_names) else f"Action {j}"
              print(f"      {action_name}: {prob:.4f}")
      else:
          # Find most likely action
          if target.policy_proba:
              max_prob_idx = np.argmax(target.policy_proba)
              max_prob = target.policy_proba[max_prob_idx]
              action_name = ["Up", "Right", "Down", "Left"][max_prob_idx] if max_prob_idx < 4 else f"Action {max_prob_idx}"
              print(f"    Most likely action: {action_name} ({max_prob:.4f})")
  
  print("\n" + "=" * 50)