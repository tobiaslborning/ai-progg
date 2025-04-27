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
    self.pos_buffer : List[Game] = []
    self.zero_buffer : List[Game] = []
    self.neg_buffer : List[Game] = []
    self.pos_reward_mult = config.pos_reward_mult

  def save_game(self, game : Game):
    if np.sum(game.rewards) > 0:
      if len(self.pos_buffer) > self.window_size / 3:
        self.pos_buffer.pop(0)
      self.pos_buffer.append(game)
    elif np.sum(game.rewards) == 0:
      if len(self.zero_buffer) > self.window_size / 3:
        self.zero_buffer.pop(0)
      self.zero_buffer.append(game)
    else:
      if len(self.neg_buffer) > self.window_size / 3:
        self.neg_buffer.pop(0)
      self.neg_buffer.append(game)

  def sort_pos_buffer(self):
    """Sorts the games in the positive buffer to keep high reward/step games"""
    def average_reward(game):
      """Calculate the average reward for a game"""
      if not game.rewards or len(game.rewards) == 0:
          return 0  # Handle edge case of empty rewards list
      return sum(game.rewards) / len(game.rewards)
    
    sorted_games = sorted(self.pos_buffer, key=average_reward)
    self.pos_buffer = sorted_games


  def sample_batch(self, num_unroll_steps: int, td_steps: int) -> List[SampleData]:
    
    games = [self.sample_game() for _ in range(self.batch_size)]
    game_pos = [(g, self.sample_position(g)) for g in games]
    
    batch : List[SampleData] = []
    idx = 0
    while len(batch) < self.batch_size and idx < self.batch_size:
      g,i = game_pos[idx]
      image = g.make_image(i)
      actions : List[Action] = g.actions[i:i + num_unroll_steps]

      if len(actions) < 1:
        idx += 1
        continue
      while len(actions) < num_unroll_steps:
         actions.append(actions[-1])

      one_hot_encoded_actions : List[torch.Tensor] = [F.one_hot(torch.tensor([action.index]), len(g.action_space())) 
                                                      for action in actions] # Convert actions to one hot vector
      targets = g.make_target(i, num_unroll_steps, td_steps, g.to_play())
      if 1.0 in [t.reward for t in targets]:
        # Add ten of the sample if it includes a positive reward
        for _ in range(self.pos_reward_mult):
          batch.append(SampleData(image, one_hot_encoded_actions, targets))
      else:
        batch.append(SampleData(image, one_hot_encoded_actions, targets))
      
      idx += 1

    return batch


  def sample_game(self) -> Game:
    """
    Sample random game from replay buffer
    -> random Game
    """
    # Sample game from buffer either uniformly or according to some priority.
    buffer_type = random.randint(-1,1)
    num_games_pos = len(self.pos_buffer)
    num_games_zero = len(self.zero_buffer)
    num_games_neg = len(self.neg_buffer)
    if (buffer_type == 1) and num_games_pos > 0:
      return self.pos_buffer[random.randint(0, num_games_pos - 1)] # Sample random game
    if (buffer_type == 0) and num_games_zero > 0:
      return self.zero_buffer[random.randint(0, num_games_zero - 1)] # Sample random game
    if (buffer_type == -1) and num_games_neg > 0:
      return self.neg_buffer[random.randint(0, num_games_neg - 1)] # Sample random game
    # If buffers some buffers are empty, pick the first one with an available sample
    if num_games_pos > 0:
       return self.pos_buffer[0]
    if num_games_zero > 0:
       return self.zero_buffer[0]
    if num_games_neg > 0:
       return self.neg_buffer[0]

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
  print(sample.observation[:,:, 1])
  print(sample.observation[:,:, 0])
  
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
      print(action.clone())
      print(["Up", "Right", "Down", "Left"])

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

def print_batch_stats(batch : List[SampleData]):
  rewards = {"-1.0" : 0,
             "0.0" : 0,
             "1.0" : 0}
  for sample in batch:
     for targets in sample.targets:
        if (targets.reward) < 0:
           rewards["-1.0"] += 1
        elif (targets.reward) > 0:
           rewards["1.0"] += 1
        else:
           rewards["0.0"] += 1

  print(rewards)