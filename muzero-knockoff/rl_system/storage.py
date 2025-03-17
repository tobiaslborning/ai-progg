### Main storage object ###
# Used to store both the replay buffer, and the trained neural networks 
from typing import List
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

