### Main storage object ###
# Used to store both the replay buffer, and the trained neural networks 
from typing import List
from configs import MuZeroConfig
from models import SampleData
from rl_system.game import Game
from nn_manager.networks import Network
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
    return [SampleData(g.make_image(i), g.actions[i:i + num_unroll_steps],
             g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
            for (g, i) in game_pos]

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

