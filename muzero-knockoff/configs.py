import collections
from typing import Optional
from rl_system.game import Game
from models import KnownBounds

class MuZeroConfig(object):

    def __init__(self,
                action_space_size: int,
                max_moves: int,
                discount: float,
                dirichlet_alpha: float,
                num_simulations: int,
                batch_size: int,
                td_steps: int,
                num_actors: int,
                lr_init: float,
                lr_decay_steps: float,
                pb_c_base : float,
                pb_c_init : float,
                window_size : float,
                exploration_fraction : float,
                num_unroll_steps : float,
                training_steps : int,
                td_discount : float,
                visit_softmax_temperature_fn,
                known_bounds: Optional[KnownBounds] = None):
      ### Self-Play
      self.action_space_size = action_space_size
      self.num_actors = num_actors

      self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
      self.max_moves = max_moves
      self.num_simulations = num_simulations
      self.discount = discount

      # Root prior exploration noise.
      self.root_dirichlet_alpha = dirichlet_alpha
      self.root_exploration_fraction = exploration_fraction # default 0.25

      # UCB formula
      self.pb_c_base = pb_c_base
      self.pb_c_init = pb_c_init # default 1.25

      # If we already have some information about which values occur in the
      # environment, we can use them to initialize the rescaling.
      # This is not strictly necessary, but establishes identical behaviour to
      # AlphaZero in board games.
      self.known_bounds = known_bounds

      ### Training
      self.training_steps = training_steps
      self.checkpoint_interval = int(1e3)
      self.window_size = window_size
      self.batch_size = batch_size
      self.num_unroll_steps = num_unroll_steps
      self.td_steps = td_steps
      self.td_discount = td_discount
      self.weight_decay = 1e-4
      self.momentum = 0.9

      # Exponential learning rate schedule
      self.lr_init = lr_init
      self.lr_decay_rate = 0.1
      self.lr_decay_steps = lr_decay_steps

    def new_game(self):
      return Game(self.action_space_size, self.discount)

def make_fruit_picker_config() -> MuZeroConfig:
  def visit_softmax_temperature(num_moves):
    return max(0.1, 1.0 - (num_moves / 40))  # Decays to 0.1 by move 36
      
  # 10x10 grid plus 4 possible actions (up, down, left, right)
  action_space_size = 4
  
  return MuZeroConfig(
      action_space_size=action_space_size,
      max_moves=64,  # Maximum moves before game ends
      discount=0.4,  # MCTS backprop value discount weights reward vs value
      dirichlet_alpha=0.4,
      exploration_fraction=0.35,
      num_simulations=64, # MCTS SIMULATIONS
      batch_size=256, # 128
      td_steps=2, # 10
      td_discount=0.95, # 
      num_unroll_steps=3,
      num_actors=8, # Not used
      lr_init=0.13,
      lr_decay_steps=20e3,
      window_size=128,
      pb_c_init=1.50,
      pb_c_base=200,
      training_steps=10000,
      visit_softmax_temperature_fn=visit_softmax_temperature)