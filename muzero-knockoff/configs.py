import collections
from typing import Optional
from rl_assets.game import Game
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
                lr: float,
                lr_decay_steps: float,
                pb_c_base : float,
                pb_c_init : float,
                window_size : float,
                exploration_fraction : float,
                num_unroll_steps : float,
                training_steps : int,
                td_discount : float,
                pos_reward_mult : int,
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
      self.pos_reward_mult = pos_reward_mult

      # Exponential learning rate schedule
      self.lr_init = lr_init
      self.lr = lr
      self.lr_decay_rate = 0.1
      self.lr_decay_steps = lr_decay_steps

    def new_game(self):
      return Game(self.action_space_size, self.discount)

def make_fruit_picker_config() -> MuZeroConfig:
  def visit_softmax_temperature(num_moves, training_step):
    """
    Rule for choosing which action to take after MCTS simulation
    """
    if training_step < 8000:
        # Early in training - stay more exploratory
        return max(0.5, 1.0 - (num_moves / 50))
    elif training_step < 16000:
        # Early in training - stay more exploratory
        return max(0.3, 1.0 - (num_moves / 40))
    else:
        # Later in training - become more exploitative
        return max(0.1, 1.0 - (num_moves / 30))
      
  # 10x10 grid plus 4 possible actions (up, down, left, right)
  action_space_size = 4
  
  return MuZeroConfig(
      action_space_size=action_space_size,
      max_moves=32,  # Maximum moves before game ends

      ## MCTS
      discount=0.5,  # MCTS backprop value discount weights reward vs value
      dirichlet_alpha=0.5, # How uniform / concentrated action distribution is. 
      #^ (0.1 = basically one action, 0.4 = moderate concentration, 0.8 = balanced, 1.6 = uniform)
      exploration_fraction=0.25, # How much percentage of the action picking is from exploration noise
      num_simulations=64, # Number of steps in the MCTS simulation
      pb_c_init=1.75, # MCTS UCB constant
      #^ Higher values increase the exploration bonus
      #^ Lower values favor exploitation of known good moves
      pb_c_base=200, # MCTS UCB constant, 200 should be good
      
      ## Training
      batch_size=512, # Batch size of training data
      pos_reward_mult=5, # How many copies of samples including a postive rewards to add to batch
      #^ Purpose, 1.0 rewards are quite sparse in FruitCatcher, to normalize batch data, we need this
      td_steps=1, # Steps ahead of action to assign value in the training targets
      td_discount=0.95, # Discount for value td_steps ahead td_discount^td_steps
      num_unroll_steps=3, # Number of actions and vaule targets to have in each training sample
      num_actors=8, # Not used
      lr_init=0.001, # Use for the first window_size // 2 simulations, while batch is being built
      lr=0.1, 
      lr_decay_steps=20e3, # Not used
      window_size=256, # Window size of numbers games stored in the replay buffer (256 is max for 16GB RAM)
      training_steps=35000,
      visit_softmax_temperature_fn=visit_softmax_temperature)