### Main Muzero entry function ###
# Running the self-play and training 

from configs import MuZeroConfig
from rl_system.game import Game
from mcts import MCTS
from models import Node
from nn_manager.networks import Network
from rl_system.storage import ReplayBuffer, SharedStorage
import tensorflow as tf

# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def muzero(config: MuZeroConfig):
  storage = SharedStorage()
  replay_buffer = ReplayBuffer(config)

  for _ in range(config.num_actors):
    launch_job(run_selfplay, config, storage, replay_buffer)

  train_network(config, storage, replay_buffer)

  return storage.latest_network()

###########################################
### START SELF PLAY FOR DATA GENERATION ###

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: MuZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
  while True:
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)

# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, network: Network) -> Game:
    game = config.new_game()
    mcts = MCTS(config=config)

    while not game.terminal() and len(game.history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        mcts.expand_node(root, game.to_play(), game.legal_actions(),
                    network.initial_inference(current_observation)) 
        mcts.add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        mcts.run_mcts(config, root, game.action_history(), network)
        action = mcts.select_action(num_moves=len(game.history), 
                                    node=root, 
                                    network=network)
        game.apply(action)
        game.store_search_statistics(root)
    return game

### END SELF PLAY FOR DATA GENERATION ###
#########################################

##############################
### START TRAINING PROCESS ###

def train_network(config: MuZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
  network = Network()
  learning_rate = config.lr_init * config.lr_decay_rate**(
      tf.train.get_global_step() / config.lr_decay_steps)
  optimizer = tf.train.MomentumOptimizer(learning_rate, config.momentum)

  for i in range(config.training_steps):
    if i % config.checkpoint_interval == 0:
      storage.save_network(i, network)
    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
    update_weights(optimizer, network, batch, config.weight_decay)
  storage.save_network(config.training_steps, network)


def update_weights(optimizer: tf.train.Optimizer, network: Network, batch,
                   weight_decay: float):
  loss = 0
  for image, actions, targets in batch:
    # Initial step, from the real observation.
    value, reward, policy_logits, hidden_state = network.initial_inference(
        image)
    predictions = [(1.0, value, reward, policy_logits)]

    # Recurrent steps, from action and previous hidden state.
    for action in actions:
      value, reward, policy_logits, hidden_state = network.recurrent_inference(
          hidden_state, action)
      predictions.append((1.0 / len(actions), value, reward, policy_logits))

      hidden_state = tf.scale_gradient(hidden_state, 0.5)

    for prediction, target in zip(predictions, targets):
      gradient_scale, value, reward, policy_logits = prediction
      target_value, target_reward, target_policy = target

      l = (
          scalar_loss(value, target_value) +
          scalar_loss(reward, target_reward) +
          tf.nn.softmax_cross_entropy_with_logits(
              logits=policy_logits, labels=target_policy))

      loss += tf.scale_gradient(l, gradient_scale)

  for weights in network.get_weights():
    loss += weight_decay * tf.nn.l2_loss(weights)

  optimizer.minimize(loss)


def scalar_loss(prediction, target) -> float:
  # MSE in board games, cross entropy between categorical values in Atari.
  return -1

### END TRAINING PROCESS ###
############################


# Stubs to make the typechecker happy.
def launch_job(f, *args):
  f(*args)