"""
Responsible for training the MuZero network, given batches of simulated data from the RL system.
"""

from typing import List

import torch
from configs import MuZeroConfig
from models import SampleData
from nn_manager.networks import MuZeroNetwork
from nn_manager.storage import SharedStorage
from rl_system.storage import ReplayBuffer
import torch.nn.functional as F

class NetworkTrainer():
    def __init__(self):
        self.step = 0 # Counting the number of steps in the training

    def train_network(self, config: MuZeroConfig, storage: SharedStorage,
                    replay_buffer: ReplayBuffer):
        network = MuZeroNetwork()
        learning_rate = config.lr_init * config.lr_decay_rate**(
            self.step / config.lr_decay_steps)
        optimizer = tf.train.MomentumOptimizer(learning_rate, config.momentum)

        for i in range(config.training_steps):
            if i % config.checkpoint_interval == 0:
                storage.save_network(i, network)
            batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
            self.update_weights(optimizer, network, batch, config.weight_decay)
        
        storage.save_network(config.training_steps, network)
        self.step += 1

    def update_weights(self, optimizer: tf.train.Optimizer, network: MuZeroNetwork, batch : List[SampleData],
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

            # hidden_state = tf.scale_gradient(hidden_state, 0.5)

            for prediction, target in zip(predictions, targets):
                gradient_scale, value, reward, policy_logits = prediction
                target_value, target_reward, target_policy = target

                l = (
                    scalar_loss(value, target_value) +
                    scalar_loss(reward, target_reward) +
                    softmax_cross_entropy_with_logits(
                        logits=policy_logits, labels=target_policy))

                loss += l # tf.scale_gradient(l, gradient_scale)

        for weights in network.get_weights():
            loss += weight_decay * F.l1_loss(weights) # L2 in pseudocode

        optimizer.minimize(loss)

def softmax_cross_entropy_with_logits(logits : torch.tensor, targets : torch.tensor):
    """PyTorch version of tf.nn.softmax_cross_entropy_with_logits"""
    # Apply log softmax to the logits
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Compute cross-entropy loss manually (with one-hot targets)
    loss = -(targets * log_probs).sum(dim=-1)
    
    return loss

def scalar_loss(prediction, target) -> float:
  # MSE in board games, TODO cross entropy between categorical values in Atari.
  return F.mse_loss(torch.tensor(prediction), torch.tensor(target))

