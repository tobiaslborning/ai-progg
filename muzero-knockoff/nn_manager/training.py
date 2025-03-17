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
import torch.nn.utils as U

class NetworkTrainer():
    def __init__(self):
        self.step = 0 # Counting the number of steps in the training
        self.losses = []

    def train_network(self, config: MuZeroConfig, storage: SharedStorage,
                    replay_buffer: ReplayBuffer):
        network = MuZeroNetwork()
        learning_rate = config.lr_init * config.lr_decay_rate**(
            self.step / config.lr_decay_steps)
        optimizer = torch.optim.SGD(params=network.parameters(),
                                    lr=learning_rate,
                                    momentum=config.momentum)
        
        for i in range(config.training_steps):
            if i % config.checkpoint_interval == 0:
                storage.save_network(i, network)
            batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
            self.update_weights(optimizer, network, batch, config.weight_decay)
        
        storage.save_network(config.training_steps, network)
        self.step += 1

    def update_weights(self, optimizer: torch.optim, network: MuZeroNetwork, batch : List[SampleData],
                    weight_decay: float) -> float:
        optimizer.zero_grad()
        loss = 0
        for image, actions, targets in batch:
            # Initial step, from the real observation.
            value, reward, policy_logits, hidden_state = network.initial_inference(
                image)

            predictions = [(1.0, value, reward, policy_logits)]
            # Recurrent steps, from action and previous hidden state.
            
            # TODO some batches lack actions
            if not len(actions) > 0:
                continue

            for action in actions:
                value, reward, policy_logits, hidden_state = network.recurrent_inference(
                    hidden_state, action)
                
                predictions.append((1.0 / len(actions), value, reward, policy_logits))
                
            # hidden_state = tf.scale_gradient(hidden_state, 0.5)

            for prediction, target in zip(predictions, targets):
                gradient_scale, value, reward, policy_logits = prediction
                target_value, target_reward, target_policy = target

                # TODO PROBLEM: some target policies are empty find more elegant solution
                if not len(target_policy) > 0:
                    continue

                # Stack individual policys from action dict into one tensor
                policy_logits = torch.stack([policy for (_, policy) in policy_logits.items()])
                target_policy = torch.tensor(target_policy)

                l = (
                    scalar_loss(value, torch.tensor([[target_value]])) +
                    scalar_loss(reward, torch.tensor([[target_reward]])) +
                    softmax_cross_entropy_with_logits(
                        logits=policy_logits, targets=target_policy))
                loss += l # tf.scale_gradient(l, gradient_scale)
        loss.backward()
        
        U.clip_grad_value_(network.parameters(), 0.5) # Gradient clipping, ensuring gradients dont explode
        optimizer.step()
        # for weights in network.get_weights():
        #     loss += weight_decay * F.l1_loss(weights) # L2 in pseudocode

        # optimizer.minimize(loss)
        return loss.item()

def softmax_cross_entropy_with_logits(logits : torch.tensor, targets : torch.tensor):
    """PyTorch version of tf.nn.softmax_cross_entropy_with_logits"""
    # Apply log softmax to the logits
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Compute cross-entropy loss manually (with one-hot targets)
    loss = -(targets * log_probs).sum(dim=-1)
    
    return loss

def scalar_loss(prediction, target) -> float:
    # MSE in board games, TODO cross entropy between categorical values in Atari
    mse_loss = F.mse_loss(prediction, target)
    # print("\nScalar loss")
    # print("prediction", prediction)
    # print("target", target)
    # print("MSE loss")
    # print(mse_loss)
    return mse_loss

