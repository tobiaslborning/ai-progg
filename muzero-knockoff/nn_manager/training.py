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
                    weight_decay: float) -> dict[str, float]:
        optimizer.zero_grad()
        total_loss = 0
        policy_loss = 0
        reward_loss = 0
        value_loss = 0
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
                # TODO PROBLEM: some target policies are empty find more elegant solution
                if not len(target_policy) > 0:
                    continue

                # Stack individual policys from action dict into one tensor
                policy_logits = torch.stack([policy for (_, policy) in policy_logits.items()])
                target_policy = torch.tensor(target_policy)
                l_value = scalar_loss(value, torch.tensor([[target_value]]))
                l_reward = scalar_loss(reward, torch.tensor([[target_reward]]))
                l_policy = softmax_cross_entropy_with_logits(logits=policy_logits, targets=target_policy) * 0.2
                
                total_loss += (l_value + l_reward + l_policy) / len(batch) # Dividing loss by batch size
                value_loss += l_value / len(batch)
                reward_loss += l_reward / len(batch)
                policy_loss += l_policy / len(batch)
                # if target_reward < 0:
                #     print("First frame  :", image[:,:,0])
                #     print("Second frame :", image[:,:,1])
                #     print("Actions:", actions)
                #     print(f"Prediction: v:{value} r:{reward} p:{policy_logits}")
                #     print(f"Target    : v:{target_value} r:{target_reward} p:{target_policy}")
                #     print(f"Value loss: {l_value}")
                #     print(f"Reward loss: {l_reward}")
                #     print(f"Policy loss: {l_policy}")
                
        total_loss.backward()
        
        U.clip_grad_norm_(network.parameters(), 5.0) # Gradient clipping, ensuring gradients dont explode L2 norm
        optimizer.step()
        # for weights in network.get_weights():
        #     loss += weight_decay * F.l1_loss(weights) # L2 in pseudocode
        # optimizer.minimize(loss)
        return {"total_loss" : total_loss.item(),
                "value_loss" : value_loss,
                "policy_loss" : policy_loss,
                "reward_loss" : reward_loss}

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

