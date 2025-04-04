import datetime
import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import datetime

class TensorBoardLogger():
    """
    Start logger by writing tensorboard --logdir=runs into the terminal
    """
    def __init__(self, game_type : str):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("runs", f"{game_type}", f"{game_type}_{current_time}")
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def log_game(self, sim_num, simulation, game):
        self.writer.add_scalar('Game/Steps', sim_num, simulation)
        self.writer.add_scalar('Game/TotalReward', np.sum(game.rewards), simulation)

    def log_loss(self, loss, simulation):
        if isinstance(loss, dict):  # If loss function returns components
            self.writer.add_scalar('Loss/Total', loss['total_loss'], simulation)
            self.writer.add_scalar('Loss/Value', loss['value_loss'], simulation)
            self.writer.add_scalar('Loss/Policy', loss['policy_loss'], simulation)
            self.writer.add_scalar('Loss/Reward', loss['reward_loss'], simulation)

    def log_gradients(self, network, simulation):
        for name, param in network.named_parameters():
            self.writer.add_histogram(f'Gradients/{name}', 
                                param.grad.data, 
                                simulation)
            self.writer.add_histogram(f'Weights/{name}', 
                               param.data, 
                               simulation)