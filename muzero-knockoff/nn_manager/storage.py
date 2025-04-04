"""Responsible for storing the MuZero networks"""

import os

import torch
from configs import MuZeroConfig
from nn_manager.networks import MuZeroNetwork

class SharedStorage(object):
    """"""
    def __init__(self):
        self._networks = {}

    def save_network(self, game_type : str, sim_num, loss : str, optimizer : torch.optim, network: MuZeroNetwork):
        nn_dir = os.path.join("nn_manager", "stored_networks", f"{game_type}", f"{game_type}_network_sim_{sim_num}.pt")
        torch.save({
            'epoch': sim_num,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, nn_dir)

    def latest_network(self, game_type : str) -> MuZeroNetwork:
        nn_dir = os.path.join("nn_manager", "stored_networks", f"{game_type}")

        network = MuZeroNetwork(
            observation_dimensions=(5, 5), 
            num_observations=2, 
            hidden_state_dimension=64, 
            hidden_layer_neurons=128,
            num_actions=4,
            value_size=1,
            reward_size=1
        )
        try:
            network_files = [f for f in os.listdir(nn_dir) if f.startswith(f"{game_type}_network_sim_") and f.endswith(".pt")]
            if not network_files:
                print("Failed to load network")
                return network
        except:
            print("Failed to load network")
            return network    
        
        
        # Extract simulation numbers and find the highest one
        sim_numbers = [int(f.split("_sim_")[1].split(".pt")[0]) for f in network_files]
        latest_sim = max(sim_numbers)
        latest_file = f"{game_type}_network_sim_{latest_sim}.pt"

        checkpoint = torch.load(os.path.join(nn_dir, latest_file))
        network.load_state_dict(checkpoint['model_state_dict'])

        print(f"Loaded network from simulation {latest_sim}")
        return network

    

# Snake network
def make_uniform_network():
  return MuZeroNetwork(
    observation_dimensions=(4, 4), 
    num_observations=2, 
    hidden_state_dimension=64, 
    hidden_layer_neurons=128,
    num_actions=4,
    value_size=1,
    reward_size=1
    )