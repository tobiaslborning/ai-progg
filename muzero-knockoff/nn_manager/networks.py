### Main networks module ###
# Interface to use and train the three core MuZero neural networks
# - Representation network (h)
# - Dynamics network (g)
# - Predicion network (f)

# Two main inference methods
# - initial_inference   : h -> f 
# - recurrent_inference : g -> f

from typing import NamedTuple, Dict, List, Tuple

from torch import nn
import torch
import torch.nn.functional as F

from models import NetworkOutput
from rl_system.game import Action




class Network(object):
  def initial_inference(self, image) -> NetworkOutput:
    # representation + prediction function
    return NetworkOutput(0, 0, {}, [])

  def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
    # dynamics + prediction function
    return NetworkOutput(0, 0, {}, [])

  def get_weights(self):
    # Returns the weights of this network.
    return []

  def training_steps(self) -> int:
    # How many steps / batches the network has been trained for.
    return 0
  

class MuZeroNetwork(nn.Module):
  """
  outputs 4 tuple:
    value: float
    reward: float
    policy_logits: Dict[Action, float] # Dict[Action, float]
    hidden_state: List[float]

  """
  def __init__(self, 
               observation_dimensions : Tuple[int, int],  # dimensions of the observation ex: (96, 96) for B&W Atari 
               num_observations : int,                    # Number of observations in history ex: 8 for Go, 100 for Chess
               hidden_state_dimension : int,              # Hidden state dimension
               hidden_layer_neurons: int,                 # Number of neurons in the hidden layer
               num_actions: int,                          # Number of possible actions
               reward_size: int = 1,                      # Reward size (dimension of predicted reward)
               value_size: int = 1,                       # Value size (dimension of predicted value)
               ):
    super().__init__()
    self.representation_network = RepresentationNetwork(
        observation_dimensions, 
        num_observations, 
        hidden_state_dimension, 
        hidden_layer_neurons)
    self.prediction_network = PredictionNetwork(
        num_actions,
        hidden_state_dimension,
        value_size,
        hidden_layer_neurons
    )
    self.dynamics_network = DynamicsNetwork(
        num_actions,
        hidden_state_dimension,
        reward_size,
        hidden_layer_neurons
    )

  def initial_inference(self, raw_state : torch.Tensor) -> NetworkOutput:
    # representation + prediction function
    hidden_state : torch.Tensor = self.representation_network(raw_state)
    policy_logits, value = self.prediction_network(hidden_state)

    reward = torch.tensor(0) # Must be a single reward
    policy_logits_dict = {}
    for i, logit in enumerate(policy_logits[0]):
      policy_logits_dict[Action(i)] = logit # torch still tracks this variable

    # print("\nInitial inference")
    # print("Policy logits ", policy_logits)
    # print("Reward", reward)
    # print("Value", value)
    # print("Policy logits dict", policy_logits_dict)

    return NetworkOutput(value, reward, policy_logits_dict, hidden_state)

  def recurrent_inference(self, hidden_state : torch.Tensor, action : torch.Tensor) -> NetworkOutput:
    # dynamics + prediction function
    hidden_state_new, reward = self.dynamics_network(hidden_state, action)
    policy_logits, value = self.prediction_network(hidden_state_new)

    policy_logits_dict = {}
    for i, logit in enumerate(policy_logits[0]):
      policy_logits_dict[Action(i)] = logit

    return NetworkOutput(value, reward, policy_logits_dict, hidden_state_new)


#############################
### START NETWORKS CONFIG ###

class RepresentationNetwork(nn.Module):
  """ 
  raw_state -> hidden_state 
  raw state consists of (batch_size, num_observations, dimensions...)
  
  """
  def __init__(self,
               observation_dimensions : Tuple[int, int],  # dimensions of the observation ex: (96, 96) for B&W Atari 
               num_observations : int,                    # Number of observations in history ex: 8 for Go, 100 for Chess
               hidden_state_dimension : int,              # Hidden state dimension
               hidden_layer_neurons: int,                 # Number of neurons in the hidden layer
               ):   
    super().__init__()
    height, width = observation_dimensions
    input_size = height * width * num_observations

    # Simple FNN with 1 hidden layer of 64 neurons.
    self.network = nn.Sequential(
      nn.Linear(in_features=input_size, 
                out_features=hidden_layer_neurons), 
      nn.ReLU(),
      nn.Linear(in_features=hidden_layer_neurons, 
                out_features=hidden_state_dimension)
    )

  def forward(self, raw_states : torch.Tensor) -> torch.Tensor:
    x = raw_states.view(1, -1)
    hidden_state = self.network(x)
    return hidden_state

class DynamicsNetwork(nn.Module):
  """ 
  hidden_state_t0, action -> hiddens_state_new, reward_logits 
  given a hidden state and an action, this network will try to simulate the next state, along with the reward logits
  """
  def __init__(self,
               num_actions: int,
               hidden_state_dimension: int,
               reward_size: int,
               hidden_layer_neurons: int
               ):
    super().__init__()
    self.num_actions = num_actions

    transnet_input_size = hidden_state_dimension + num_actions # Concatinated tensor with hidden state and action taken

    self.transition_network = nn.Sequential(
      nn.Linear(in_features=transnet_input_size, 
                out_features=hidden_layer_neurons), 
      nn.ReLU(),
      nn.Linear(in_features=hidden_layer_neurons, 
                out_features=hidden_state_dimension)
    )

    self.reward_network = nn.Sequential(
      nn.Linear(in_features=hidden_state_dimension, 
                out_features=hidden_layer_neurons), 
      nn.ReLU(),
      nn.Linear(in_features=hidden_layer_neurons, 
                out_features=reward_size)
    )

  def forward(self, hidden_state : torch.Tensor, action : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    hidden_state, action -> hidden_state_new, reward_logits
    """
    # TODO implement batch support currently just picking the first from the batch of one observation
    x = torch.cat((hidden_state, action), dim=1) 
    hidden_state_new = self.transition_network(x)
    reward = self.reward_network(hidden_state_new)

    return hidden_state_new, reward

class PredictionNetwork(nn.Module):
  """ hidden_state -> policy_logits, value """
  def __init__(self,
               num_actions : int,    # for policy prediction
               hidden_state_dimension: int,
               value_size: int,      
               hidden_layer_neurons: int
               ):
    super().__init__()

    self.policy_network = nn.Sequential(
      nn.Linear(in_features=hidden_state_dimension, 
                out_features=hidden_layer_neurons), 
      nn.ReLU(),
      nn.Linear(in_features=hidden_layer_neurons, 
                out_features=num_actions)
    )

    self.value_network = nn.Sequential(
      nn.Linear(in_features=hidden_state_dimension, 
                out_features=hidden_layer_neurons), 
      nn.ReLU(),
      nn.Linear(in_features=hidden_layer_neurons, 
                out_features=value_size)
    )

  def forward(self, hidden_state : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    hidden_state -> policy_logits, value_logits
    """
    policy_logits = self.policy_network(hidden_state) # Predict the optimal policy
    value = self.value_network(hidden_state)   # Predict the correct value of the hidden state
    
    return policy_logits, value

### END NETWORKS CONFIG ###
###########################