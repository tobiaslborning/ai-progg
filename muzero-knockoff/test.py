
import torch
from models import Node, Player
from nn_manager.networks import MuZeroNetwork
from rl_system.game import Game
from configs import make_snake_config
from mcts import MCTS
from rl_system.storage import ReplayBuffer


network = MuZeroNetwork(
    observation_dimensions=(10, 10), 
    num_observations=1, 
    hidden_state_dimension=16, 
    hidden_layer_neurons=64,
    num_actions=4,
    value_size=1,
    reward_size=1
)

config = make_snake_config()

game = Game(action_space_size=4, discount=0.1)


game = config.new_game()
mcts = MCTS(config=config)
sim_num = 0

while not game.terminal() and len(game.actions) < config.max_moves: # NOTE SNAKE HAS BUILT IN MAX MOVES
    # At the root of the search tree we use the representation function to
    # obtain a hidden state given the current observation.
    root = Node(0)
    current_observation = game.make_image(-1)
    mcts.expand_node(root, game.to_play(), game.action_space(),
                network.initial_inference(torch.Tensor(current_observation))) 
    mcts.add_exploration_noise(root)
    # We then run a Monte Carlo Tree Search using only action sequences and the
    # model learned by the network.
    mcts.run_mcts(root, game.action_history(), game.action_space(), network)
    action = mcts.select_action(num_moves=len(game.actions), 
                                node=root, 
                                network=network)
    print("Applying action:", action.index)
    game.apply(action)
    print(game.make_image(-1))
    game.store_search_statistics(root)
    
    sim_num += 1
    print("simulation number", sim_num)

# Test data generation
num_unroll_steps = 5
for i in range(sim_num):
    print(f"Data step {i}")
    print(f"Observation {i}", game.make_image(i))
    print(f"Actions {[action.index for action in game.action_history().history[i : i + num_unroll_steps]]}")
    for (value, reward, policy) in game.make_target(i, num_unroll_steps, 0, Player()):
        print(f"Value: {value}, Reward {reward}, Policy {policy}")
    print()

replay_buffer = ReplayBuffer(config=config)
replay_buffer.save_game(game)
print("Sample Batch:")
print(replay_buffer.sample_batch(5, 0))

# value, reward, policy_logits, hidden_state = network.initial_inference(data[0])
# print("Hidden state:", hidden_state)
# print("Policy:", policy_logits, "Value:", value, "Reward:", reward)
# value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, torch.Tensor([[0,0,0,1]]))
# print("Hidden state:", hidden_state)
# print("Policy:", policy_logits, "Value:", value, "Reward:", reward)
