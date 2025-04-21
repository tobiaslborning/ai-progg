
import numpy as np
import torch
from fruit_picker import FruitPickerEnv
from models import Node
from nn_manager.storage import SharedStorage
from nn_manager.tensorboard_logger import TensorBoardLogger
from nn_manager.training import NetworkTrainer
from rl_system.game import Game
from configs import make_snake_config
from mcts import MCTS
from rl_system.storage import ReplayBuffer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from snake import SnakeEnv

game_type = ["snake", "fruit_picker"][1]

logger = TensorBoardLogger(game_type)

config = make_snake_config()

replay_buffer = ReplayBuffer(config=config)
storage = SharedStorage()

network = storage.latest_network(game_type=game_type)

optimizer = torch.optim.SGD(params=network.parameters(),
                            lr=config.lr_init,
                            momentum=config.momentum)

# T_0: Initial restart period
# T_mult: Factor by which T_i increases after each restart
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1028, T_mult=3, eta_min=0.07)

network_trainer = NetworkTrainer()

if game_type == "snake":
    env = SnakeEnv()
if game_type == "fruit_picker":
    env = FruitPickerEnv(grid_size=5)

for simulation in range(20000):
    game = Game(action_space_size=4, discount=0.95, env=env)

    print(f"SIMULATION {simulation + 1}")

    mcts = MCTS(config=config)
    sim_num = 0

    while not game.terminal() and len(game.actions) < config.max_moves: # NOTE SNAKE HAS BUILT IN MAX MOVES
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        # print("current obs:", current_observation)
        mcts.expand_node(root, game.to_play(), game.action_space(),
                    network.initial_inference(torch.Tensor(current_observation))) 
        mcts.add_exploration_noise(root)
        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        mcts.run_mcts(root, game.action_history(), game.action_space(), network)
        action = mcts.select_action(num_moves=len(game.actions), 
                                    node=root, 
                                    network=network)
        game.apply(action)
        game.store_search_statistics(root)

        sim_num += 1

    scheduler.step()
    env.reset()
    logger.log_game(sim_num, simulation, game)

    print(f"""Survived {sim_num} steps \n    reward: {round(np.sum(game.rewards),3)} \n    lr:{round(np.sum(optimizer.param_groups[0]['lr']), 5)}""" )
    # Test data generation
    # for i in range(sim_num):
    #     print(f"Data step {i}")
    #     print(f"Observation {i}", game.make_image(i))
    #     print(f"Actions {[action.index for action in game.action_history().history[i : i + num_unroll_steps]]}")
    #     for (value, reward, policy) in game.make_target(i, num_unroll_steps, 0, Player()):
    #         print(f"Value: {value}, Reward {reward}, Policy {policy}")
    #     print()

    
    replay_buffer.save_game(game)
    
    print("Training on batches")
    batch = replay_buffer.sample_batch(num_unroll_steps=config.num_unroll_steps, td_steps=config.td_steps)
    loss = network_trainer.update_weights(optimizer=optimizer,
                                network=network,
                                batch=batch,
                                weight_decay=0) # NOT USED
    
    logger.log_loss(loss, simulation)
    
    if  simulation % 100 == 0 and simulation > 0:
        logger.log_gradients(network, simulation)
        storage.save_network(sim_num=simulation, 
                             loss=loss['total_loss'], 
                             optimizer=optimizer, 
                             network=network,
                             game_type=game_type)
            
    print()
    print("Simulation loss:", loss["total_loss"], "\n")

print("observations")
print("last action:", action.index)
for obs in game.observations:
    print()
    print(obs)