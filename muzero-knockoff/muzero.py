
import glob
import os
from pathlib import Path
import numpy as np
import torch
from fruit_picker import FruitPickerEnv
from models import Node
from nn_manager.storage import SharedStorage
from nn_manager.tensorboard_logger import TensorBoardLogger
from nn_manager.training import NetworkTrainer
from rl_system.game import Game
from configs import make_fruit_picker_config
from mcts import MCTS
from rl_system.storage import ReplayBuffer, print_sample_data
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from snake import SnakeEnv

DEBUG = False

game_type = ["snake", "fruit_picker"][1]

logger = TensorBoardLogger(game_type)

config = make_fruit_picker_config()

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

replay_path = os.path.join("rl_system", "replay")

# VISUALIZATION INITIALIZATION
Path(replay_path).mkdir(parents=True, exist_ok=True)
# Delete all existing gif files in replay directory
gif_files = glob.glob(os.path.join(replay_path, "*.gif"))
for gif_file in gif_files:
    try:
        os.remove(gif_file)
    except Exception as e:
        print(f"Error deleting {gif_file}: {e}")


if game_type == "snake":
    env = SnakeEnv()
if game_type == "fruit_picker":
    env = FruitPickerEnv(grid_size=5, num_fruits=2)

for simulation in range(config.training_steps):
    game = Game(action_space_size=4, discount=config.td_discount, env=env)

    print(f"SIMULATION {simulation + 1}")

    mcts = MCTS(config=config)
    sim_num = 0

    while not game.terminal() and len(game.actions) < config.max_moves: # NOTE SNAKE HAS BUILT IN MAX MOVES
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        # print("current obs:", current_observation)
        init_output = network.initial_inference(torch.Tensor(current_observation))
        if DEBUG: print(f"Initial prediciton value : {init_output.value.clone().item()}")
        mcts.expand_node(root, game.to_play(), game.action_space(),
                         init_output) 
        mcts.add_exploration_noise(root)
        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        mcts.run_mcts(root, game.action_history(), game.action_space(), network)
        action = mcts.select_action(num_moves=len(game.actions), 
                                    node=root, 
                                    network=network)
        game.apply(action)
        game.store_search_statistics(root)
        if DEBUG: print("Root value prediction        : ", root.value().clone().item())
        if DEBUG: print("Chose action: ", action.index, "\n")
        if DEBUG: print("Chosen child reward prediction : ", root.children[action].reward.clone().item())
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
        game.visualize_game(simulation)

        random_sample = batch[-1]
        print_sample_data(random_sample)
            
    print()
    print("Simulation loss:", loss["total_loss"], "\n")
