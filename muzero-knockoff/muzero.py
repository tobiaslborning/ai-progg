"""
Entry file for running the MuZero RL training loop
"""
import glob
import os
from pathlib import Path
import numpy as np
import torch
from environments.fruit_picker import FruitPickerEnv
from environments.snake import SnakeEnv
from models import Node
from nn_manager.storage import SharedStorage
from nn_manager.tensorboard_logger import TensorBoardLogger
from nn_manager.training import NetworkTrainer
from rl_assets.game import Game
from configs import make_fruit_picker_config
from mcts import MCTS
from rl_assets.storage import ReplayBuffer, print_batch_stats, print_sample_data
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

DEBUG = False

game_type = ["snake", "fruit_picker"][1]

logger = TensorBoardLogger(game_type)

config = make_fruit_picker_config()

replay_buffer = ReplayBuffer(config=config)
storage = SharedStorage()

network = storage.latest_network(game_type=game_type)

optimizer = torch.optim.Adam(params=network.parameters(),
                            lr=config.lr_init)

# T_0: Initial restart period
# T_mult: Factor by which T_i increases after each restart
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1028, T_mult=3, eta_min=0.0007)

network_trainer = NetworkTrainer()

replay_path = os.path.join("rl_assets", "replay")

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

    while not game.terminal() and len(game.actions) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        init_output = network.initial_inference(torch.Tensor(current_observation))
        mcts.expand_node(root, game.to_play(), game.action_space(),
                         init_output) 
        mcts.add_exploration_noise(root)
        # Run a Monte Carlo Tree Search using action sequences and the model learned by the network.
        mcts.run_mcts(root, game.action_history(), game.action_space(), network)
        action = mcts.select_action(num_moves=len(game.actions), 
                                    node=root, 
                                    network=network)
        game.apply(action)
        game.store_search_statistics(root)
        sim_num += 1

        if DEBUG: 
            print(f"Initial prediciton value : {init_output.value.clone().item()}")
            print("Root value prediction        : ", root.value().clone().item())
            print("Chose action: ", action.index, "\n")
            print("Chosen child reward prediction : ", root.children[action].reward.clone().item())

    # Prepare for new game simulation        
    mcts.step() # increment step count
    scheduler.step() # increment step count
    env.reset()
    logger.log_game(sim_num, simulation, game) # Add game to logger
    replay_buffer.save_game(game) # Save the game to the replay buffer

    print(f"""
Survived {sim_num} steps 
    reward: {round(np.sum(game.rewards),3)}
    lr:{round(np.sum(optimizer.param_groups[0]['lr']), 5)}
""")
    
    # Train on sampled batches
    batch = replay_buffer.sample_batch(num_unroll_steps=config.num_unroll_steps, td_steps=config.td_steps)
    print("Batch reward stats:")
    print_batch_stats(batch)
    
    loss = network_trainer.update_weights(optimizer=optimizer,
                                network=network,
                                batch=batch,
                                weight_decay=0) # NOT USED
    
    # Logg the loss (note this is not logarithmic loss)
    logger.log_loss(loss, simulation)
    if simulation % 10 == 0: 
        game.visualize_game(simulation) # Create gif showing game 
        replay_buffer.sort_pos_buffer() # Keep good games
    
    if  simulation % 100 == 0 and simulation > 0:
        logger.log_gradients(network, simulation) # Log the gradients of the networks
        storage.save_network(sim_num=simulation, 
                             loss=loss['total_loss'], 
                             optimizer=optimizer, 
                             network=network,
                             game_type=game_type)
        print("Buffer lengths")
        print("pos  :", len(replay_buffer.pos_buffer))
        print("zero :", len(replay_buffer.zero_buffer))
        print("neg  :", len(replay_buffer.neg_buffer))
        print("Top 20 postive games:")
        for game in reversed(replay_buffer.pos_buffer):
            print("reward:", np.sum(game.rewards), "steps:", len(game.rewards))

        random_sample = batch[-1] 
        print_sample_data(random_sample) # Print random sample for insights

    if simulation % 1000:
        # Keep only top 20 every 1000th step
        replay_buffer.pos_buffer = replay_buffer.pos_buffer[-20:]        

    if simulation == (config.window_size // 4):
        optimizer = torch.optim.Adam(params=network.parameters(),
                            lr=config.lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1028, T_mult=3, eta_min=0.007)

    print()
    print("Simulation loss:", loss["total_loss"], "\n")
