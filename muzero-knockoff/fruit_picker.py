import numpy as np
import gym
from gym import spaces
import random
from typing import Optional, Tuple, Dict, Any

class FruitPickerEnv(gym.Env):
    """
    A simple grid-based Fruit Picker environment following Gym interface.
    The agent (a single point) moves around collecting fruits.
    No growing tail, just a simple agent moving on the grid.
    The agent cannot move outside the boundaries of the grid.
    
    Actions:
    - 0: Move Up
    - 1: Move Right
    - 2: Move Down
    - 3: Move Left
    
    Observation:
    - Single channel grid_size x grid_size grid where:
      - 0: Empty space
      - 1: Agent (the picker)
      - 2: Fruit
    
    Rewards:
    - Collecting fruit: +1.0
    - Invalid move (trying to leave grid): -0.1
    - Otherwise: 0.0
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, grid_size=4, max_steps=1000):
        super(FruitPickerEnv, self).__init__()
        
        # Grid size
        self.grid_size = grid_size
        
        # Action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        
        # Observation space: single channel with values 0-2
        self.observation_space = spaces.Box(
            low=0, high=2, 
            shape=(self.grid_size, self.grid_size), 
            dtype=np.uint8
        )
        
        # State encoding values
        self.EMPTY = 0
        self.AGENT = 1
        self.FRUIT = 2
        
        # Initialize state variables
        self.agent_pos = None
        self.fruit_pos = None
        self.direction = None
        self.score = 0
        self.steps = 0
        self.max_steps = max_steps
        
        # Direction mappings (0=up, 1=right, 2=down, 3=left)
        self.direction_map = [
            (-1, 0),  # up
            (0, 1),   # right
            (1, 0),   # down
            (0, -1)   # left
        ]
        
        # Initial reset
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        # Seed the RNG if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize agent at center
        self.agent_pos = (random.randint(1, 3), 
                  random.randint(1, 3))
        
        # Initialize direction randomly
        self.direction = random.randint(0, 3)
        
        # Place fruit at random location not occupied by agent
        self._place_fruit()
        
        # Reset score and steps
        self.score = 0
        self.steps = 0
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        self.steps += 1
        
        # Update direction
        self.direction = action
        
        # Calculate new agent position without wrapping
        dy, dx = self.direction_map[self.direction]
        new_y = self.agent_pos[0] + dy
        new_x = self.agent_pos[1] + dx
        
        # Check if the move is valid (within grid boundaries)
        reward = 0.0
        if 0 <= new_y < self.grid_size and 0 <= new_x < self.grid_size:
            # Valid move, update agent position
            self.agent_pos = (new_y, new_x)
            
            # Check if fruit was collected
            if self.agent_pos == self.fruit_pos:
                self.score += 1
                reward = 1.0
                self._place_fruit()
            # Check if max steps reached
            done = self.steps >= self.max_steps
        else:
            # Invalid move (trying to leave the grid)
            reward = -1.0  # Small penalty for invalid moves
            done = True 

            
        return self._get_observation(), reward, done, {'score': self.score}
    
    def _place_fruit(self) -> None:
        """Place fruit at random location not occupied by agent."""
        available_positions = [
            (i, j) for i in range(self.grid_size) for j in range(self.grid_size)
            if (i, j) != self.agent_pos
        ]
        
        if available_positions:
            self.fruit_pos = random.choice(available_positions)
        else:
            # This shouldn't happen but just in case
            self.fruit_pos = None
    
    def _get_observation(self) -> np.ndarray:
        """Convert current state to the observation format."""
        # Initialize grid with zeros (empty spaces)
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        
        # Place agent (value 1)
        grid[self.agent_pos[0], self.agent_pos[1]] = self.AGENT
        
        # Place fruit (value 2)
        if self.fruit_pos is not None:
            grid[self.fruit_pos[0], self.fruit_pos[1]] = self.FRUIT
        
        return grid
    
    def _get_action_space(self):
        num_actions = self.action_space.n
        # Create a list of all possible actions
        return list(range(num_actions))
        

    def get_action_mask(self) -> np.ndarray:
        """
        Returns a mask of valid actions.
        1 = valid action, 0 = invalid action (would move outside grid)
        """
        mask = np.ones(4, dtype=np.int8)
        
        # Check each direction for validity
        for action in range(4):
            dy, dx = self.direction_map[action]
            new_y = self.agent_pos[0] + dy
            new_x = self.agent_pos[1] + dx
            
            # If this action would lead to leaving the grid, mark it as invalid
            if new_y < 0 or new_y >= self.grid_size or new_x < 0 or new_x >= self.grid_size:
                mask[action] = 0
        
        return mask
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'rgb_array':
            # RGB array representation
            rgb_grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
            
            # Get the current state
            grid = self._get_observation()
            
            # Map state values to colors
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if grid[i, j] == self.AGENT:
                        rgb_grid[i, j] = [0, 0, 255]  # Blue for agent
                    elif grid[i, j] == self.FRUIT:
                        rgb_grid[i, j] = [255, 0, 0]  # Red for fruit
            
            return rgb_grid
        
        elif mode == 'human':
            # ASCII representation
            grid = self._get_observation()
            symbols = {
                self.EMPTY: '.',
                self.AGENT: 'A',
                self.FRUIT: 'F'
            }
            
            # Create top border
            print("+" + "-" * self.grid_size + "+")
            
            # Print grid with side borders
            for row in grid:
                line = "|"
                for cell in row:
                    line += symbols[cell]
                line += "|"
                print(line)
            
            # Create bottom border
            print("+" + "-" * self.grid_size + "+")
            print(f"Score: {self.score}")
        
        else:
            super(FruitPickerEnv, self).render(mode=mode)
    
    def close(self):
        """Clean up resources."""
        pass


# Example usage
if __name__ == "__main__":
    env = FruitPickerEnv(grid_size=5, max_steps=20)
    obs = env.reset()
    done = False
    total_reward = 0
    
    print("Welcome to Fruit Picker!")
    print("Use w/d/s/a keys to move the agent (A) to collect fruits (F)")
    print("You cannot move outside the grid boundaries.")
    
    while not done:
        print("\nCurrent state:")
        print(env._get_observation())
        
        # Get valid actions
        action_mask = env.get_action_mask()
        valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
        valid_keys = [["w", "d", "s", "a"][i] for i in valid_actions]
        
        print(f"Valid moves: {', '.join(valid_keys)}")
        
        # Get user input for next move
        while True:
            inp = input("\nMove (w=up, d=right, s=down, a=left): ")
            if inp in ["w", "d", "s", "a"]:
                action = ["w", "d", "s", "a"].index(inp)
                break
            else:
                print("Invalid input. Use w/d/s/a keys.")
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Reward: {reward}")
        
        if reward == 1.0:
            print(f"Yum! You collected a fruit! Score: {info['score']}")
        elif reward < 0:
            print("Oops! You can't move outside the grid.")
    
    print(f"\nGame over! Total fruits collected: {info['score']}")