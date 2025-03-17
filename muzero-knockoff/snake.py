import numpy as np
import gym
from gym import spaces
import random
from typing import Optional, Tuple, Dict, Any

class SnakeEnv(gym.Env):
    """
    A simple 10x10 Snake game environment following Gym interface.
    Designed to be compatible with MuZero reinforcement learning.
    
    Actions:
    - 0: Move Up
    - 1: Move Right
    - 2: Move Down
    - 3: Move Left
    
    Observation:
    - Single channel 10x10 grid where:
      - 0: Empty space
      - 1: Snake body
      - 2: Snake head
      - 3: Food
    
    Rewards:
    - Eating food: +1.0
    - Game over (collision): -1.0
    - Otherwise: 0.0
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self):
        super(SnakeEnv, self).__init__()
        
        # Grid size
        self.grid_size = 4
        
        # Action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        
        # Observation space: single channel with values 0-3
        self.observation_space = spaces.Box(
            low=0, high=3, 
            shape=(self.grid_size, self.grid_size), 
            dtype=np.uint8
        )
        
        # State encoding values
        self.EMPTY = 0
        self.BODY = 1
        self.HEAD = 2
        self.FOOD = 3
        
        # Initialize state variables
        self.snake = None
        self.food = None
        self.direction = None
        self.score = 0
        self.steps = 0
        self.max_steps = 1000  # Terminate episode after 100 steps
        
        # Direction mappings (0=up, 1=right, 2=down, 3=left)
        self.direction_map = [
            (-1, 0),  # up
            (0, 1),   # right
            (1, 0),   # down
            (0, -1)   # left
        ]
        
        # Direction opposites (can't move directly opposite current direction)
        self.opposite_direction = {0: 2, 1: 3, 2: 0, 3: 1}
        
        # Initial reset
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        # Seed the RNG if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize snake at center
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        
        # Initialize direction randomly
        self.direction = random.randint(0, 3)
        
        # Place food at random location not occupied by snake
        self._place_food()
        
        # Reset score and steps
        self.score = 0
        self.steps = 0
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        self.steps += 1
        
        # Prevent opposite direction movement (snake can't reverse)
        if action == self.opposite_direction[self.direction]:
            action = self.direction
        
        # Update direction
        self.direction = action
        
        # Calculate new head position
        head = self.snake[0]
        dy, dx = self.direction_map[self.direction]
        new_head = ((head[0] + dy) % self.grid_size, (head[1] + dx) % self.grid_size)
        
        # Check if snake collided with itself
        if new_head in self.snake:
            return self._get_observation(), -1.0, True, {'score': self.score}
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Check if food was eaten
        reward = 0.0
        if new_head == self.food:
            self.score += 1
            reward = 1.0
            self._place_food()
        else:
            self.snake.pop()  # Remove tail if no food eaten
        
        # Check if max steps reached
        done = self.steps >= self.max_steps
        
        # In snake game, episode could also end if the board is full
        # (snake length equals grid size squared)
        if len(self.snake) == self.grid_size * self.grid_size:
            done = True
            
        return self._get_observation(), reward, done, {'score': self.score}
    
    def _place_food(self) -> None:
        """Place food at random location not occupied by snake."""
        available_positions = [
            (i, j) for i in range(self.grid_size) for j in range(self.grid_size)
            if (i, j) not in self.snake
        ]
        
        if available_positions:
            self.food = random.choice(available_positions)
        else:
            # No space left, game is won
            self.food = None
    
    def _get_observation(self) -> np.ndarray:
        """Convert current state to the observation format."""
        # Initialize grid with zeros (empty spaces)
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        
        # Place snake body (value 1)
        for segment in self.snake[1:]:
            grid[segment[0], segment[1]] = self.BODY
        
        # Place snake head (value 2)
        if self.snake:
            head = self.snake[0]
            grid[head[0], head[1]] = self.HEAD
        
        # Place food (value 3)
        if self.food is not None:
            grid[self.food[0], self.food[1]] = self.FOOD
        
        return grid
    
    def _get_action_space(self):
        num_actions = self.action_space.n
        # Create a list of all possible actions
        return list(range(num_actions))
        
        
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
                    if grid[i, j] == self.BODY:
                        rgb_grid[i, j] = [0, 255, 0]  # Green for body
                    elif grid[i, j] == self.HEAD:
                        rgb_grid[i, j] = [0, 0, 255]  # Blue for head
                    elif grid[i, j] == self.FOOD:
                        rgb_grid[i, j] = [255, 0, 0]  # Red for food
            
            return rgb_grid
        
        elif mode == 'human':
            # ASCII representation
            grid = self._get_observation()
            symbols = {
                self.EMPTY: '.',
                self.BODY: 'o',
                self.HEAD: 'O',
                self.FOOD: 'F'
            }
            
            display = [[symbols[cell] for cell in row] for row in grid]
            print('\n'.join([''.join(row) for row in display]))
            print(f"Score: {self.score}")
        
        else:
            super(SnakeEnv, self).render(mode=mode)
    
    def get_action_mask(self) -> np.ndarray:
        """
        Returns a mask of valid actions (useful for MuZero).
        1 = valid action, 0 = invalid action
        """
        mask = np.ones(4, dtype=np.int8)
        
        # Mask the opposite direction (can't go backwards)
        mask[self.opposite_direction[self.direction]] = 0
        
        return mask
    
    def close(self):
        """Clean up resources."""
        pass

# # Example usage
# if __name__ == "__main__":
#     env = SnakeEnv()
#     obs = env.reset()
#     done = False
#     total_reward = 0
#     action = 1
#     while not done:
#         # Random action for demonstration
#         obs, reward, done, info = env.step(action)
#         total_reward += reward
        
#         # Render the environment
#         env.render('human')
#         print(f"Action: {action}, Reward: {reward}, Done: {done}")
#         print(obs)
#         print(f"Observation shape: {obs.shape}")
        
#         # Simple pause to see the game
#         inp = str(input("Chose action : "))
#         # Direction mappings (0=up, 1=right, 2=down, 3=left)
#         if (inp in ["w","d","s","a"]):
#             action = ["w","d","s","a"].index(inp)
    
#     print(f"Game over! Total reward: {total_reward}, Score: {info['score']}")