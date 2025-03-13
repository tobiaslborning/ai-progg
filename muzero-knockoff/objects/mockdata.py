import torch
import numpy as np
import matplotlib.pyplot as plt

def create_snake_mock_data(num_samples=4, grid_size=10, sequence_length=4):
    """
    Create mock data for a Snake game.
    
    Args:
        num_samples: Number of batches to create
        grid_size: Size of the grid (grid_size x grid_size)
        sequence_length: Number of frames per sequence
        
    Returns:
        Tensor of shape [num_samples, sequence_length, grid_size, grid_size, 1]
    """
    # Create a batch of empty data
    data = np.zeros((num_samples, sequence_length, grid_size, grid_size), dtype=np.float32)
    
    for sample in range(num_samples):
        # Create a snake with random initial position
        snake_len = np.random.randint(3, 6)
        head_pos = [np.random.randint(snake_len, grid_size-2), np.random.randint(snake_len, grid_size-2)]
        
        # Random direction: 0=right, 1=down, 2=left, 3=up
        direction = np.random.randint(0, 4)
        
        # Create frames showing snake movement
        for frame in range(sequence_length):
            # Place food
            food_pos = [np.random.randint(0, grid_size), np.random.randint(0, grid_size)]
            data[sample, frame, food_pos[0], food_pos[1]] = 3
            
            # Move snake head based on direction
            if frame > 0:
                if direction == 0:  # right
                    head_pos[1] = min(head_pos[1] + 1, grid_size - 1)
                elif direction == 1:  # down
                    head_pos[0] = min(head_pos[0] + 1, grid_size - 1)
                elif direction == 2:  # left
                    head_pos[1] = max(head_pos[1] - 1, 0)
                else:  # up
                    head_pos[0] = max(head_pos[0] - 1, 0)
            
            # Place snake head
            data[sample, frame, head_pos[0], head_pos[1]] = 2
            
            # Place snake body
            for i in range(1, snake_len):
                if direction == 0:  # right
                    body_pos = [head_pos[0], max(0, head_pos[1] - i)]
                elif direction == 1:  # down
                    body_pos = [max(0, head_pos[0] - i), head_pos[1]]
                elif direction == 2:  # left
                    body_pos = [head_pos[0], min(grid_size-1, head_pos[1] + i)]
                else:  # up
                    body_pos = [min(grid_size-1, head_pos[0] + i), head_pos[1]]
                
                data[sample, frame, body_pos[0], body_pos[1]] = 1
    
    # Convert to PyTorch tensor with channel dimension
    return torch.tensor(data).unsqueeze(-1)

def visualize_snake_frames(data, sample_idx=0):
    """
    Visualize the snake game frames for a specific sample.
    
    Args:
        data: Tensor of shape [num_samples, sequence_length, grid_size, grid_size, channels]
        sample_idx: Index of the sample to visualize
    """
    sequence = data[sample_idx]
    frames = sequence.shape[0]
    
    fig, axes = plt.subplots(1, frames, figsize=(frames*3, 3))
    cmap = plt.cm.get_cmap('viridis', 4)
    
    for i in range(frames):
        if frames > 1:
            ax = axes[i]
        else:
            ax = axes
        
        ax.imshow(sequence[i, :, :, 0], cmap=cmap, vmin=0, vmax=3)
        ax.set_title(f"Frame {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    return fig

# Create mock data
mock_data = create_snake_mock_data(num_samples=4, grid_size=10, sequence_length=4)

# Print shape information
print(f"Mock data shape: {mock_data.shape}")
print(f"Single sample shape: {mock_data[0].shape}")

# Example of how to use the data with your RepresentationNetwork
# Assuming you adjust your network to handle the correct input shape
batch_size = mock_data.shape[0]
sequence_length = mock_data.shape[1]
height = mock_data.shape[2]
width = mock_data.shape[3]
channels = mock_data.shape[4]

# Reshape for your network
# flat_data = mock_data.view(batch_size, -1)
# print(f"Flattened data shape (for network input): {flat_data.shape}")

# Code to display an example (uncomment to run with visualization)
# fig = visualize_snake_frames(mock_data, sample_idx=0)
# plt.show()