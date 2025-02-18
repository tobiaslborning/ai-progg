# Plant and controller
plant_type = ["Bathtub", "Cournot", "Tailgating"][2]
controller_type = ["PID","NN"][1]

# NN
nn_hidden_layers = [6] 
nn_activation_func = ["Sigmoid", "ReLU", "Tanh", "Linear"][2]
nn_init_weight_range = [-0.1, 0.1]

# Training
epochs = 20
epoch_size = 100 # Simulation timesteps
lrate_init = 1.0*10**-7
lrate_max = 1.0*10**-7
noise_range = [-0.02, 0.02]
debug = False # Jax debug prints

# Bathtub pivotals
A = 10   # Cross-sectional area of bathtub
C = 0.7    # Cross-section area of bathtub drain
H0 = 1 # Init Height of water in bathtub

# Cournot pivotals
p_max = 4 # Max price
c_m = 0.1 
target_profit = 2.5
c1_init = 0.5
c2_init = 0.5

# Tailgating plant
v_1 = 0  # inital velocity of car 1
v_2 = 0.3 # intital velocity of car 2
d_0 = 0.01 # inital distance between cars 
target_d = 0.1 # Target distance between cars

# PID init values pivotals
Kp : float = 1.0 # Prev error constant
Kd : float = 1.0 # Error derivative constant
Ki : float = 1.0 # Error integral constant
