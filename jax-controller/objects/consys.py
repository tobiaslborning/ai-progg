
import time

import numpy as np
import pivotals as conf
from plants import BathtubPlant, CournotPlant, TailgatingPlant
from controllers import Controller, NNController, PIDController

import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt

# Select controller
if (conf.controller_type == "NN"):
    layers = [3]
    layers.extend(conf.nn_hidden_layers)
    layers.append(1)
    controller = NNController(
        layers=layers,
        activation_func=conf.nn_activation_func,
        init_value_range=conf.nn_init_weight_range
    )
if (conf.controller_type == "PID"):
    controller = PIDController()

lr, epochs, epoch_size = conf.lrate_init, conf.epochs, conf.epoch_size

if (conf.plant_type == "Bathtub"):
    plant = BathtubPlant(area=conf.A, 
                         drain_area=conf.C,
                         target=conf.H0)
if (conf.plant_type == "Cournot"):
    plant = CournotPlant(p_max=conf.p_max, 
                         c_m=conf.c_m, 
                         target=conf.target_profit,
                         c1_init=conf.c1_init,
                         c2_init=conf.c2_init
                         )
if (conf.plant_type == "Tailgating"):
    plant = TailgatingPlant(v_lim=0.4,
                            v_1=0,
                            v_2=0.3,
                            d_0=0.3,
                            target_d=0.1)

debug = conf.debug

target = plant.get_target_value() # get the plant target value
params = controller.initializeParams() # initialize the params that shall be fitted
# initialize timestep array, since the timestep value shall be passed into functions tranced by jax, it needs to be a float.
timesteps = jnp.arange(epoch_size) 
def run_one_epoch(params):
    errors = jnp.zeros(epoch_size) # Initialize errors array to track the erros across the epoch
    states = plant.init_state_array(epoch_size) # Initialize the plants state array to track the states across the epoch
    D = np.random.uniform(conf.noise_range[0], conf.noise_range[1], epoch_size) # Generate noise
    U = 0 # Initalize the controller output U to a value of 0.
    for T in timesteps:
        # Run the plant one timestep, returning botht the updated state, and the output of the plant
        updated_state, output =  plant.plant_timestep_change(T, states, U, D) 
        states = states.at[T + 1].set(updated_state) # Add the new state to the state array
        errors = errors.at[T + 1].set(target - output) # Add the error to the errror array
        # Calculate a new controller output based on the errors obtained
        U = controller.calcU(
            params=params,
            t=T, 
            error_history=errors
        )
    if (debug):
        jax.debug.print(" - - - - - - Last prediction of epoch - - - - - - - ")
        jax.debug.print("Ouput (Profit): {}", output)
        jax.debug.print("Error: {}", (target - output))
        jax.debug.print("Updated state {}", updated_state)
        jax.debug.print("Predicted U: {}", U)

    return errors, states # Return errors(for calculating loss) and states(for plotting)

def loss_func(params):
    errors, _ = run_one_epoch(params) # Retrieve errors from running the system one epoch.
    MSE = jnp.mean(jnp.square(errors)) # Calculate mean squared error
    if debug:
        jax.debug.print("Errors: {}", errors)
        jax.debug.print("MSE: {}", MSE)
    return MSE

Kp_history = np.zeros(epochs)
Kd_history = np.zeros(epochs)
Ki_history = np.zeros(epochs)

def updateParams(controller : Controller, params, gradients, epoch):
    # Helper function to visualise NN gradients (AI generated)
    def visualize_gradients_terminal(gradients):
        """
        Prints gradients as a textual visualization of connections in a neural network.

        Args:
            gradients: A list of tuples/lists representing the gradients for weights and biases
                    in each layer. Each tuple contains two elements:
                    - Weight gradients (2D array)
                    - Bias gradients (1D or 2D array)
        """
        for layer_idx, (weight_grad, bias_grad) in enumerate(gradients):
            print(f"\nLayer {layer_idx + 1} Gradients:")
            print("-" * 40)

            # Number of neurons in the current and next layer
            num_current_neurons = weight_grad.shape[0]
            num_next_neurons = weight_grad.shape[1]

            # Iterate over each neuron in the current layer
            for i in range(num_current_neurons):
                connections = []
                for j in range(num_next_neurons):
                    grad = weight_grad[i, j]
                    # Represent the connection visually with its gradient value
                    if grad > 0:
                        connections.append(f"({i} -> {j}: +{grad:.4f})")
                    elif grad < 0:
                        connections.append(f"({i} -> {j}: {grad:.4f})")
                    else:
                        connections.append(f"({i} -> {j}:  0.0000)")

                # Print connections for the current neuron
                print(f"Neuron {i}: " + ", ".join(connections))

            # Print bias gradients for this layer
            print("\nBias Gradients:")
            for j, bias_grad_value in enumerate(bias_grad[0]):
                print(f"Bias for Neuron {j}: {bias_grad_value:.4f}")

    # Check which controller is used, and updates params based on gradients.
    if isinstance(controller, NNController):
        updated_params = params
        for connection in range(len(params)):
            weights, biases = params[connection] 
            grad_weights, grad_biases = gradients[connection]
            updated_params[connection] = [weights - lr * grad_weights, biases - lr * grad_biases]
        visualize_gradients_terminal(gradients=gradients)
        return updated_params

    if isinstance(controller, PIDController):
        Kp, Kd, Ki = params
        Kp_history[epoch] = Kp
        Kd_history[epoch] = Kd
        Ki_history[epoch] = Ki
        dKp, dKd, dKi = gradients
        print(" - - - - - - - Gradients - - - - - - - ")
        print(gradients)
        print(" - - - - - - -  Params   - - - - - - - - ")
        print(params)
        return Kp - lr * dKp, Kd - lr * dKd, Ki - lr * dKi

losses = np.zeros(epochs) # Track losses across epochs to plot

for epoch in range(epochs):
    # Calculate gradients and value from running one epoch
    loss_value, gradients = jax.value_and_grad(loss_func, argnums=0)(params) 
    params = updateParams(controller=controller, params=params, gradients=gradients, epoch=epoch)
    lr = min(lr*10, conf.lrate_max) # Update the learning rate
    losses[epoch] = loss_value # Save the loss value for plotting

    if jnp.isnan(loss_value): # Check if the loss diverged due to overflow/underflow.
        print("Aborting: diverging loss")
        break
    print()
    print(f"Epoch: {epoch}")
    print(f"Loss: {loss_value}")
    print()

plt.xlabel("Epochs")
plt.title("PID constants")
plt.plot(Kp_history, label="Kp")
plt.plot(Kd_history, label="Kd")
plt.plot(Ki_history, label="Ki")
plt.legend()
plt.figure()

plt.title("MSE across epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.plot(losses)
plt.figure()

_, states = run_one_epoch(params)
plt.title("States from one simulated epoch")
plt.xlabel("Timestep")
plt.ylabel("Velocity")
plt.plot(states[:, 0], label="v_1")  # First column
plt.plot(states[:, 1], label="v_2")  # Second column
plt.figure()

plt.title("Distances from one simulated epoch")
plt.xlabel("Timestep")
plt.ylabel("Distance")
plt.plot(states[:, 2], label="d")  # Second column
plt.show()