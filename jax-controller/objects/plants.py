# This file contains the general plant superclass, along with the three different plants
from abc import ABC, abstractmethod
import time
import numpy as np
import jax
import jax.numpy as jnp

class Plant(ABC):
    @abstractmethod
    def plant_timestep_change(
            self,
            t : float, 
            states : jnp.array,
            U : float, 
            D : float
    ):
        """
        Method to run the plant one timestep
        returns a tuple of the updated state, and the plant output.
        """
        pass

    @abstractmethod
    def init_state_array(self, timesteps) -> jnp.array:
        """
        Initializes the state array, and puts the initial state at the first index
        """
        pass

class BathtubPlant(Plant):
    """
    state: water height
    outputs: water height
    """
    def __init__(self, area, drain_area, target):
        self.area = area
        self.drain_area = drain_area
        self.target = target

    def init_state_array(self, timesteps) -> jnp.array:
        states = jnp.zeros(timesteps)
        return states.at[0].set(self.target)

    def get_target_value(self):
        return self.target

    def volume_timestep_change(
            self,
            t : float, 
            states : jnp.array,
            U : float, 
            D : float
        ):
        height = states[int(t)]
        if height < 0:
            return U + D[int(t)]
        exit_velocity = jnp.sqrt(2 * 9.81 * height)
        flow_rate = exit_velocity * self.drain_area
        return U + D[int(t)] - flow_rate
    
    def plant_timestep_change(
            self,
            t : float, 
            state : jnp.array,
            U : float, 
            D : float
        ):
        vol_change = self.volume_timestep_change(t, state, U, D)
        updated_state = state[int(t)] + (vol_change / self.area)
        output = updated_state
        return updated_state, output 

        # jax.debug.print("State height: {}", state[int(t)])
        # jax.debug.print("flow rate: {}",flow_rate)
        # jax.debug.print("U: {}", U)
        # jax.debug.print("D: {}", D)

class CournotPlant(Plant):
    """
    state: (c_1, c_2)
    output: Profit P
    """
    def __init__(self, p_max, c_m, target, c1_init, c2_init):
        self.p_max = float(p_max)
        self.c_m = float(c_m)
        self.target = float(target)
        self.c1_init = c1_init
        self.c2_init = c2_init

    def init_state_array(self, timesteps) -> jnp.array:
        init_state = jnp.array([self.c1_init, self.c2_init])
        state_array = jnp.zeros((timesteps,2))
        return state_array.at[0].set(init_state)
    
    def get_target_value(self):
        return self.target

    def p(self, q):
        return self.p_max - q

    def plant_timestep_change(
            self,
            t : float, 
            states : jnp.array,
            U : float, 
            D : float
        ):
        q1, q2 = states[int(t)]
        # jax.debug.print("q1 = {}, q2 = {} U = {} D = {}", q1, q2, U, D[t])
        q1 = U + q1 # Update q1
        q2 = D[t] + q2 # Update q2
        q = q1 + q2 
        P_1 = q1 * (self.p(q) - self.c_m) # Profit for prod 1
        updated_state = jnp.array([q1, q2])
        output = P_1
        return updated_state, output 
    

class TailgatingPlant(Plant):
    def __init__(self, v_1, v_2, v_lim, d_0, target_d):
        self.v_1 = v_1
        self.v_2 = v_2
        self.d_0 = d_0
        self.v_lim = v_lim
        self.target_d = target_d

    def init_state_array(self, timesteps) -> jnp.array:
        init_state = jnp.array([self.v_1, self.v_2, self.d_0])
        state_array = jnp.zeros((timesteps,3))
        return state_array.at[0].set(init_state)
    
    def get_target_value(self):
        return self.target_d

    def plant_timestep_change(
            self,
            t : float, 
            states : jnp.array,
            U : float, 
            D : float
        ):
        v1, v2, d = states[int(t)]
        # jax.debug.print("v1 = {}, v2 = {}, d={}, U = {} D = {}", v1, v2, d, U, D[t])
        v1 = U + v1 # Update v1
        v2 = D[t] + v2 # Update v2
        d_change = v2 - v1
        d = d + d_change
        updated_state = jnp.array([v1, v2, d])
        return updated_state, d 
