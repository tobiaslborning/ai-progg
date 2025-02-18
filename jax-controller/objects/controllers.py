
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import numpy as np
from jaxnet import JAXNet
import pivotals as conf

class Controller(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def calcU(params, t : float, error_history : jnp.array) -> float:
        pass
    
    @abstractmethod
    def initializeParams(self) -> any:
        pass

class PIDController(Controller):
    def calcU(self, params, t : float, error_history : jnp.array) -> float:
        Kp, Kd, Ki = params
        E = error_history[int(t) + 1]
        dEdt = E - error_history[int(t)]
        sumE = jnp.sum(error_history) / len(error_history)
        return Kp * E + Kd * dEdt + Ki * sumE
    
    def initializeParams(self) -> any:
        return conf.Kp, conf.Kd, conf.Ki

class NNController(Controller):
    def __init__(self, layers, activation_func, init_value_range):
        pass
        self.jaxnet = JAXNet(layers, activation_func, init_value_range)
    
    def initializeParams(self) -> any:
        return self.jaxnet.gen_jaxnet_params()

    def calcU(self, params, t, error_history : jnp.array) -> float:
        E = error_history[int(t) + 1]
        dEdt = E - error_history[int(t)]
        sumE = jnp.sum(error_history) / len(error_history)
        # jax.debug.print("{}, {}, {}",E, dEdt, sumE)
        return self.jaxnet.predict(params, jnp.array([E,dEdt, sumE]))[0][0] 

    def updateParams(self, features, targets):
        self.jaxnet.train(features, targets, 10)