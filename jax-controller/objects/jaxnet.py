# Implementation of a neural network using JAX

import numpy as np
import jax.numpy as jnp
import jax

class JAXNet:
    def __init__(self, layers, activation_func, init_value_range):
        self.layers = layers
        self.min_init_val = init_value_range[0]
        self.max_init_val = init_value_range[1]
        if activation_func == "Sigmoid":
            self.activation_func = lambda x : 1 / (1 + jnp.exp(-x)) 
        elif activation_func == "ReLU":
            self.activation_func = lambda x : jnp.maximum(0,x)
        elif activation_func == "Tanh":
            self.activation_func = lambda x : jnp.tanh(x)
        elif activation_func == "Linear":
            self.activation_func = lambda x : x
    

    def gen_jaxnet_params(self): 
        sender = self.layers[0]
        params = []
        key = jax.random.PRNGKey(0) # Used for generating random jax-traceable arrays
        for receiver in self.layers[1:]:
            key, weight_key, bias_key = jax.random.split(key,3) # Used to split the key such that new values are generated
            weights = jax.random.uniform(key=weight_key,
                                         minval=self.min_init_val,
                                         maxval=self.max_init_val,
                                         shape=(sender,receiver))
            biases = jax.random.uniform(key=bias_key,
                                        minval=self.min_init_val,
                                        maxval=self.max_init_val,
                                        shape=(1,receiver))
            # weights = jnp.array(np.random.uniform(-0.1,0.1,(sender,receiver))) 
            # biases = jnp.array(np.random.uniform(-0.1,0.1,(1,receiver)))
            sender = receiver
            params.append([weights, biases])
        return params
    

    def predict(self, all_params, features):
        activations = features
        for weights, biases in all_params:
            activations = self.activation_func(jnp.dot(activations,weights) + biases) 
        return activations

    def calc_loss(params, features, targets, activation_func): 
        print("Features")
        print(features)
        print("Targets")
        print(targets)
        print("Params")
        print(params)
        batched_predict = jax.vmap(JAXNet.predict, in_axes=0)
        predictions = batched_predict(params, features, activation_func) 
        return jnp.mean(jnp.square(targets - predictions))
    
    def train_one_epoch(self, params, features, targets, lrate):
        mse, gradients = jax.value_and_grad(JAXNet.calc_loss)(params, features, targets, self.activation_func) 
        return [(w - lrate * dw, b - lrate * db) for (w, b), (dw, db) in zip(params, gradients)], mse
    
    def train(self, features, targets, epochs): 
        curr_params = self.params # Vet ikke om dette egt er nødvendig, men params vil bli tracet av JAX
        for _ in range(epochs):
            curr_params, mse = JAXNet.train_one_epoch(self, curr_params,features,targets, self.lrate) 
        self.params = curr_params
    
    