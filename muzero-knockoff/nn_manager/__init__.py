"""
Module for deploying, storing, and training neural networks  

The MuZero network architecture consists of three networks:   
1. Representation network  
2. Prediction network  
3. Dynamics network  

Together, these three networks make large one network with two inference methods.  
initial_inference(observation) : repnet + prednet  
and  
recurrent_inference(hidden_state) : dynnet + prednet  

Neural network output for uMCTS. Are packed inside a NetworkOutput object.  

Attributes:
    value (float): Estimated value of the current state
    reward (float): Immediate reward for the current state
    policy_logits (Dict[Action, float]): Action probabilities (logits)
    hidden_state (torch.Tensor): Internal representation state

unpack like this:  
value, reward, policy_logits, hidden_state = NetworkOutput()
  
"""