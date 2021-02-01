import random
import numpy as np
import os
from .rl_env import _transform_state

class Agent:
    def __init__(self):
        self.qlearning_estimate = np.load(__file__[:-8] + "/agent.npz")
        self.qlearning_estimate = self.qlearning_estimate.f.arr_0
        
    def act(self, state):
        state = _transform_state(state)
        return np.argmax(self.qlearning_estimate[state])

    def reset(self):
        pass
