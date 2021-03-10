import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl", map_location="cpu")
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float)
            _, mean, _, _ = self.model.sample(state)
        return torch.tanh(mean[0]).numpy()

    def reset(self):
        pass
