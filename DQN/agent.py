import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        
    def act(self, state):
        state = torch.tensor(state)
        with torch.no_grad():
            return self.model(state.unsqueeze(0)).max(1)[1].item()

    def reset(self):
        pass
