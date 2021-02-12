import torch
import torch.nn as nn

from params import HIDDEN_LAYERS

def create_linear_layers(layers_sz):
    layers = list()
    for in_sz, out_sz in zip(layers_sz, layers_sz[1:]):
        layers.append(nn.Linear(in_sz, out_sz))
        layers.append(nn.LeakyReLU(inplace=True))

    layers.pop() # last nonlinearity
    return layers


class DQNNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNet, self).__init__()
        self.seq = nn.Sequential(*create_linear_layers([state_dim] + HIDDEN_LAYERS + [action_dim]))

    def forward(self, state):
        return self.seq(state)

