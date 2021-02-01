import numpy as np
import gym

from .params import GRID_SIZE_X, GRID_SIZE_Y

# Simple discretization 
def _transform_state(state):
    state = (np.array(state) + np.array((1.2, 0.07))) / np.array((1.8, 0.14))
    x = min(int(state[0] * GRID_SIZE_X), GRID_SIZE_X-1)
    y = min(int(state[1] * GRID_SIZE_Y), GRID_SIZE_Y-1)
    return x + GRID_SIZE_X*y

# Wrapper that discretizes state space
#   and modifies reward
class MountainCar():
    def __init__(self):
        self.env = gym.make("MountainCar-v0")

    def reset(self):
        state = self.env.reset()
        return _transform_state(state)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        true_reward, reward = reward, self._transform_reward(state, reward, done)
        state = _transform_state(state)
        info['true_reward'] = true_reward

        return state, reward, done, info

    def random_action(self):
        return self.env.action_space.sample()

    def _transform_reward(self, state, reward, done):
        return reward + abs(state[1]) + (done and state[0] > 0.5) * 100

