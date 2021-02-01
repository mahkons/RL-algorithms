import numpy as np

from params import LR, GAMMA

class QLearning:
    def __init__(self, state_dim, action_dim):
        self.qlearning_estimate = np.zeros((state_dim, action_dim)) + 2.

    def update(self, transition):
        state, action, next_state, reward, done = transition
        next_value = GAMMA * np.max(self.qlearning_estimate[next_state]) * (1 - done) + reward
        self.qlearning_estimate[state, action] = (1 - LR) * self.qlearning_estimate[state, action] + LR * next_value

    def act(self, state):
        return np.argmax(self.qlearning_estimate[state])

    def save(self, path):
        np.savez("agent.npz", self.qlearning_estimate)
