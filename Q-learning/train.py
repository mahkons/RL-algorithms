import numpy as np
import torch
import copy
from collections import deque
import random
import os
import math

from .rl_env import MountainCar
from .QLearning import QLearning
from .params import EPS_START, EPS_END, EPS_DECAY, GRID_SIZE_X, GRID_SIZE_Y

def evaluate_policy(agent, episodes=5):
    env = MountainCar()
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, info = env.step(agent.act(state))
            total_reward += info['true_reward']
        returns.append(total_reward)
    return returns


def train(ql, steps):
    trajectory = []
    env = MountainCar()
    state = env.reset()


    ql.qlearning_estimate = np.load(__file__[:-8] + "/agent.npz")
    ql.qlearning_estimate = ql.qlearning_estimate.f.arr_0

    best = -1e9

    for step in range(steps):
        total_reward = 0
        
        #Epsilon-greedy policy
        EPS_THRESHOLD = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step / EPS_DECAY)
        if random.random() < EPS_THRESHOLD:
            action = env.random_action()
        else:
            action = ql.act(state)

        next_state, reward, done, _ = env.step(action)
        trajectory.append((state, action, next_state, reward, done))
        
        if done:
            for transition in reversed(trajectory):
                ql.update(transition)
            trajectory = []
        
        state = next_state if not done else env.reset()
        
        if (step + 1) % (steps//100) == 0:
            rewards = evaluate_policy(ql, 200)
            print(f"Step: {step+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            if np.mean(rewards) > best:
                best = np.mean(rewards)
                ql.save(path=os.curdir)

if __name__ == "__main__":
    ql = QLearning(state_dim=GRID_SIZE_X*GRID_SIZE_Y, action_dim=3)
    train(ql, 10000000)
