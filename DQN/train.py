import numpy as np
import gym
import random
import torch
import math

from DQN import DQN
from params import TRANSITIONS, EPS_START, EPS_END, EPS_DECAY

def evaluate_policy(agent, episodes=5):
    env = gym.make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    env.close()
    return returns


def init_random_seeds(RANDOM_SEED):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def train(transitions):
    env = gym.make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    
    state = env.reset()
    for i in range(transitions):
        EPS_THRESHOLD = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * i / EPS_DECAY)
        if random.random() < EPS_THRESHOLD:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
        if (i + 1) % (transitions//100) == 0:
            rewards = evaluate_policy(dqn, 10)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            dqn.save()

    env.close()


if __name__ == "__main__":
    init_random_seeds(23)
    train(TRANSITIONS)
