import pybullet_envs
import gym
import torch
import numpy as np
import random

from SAC import SAC
from params import ENV_NAME, TRANSITIONS, START_STEPS


def evaluate_policy(env, agent, episodes):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        while not done:
            state, reward, done, _ = env.step(agent.act(state, train=False))
            total_reward += reward
        returns.append(total_reward)
    return returns


def train():
    env = gym.make(ENV_NAME)
    test_env = gym.make(ENV_NAME)
    agent = SAC(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    state = env.reset()
    best_reward = -1e9
    episode_count = 0
    
    step = 0
    for i in range(TRANSITIONS):
        step += 1
        if i < START_STEPS:
            action = env.action_space.sample()
        else:
            action = agent.act(state, train=True)

        next_state, reward, done, _ = env.step(action)
        agent.update((state, action, next_state, reward, done if step != env._max_episode_steps else False))
        agent.optimize()
        
        if done: 
            episode_count += 1
            step = 0
        state = next_state if not done else env.reset()
        
        if (i + 1) % (TRANSITIONS//100) == 0:
            rewards = evaluate_policy(test_env, agent, 3)
            if (mean_reward := np.mean(rewards)) > best_reward:
                best_reward = mean_reward
                agent.save()
            print(f"Episode: {episode_count + 1}, Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            print("Alpha: {}".format(agent.alpha.item()))


def init_random_seeds(RANDOM_SEED):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


if __name__ == "__main__":
    init_random_seeds(23)
    train()

