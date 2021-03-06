import torch
import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0

    def push(self, state, action, next_state, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(
            torch.tensor(state, dtype=torch.float),
            torch.tensor(action, dtype=torch.float),
            torch.tensor(next_state, dtype=torch.float),
            torch.tensor(reward, dtype=torch.float),
            torch.tensor(done, dtype=torch.float),
        )
        
        self.position += 1
        if self.position == self.capacity:
            self.position = 0
            
    def sample(self, batch_size, device=torch.device("cpu")):
        return self.get_transitions(self.sample_positions(batch_size), device)

    def get_transitions(self, positions, device):
        transitions = [self.memory[pos] for pos in positions]

        batch = Transition(*zip(*transitions))
        state = torch.stack(batch.state).to(device)
        action = torch.stack(batch.action).to(device)
        reward = torch.stack(batch.reward).to(device)
        next_state = torch.stack(batch.next_state).to(device)
        done = torch.stack(batch.done).to(device)

        return state, action, next_state, reward, done

    def sample_positions(self, batch_size):
        return random.sample(range(len(self.memory)), batch_size)

    def __len__(self):
        return len(self.memory)

    def clean(self):
        self.memory = list()
        self.position = 0
