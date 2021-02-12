import torch
import torch.nn.functional as F
import copy

from ReplayBuffer import ReplayBuffer
from networks import DQNNet
from params import MEMORY_SIZE, MIN_MEMORY_SIZE, BATCH_SIZE, GAMMA, LR, \
    STEPS_PER_UPDATE, STEPS_PER_TARGET_UPDATE

class DQN():
    def __init__(self, state_dim, action_dim):
        self.steps = 0 # Do not change
        self.memory = ReplayBuffer(capacity=MEMORY_SIZE)

        self.net = DQNNet(state_dim, action_dim)
        self.target_net = copy.deepcopy(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)

    def consume_transition(self, transition):
        self.memory.push(*transition)

    def sample_batch(self):
        return self.memory.sample(batch_size=BATCH_SIZE)

    def _calc_loss(self, batch):
        state, action, next_state, reward, done = batch

        state_action_values = self.net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_values = self.target_net(next_state).max(1)[0]
            expected_state_action_values = GAMMA * next_values * (1 - done) + reward

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        return loss
        
    def train_step(self, batch):
        loss = self._calc_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def act(self, state):
        state = torch.tensor(state)
        with torch.no_grad():
            return self.net(state.unsqueeze(0)).max(1)[1].item()

    def update(self, transition):
        self.consume_transition(transition)
        if len(self.memory) < MIN_MEMORY_SIZE: # updated
            return

        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    # save full model
    def save(self):
        torch.save(self.net, "agent.pkl")
