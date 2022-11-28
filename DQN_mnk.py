#%%
import math, random
from collections import namedtuple, deque
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import game

env = game.mnk_env(3, 3, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
	def __init__(self, m, n, k):
		super(DQN, self).__init__()

		f_1, f_2, f_3 = 4, 16, 32
		k = k + 1 if k % 2 == 0 else k

		self.conv1 = nn.Conv2d(1, f_1, k, padding = k//2)
		self.bn1 = nn.BatchNorm2d(f_1)
		self.conv2 = nn.Conv2d(f_1, f_2, k, padding = k//2)
		self.bn2 = nn.BatchNorm2d(f_2)
		self.conv3 = nn.Conv2d(f_2, f_3, k, padding = k//2)
		self.bn3 = nn.BatchNorm2d(f_3)
		self.head = nn.Linear(m * n * f_3, m * n)

	def forward(self, x):
		x = x.to(device)
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		return self.head(x.view(x.size(0), -1))

env.reset()

BATCH_SIZE = 128
GAMMA = 0.5
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

m, n = env.shape
k = env.k

policy_net = DQN(m, n, k).to(device)
target_net = DQN(m, n, k).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            result = policy_net(state)
    else:
        result = torch.randn(1, m * n)

    result[0][torch.flatten(state) != 0] = -np.inf
    return torch.tensor([[result.argmax()]])

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    states = torch.cat(batch.state)
    actions = torch.cat(batch.action)
    rewards = torch.cat(batch.reward)
    next_states = torch.cat(batch.next_state)

    current_q = policy_net(states).gather(1, actions)
    max_next_q = policy_net(next_states).detach().max(1)[0]
    expected_q = rewards + (GAMMA * max_next_q)

    loss = F.mse_loss(current_q.squeeze(), expected_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#%%
num_episodes = 1000
for i_episode in range(num_episodes):
    print(f'{i_episode}/{num_episodes}')
    env.reset()
    state = env.get_state()
    for t in count():
        action = select_action(state)
        new_state, reward, done = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        memory.push(state, action, new_state, reward)
        state = new_state

        optimize_model()
        if done:
            break

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')

#%%
env.reset()
print(env.get_obs(True))
done = False
while not done:
    if env.turn > 0:
        action = int(input())
        state, reward, done = env.step(action)
        print(f'turn: {env.turn}, action: {action}, reward: {reward}')
        if done:
            print(env.get_obs(True))
    else:
        action = select_action(state).item()
        state, reward, done = env.step(action)
        print(f'turn: {env.turn}, action: {action}, reward: {reward}')
        print(env.get_obs(True))
