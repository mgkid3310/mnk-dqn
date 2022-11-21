#%%
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import game

env = game.mnk_env(5, 5, 3)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

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
		self.conv1 = nn.Conv2d(1, 4, k, padding = k//2)
		self.bn1 = nn.BatchNorm2d(4)
		self.conv2 = nn.Conv2d(4, 4, k, padding = k//2)
		self.bn2 = nn.BatchNorm2d(4)
		self.conv3 = nn.Conv2d(4, 4, k, padding = k//2)
		self.bn3 = nn.BatchNorm2d(4)
		self.head = nn.Linear(m * n * 4, m * n)

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
    # print(state)
    # print(result)
    return torch.tensor([[result.argmax()]])


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

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
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        new_state = env.get_state()
        memory.push(state, action, new_state, reward)
        state = new_state

        optimize_model()
        if done:
            episode_durations.append(t + 1)
            # plot_durations()
            break

    # print(t)
    # print(env.get_obs(True))

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')

#%%
env.reset()
done = False
while not done:
    if env.turn > 0:
        state, _, done, _ = env.step(int(input()))
        print(env.get_obs(True))
    else:
        state, _, done, _ = env.step(select_action(state).item())
        print(env.get_obs(True))
