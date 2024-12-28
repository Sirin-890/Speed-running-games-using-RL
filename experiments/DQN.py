import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import random
from copy import deepcopy
from torch import tensor
from collections import deque
import linecache
import os
import tracemalloc
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sys

N = 20
DISCOUNT = 0.2
DISCOUNT_DECAY = 0.001
ALPHA = 0.01
ALPHA_DECAY = 0.005
EPSILON = 0.1
EPSILON_DECAY = 0.01
LR_FREQ = 100
CPY_FREQ = 200
MODEL_SAVE = 1_000
END_EXPR = 3

class Qnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = nn.LazyLinear(8)
        self.l2 = nn.LazyLinear(6)
        self.l3 = nn.LazyLinear(6)
        self.l4 = nn.LazyLinear(4)


    def forward(self, X):
        X = nn.LeakyReLU()(self.l1(X))
        X = nn.LeakyReLU()(self.l2(X))
        X = nn.LeakyReLU()(self.l3(X))
        X = nn.LeakyReLU()(self.l4(X))

        return X
    
class ReplayBuffer(object):
    def __init__(self, size):
        self.buffers = torch.zeros((size, 8), dtype=torch.float32)
        self.buffera = torch.zeros(size, dtype=torch.int32)
        self.bufferr = torch.zeros(size, dtype=torch.float32)
        self.buffers_ = torch.zeros((size, 8), dtype=torch.float32)
        self.sz = 0
        self.size = size

    def push(self, sars):
        idx = self.sz % self.size
        self.buffers[idx] = sars[0]
        self.buffera[idx] = sars[1]
        self.bufferr[idx] = sars[2]
        self.buffers_[idx] = sars[3]

        self.sz+=1

        self.sz = min(self.size, self.sz)



    def sample(self, nItems):
        if self.sz < nItems:
            raise Exception("not enough items in the replay buffer")
        return random.sample([i for i  in range(self.sz)], nItems)

env = gym.make("LunarLander-v2", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env=env, video_folder="episode_replay", name_prefix="episode", episode_trigger=lambda x: x % 1_000 == 0)

target = None
behaviour = None
try:
    if sys.argv[1] == "load":
        target = Qnet()
        target.load_state_dict(torch.load(sys.argv[2])['state_dict'])
        behaviour = Qnet()
        behaviour.load_state_dict(torch.load(sys.argv[2])['state_dict'])
except:
    target = Qnet()
    behaviour = Qnet()

start, info = env.reset()

env.start_video_recorder()

replay = ReplayBuffer(1_000)

qu_sa = ReplayBuffer(N)

R_temp = ReplayBuffer(N)


device = torch.device("cuda" if torch.cuda.is_available else "cpu")

target.to(device)

behaviour_candidate = None

class dqn_optim(torch.optim.Optimizer):
    def __init__(self, params, lr, decay):
        defaults = dict(lr=lr, decay=decay)
        super().__init__(params, defaults)
        self.decay = decay
        self.lr = lr
    def step(self, replay_idx):
        for upd in replay_idx:
            maxReturn = replay.bufferr[upd] + (DISCOUNT ** (N-1)) * torch.max(behaviour.forward(replay.buffers_[upd]))
            Qstate = target.forward(replay.buffers_[upd].to(device))[replay.buffera[upd]]
            Qstate.backward(create_graph=True)
            with torch.no_grad():
                for group in self.param_groups:
                    for param in group['params']:
                        if param.grad is None:
                            continue
                        param.data -= self.lr * (maxReturn - Qstate) * param.grad

    
    def zero_grad(self):
        for group in self.param_groups:
            for param in group['params']:
                param.grad = None


writer = SummaryWriter()

episode_numbers = []
episode_rewards = []

max_episode_reward = -1000.00
qoptim = dqn_optim(target.parameters(), ALPHA, ALPHA_DECAY)

def step(optim, data):
    try:
        optim.step(replay.sample(data))
    except:
        optim.step(replay.sample(10))

for episode in tqdm(range(10_000)):
    start, info = env.reset()
    terminated = False
    truncated = False

    G = 0
    state = start
    n_steps = 0

    epi_reward = 0
    while not (truncated or terminated):
        action = env.action_space.sample() if np.random.random() > EPSILON else torch.argmax(target(tensor(state).to(device)).cpu()).item()

        n_steps += 1

        state, reward, terminated, truncated, info = env.step(action)

        epi_reward += reward

        state = tensor(state, dtype=torch.float32)

        if n_steps <= N:
            G = G + DISCOUNT ** (n_steps - 1) * reward
            qu_sa.push((state, action, 0, torch.zeros(8)))
            R_temp.push((torch.zeros(8), 0, reward, torch.zeros(8)))
        else:
            G = (G - R_temp.bufferr[0])/DISCOUNT + DISCOUNT ** (N-1) * reward
            replay.push((tensor(qu_sa.buffers[0]), qu_sa.buffera[0], G, tensor(state)))
            qu_sa.push((state, action, 0, torch.zeros(8)))
            R_temp.push((torch.zeros(8), 0, reward, torch.zeros(8)))


        if n_steps % LR_FREQ == 0:
            #experiment
            step(qoptim, 30)

    chk = 0
    for SA_idx in range(qu_sa.sz):
        G -= R_temp.bufferr[chk] * (DISCOUNT ** chk)
        G /= DISCOUNT
        replay.push((tensor(qu_sa.buffers[SA_idx]), qu_sa.buffera[SA_idx], G, tensor(qu_sa.buffers[qu_sa.sz-1]))) #error
        chk += 1

    for exp in range(0, END_EXPR):#experiment
        step(qoptim, 30)



    if episode % MODEL_SAVE == 0:
        model_state = {
            'episode' : episode,
            'state_dict': target.state_dict()
        }
        torch.save(model_state, "saved_models/" + "Qtarget_" + str(episode) + ".t7")

    

    EPSILON -= EPSILON_DECAY * EPSILON

    episode_rewards.append(epi_reward)

    if epi_reward > max_episode_reward:
        max_episode_reward = epi_reward
        behaviour_candidate = deepcopy(target)
        behaviour_candidate.cpu()


    if episode % CPY_FREQ == 0:
        behaviour = deepcopy(behaviour_candidate)

    if episode % 500 == 0:
        writer.add_scalar('Episode Reward', epi_reward, episode)

writer.close()

env.close_video_recorder()
env.close()

