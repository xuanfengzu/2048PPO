import json
import time

import torch
from torch import nn
from tqdm import tqdm
import numpy as np

from Env import Env
from PPO import PPO
from PPO_CONFIG import *


env = Env(in_channels)
agent = PPO(in_channels, out_channels, num_hidden, num_heads, final_out_features,
            dropout, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device, 1)
agent.load_models("./")

state = env.reset()
env.print_board()
done = False
while not done:
    action = agent.take_action(state)
    print(f"ai选择动作{env.index_to_action[action]}")
    state, _, done, _, _, _ = env.step(action)
    env.print_board()
    # time.sleep(2)

# for name, param in agent.actor.named_parameters():
#     print(name, param.data)
