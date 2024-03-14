import json

import torch
from torch import nn
from tqdm import tqdm
import numpy as np

from Env import Env
from PPO import PPO
from utils import EpsilonOptimizer
from PPO_CONFIG import *


def train_on_policy_agent(env, agent, num_episodes):
    result = {
        "return": [],
        "max_tile": [],
        "board_score": [],
        # "board_state":[],
        "turns": [],
        "ac_ls": [],
        "cr_ls": []
    }
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [],
                                   'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                turn = 0
                while not done:
                    turn += 1
                    action = agent.take_action(state)
                    next_state, reward, done, max_tile, board_score, board_state = env.step(
                        action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                result["return"].append(episode_return)
                result["max_tile"].append(max_tile)
                result["board_score"].append(board_score)
                # result["board_state"].append(board_state.tolist())
                result["turns"].append(turn)
                actor_loss, critic_loss = agent.update(transition_dict)
                result['ac_ls'].append(actor_loss)
                result['cr_ls'].append(critic_loss)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(result["return"][-10:]), 'turn': '%d' % np.mean(result["turns"][-10:]), 'board score': '%.3f' % np.mean(
                        result['board_score'][-10:]), 'max tile': '%.3f' % np.mean(result['max_tile'][-10:]), 'ac loss': '%.3f' % np.mean(result['ac_ls'][-10:]), 'cr loss': '%.3f' % np.mean(result['cr_ls'][-10:]), 'epsilon': '%.3f' % agent.epsilon_optimizer.get_epsilon()})
                pbar.update(1)
        agent.save_models("models")
        with open("result.json", "w", encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    agent.save_models("models")
    with open("result.json", "w", encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    return result


env = Env(in_channels)
epsilon_optimizer = EpsilonOptimizer()
agent = PPO(in_channels, out_channels, num_hidden, num_heads, final_out_features,
            dropout, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device, epsilon_optimizer)
agent.load_models("models")


result = train_on_policy_agent(env, agent, num_episodes=50000)

with open("result.json", "w", encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
