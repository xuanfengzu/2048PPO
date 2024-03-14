import os
import random

import torch
from torch import nn
from torch.nn import functional as F

from Env import Env
from Net import PolicyNet, ValueNet
from utils import compute_advantage


class PPO:
    def __init__(self, in_channels, out_channels, num_hidden, num_heads, out_features, dropout, actor_lr, crtic_lr, lmbda, epochs, eps, gamma, device, epsilon_optimizer, bias=False, action_dim=4):
        self.actor = PolicyNet(in_channels, num_hidden, num_heads,
                               out_features, dropout, out_channels, bias, action_dim).to(device)
        self.critic = ValueNet(in_channels, num_hidden, num_heads,
                               out_features, dropout, out_channels, bias).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=crtic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.epsilon_optimizer = epsilon_optimizer

    def take_action(self, state):
        try:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            probs = self.actor(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
        except:
            print("NaN detected in probs, taking fallback action")
            action = torch.tensor([torch.randint(0, 4, (1,))])
        return action.item()

    def take_action_with_random_choice(self, state):
        epsilon = self.epsilon_optimizer.get_epsilon()
        if random.random() < epsilon:
            action = torch.tensor([torch.randint(0, 4, (1,))])
        else:
            try:
                state = torch.tensor(
                    [state], dtype=torch.float).to(self.device)
                probs = self.actor(state)
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample()
            except:
                print("NaN detected in probs, taking fallback action")
                action = torch.tensor([torch.randint(0, 4, (1,))])

        return action.item()

    def take_action_play(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        print(probs)
        action = probs.argmax()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(
            transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(
            transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(
            transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(
            transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(
            transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * \
            self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(
            self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(
            states).gather(1, actions)).detach()
        actor_loss_num, critic_loss_num = 0, 0

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(
                self.critic(states), td_target.detach()))
            actor_loss_num += actor_loss.item()
            critic_loss_num += critic_loss.item()
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # epsilon-greedy策略中的参数迭代器
        # self.epsilon_optimizer.step()

        return actor_loss_num / self.epochs, critic_loss_num / self.epochs

    def save_models(self, directory):
        # 检查目录是否存在，如果不存在，则创建
        if not os.path.exists(directory):
            os.makedirs(directory)

        actor_path = os.path.join(directory, 'actor.pth')
        critic_path = os.path.join(directory, 'critic.pth')

        # 保存 actor 和 critic 网络的参数
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Models saved to {directory}")

    def load_models(self, directory):
        actor_path = os.path.join(directory, 'actor.pth')
        critic_path = os.path.join(directory, 'critic.pth')

        # 检查文件是否存在
        if not os.path.exists(actor_path) or not os.path.exists(critic_path):
            print("Model files not found in directory:", directory)
            return

        # 加载模型参数
        self.actor.load_state_dict(torch.load(
            actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(
            critic_path, map_location=self.device))
        print(f"Models loaded from {directory}")
