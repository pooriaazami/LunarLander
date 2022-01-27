import math
import random

import torch
import numpy as np

from Agent import Agent
from Memory import ReplayMemory, MemoryElement
from PolicyNetwork import DQN
from torch.nn import functional as F

from torch import nn


def get_device(use_gpu):
    if use_gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    return device


class QLearningAgent(Agent):

    def __init__(self, **kwargs):
        self.__gamma = kwargs.get('gamma', 1.)

        self.__initial_epsilon = kwargs.get('initial_epsilon', kwargs.get('epsilon', 0.5))
        self.__last_epsilon = kwargs.get('last_epsilon', self.__initial_epsilon)
        self.__epsilon_decay_rate = kwargs.get('epsilon_decay_rate', 0)

        self.__target_update_rate = kwargs.get('target_update_rate', 10)
        self.__batch_size = kwargs.get('batch_size', 128)

        self.__num_actions = kwargs.get('num_actions', 3)

        self.__device = get_device(kwargs.get('use_gpu', False))
        self.__policy_network = DQN(self.__num_actions, self.__device).to(self.__device)
        self.__target_network = DQN(self.__num_actions, self.__device).to(self.__device)

        np.random.seed(kwargs.get('random_seed', 100))

        if kwargs.get('load_weights', False):
            self.__policy_network.load_state_dict(torch.load('base_policy.pth'))

        self.__policy_network.load_state_dict(self.__target_network.state_dict())

        self.__memory = ReplayMemory(kwargs.get('buffer_capacity', 1000))
        self.__optimizer = torch.optim.Adam(self.__policy_network.parameters(), lr=kwargs.get('lr', 1e-6))

        self.__cache = None
        self.__loss_cache = None
        self.__train = kwargs.get('train', True)

        self.__cpu = 'cpu'

        self.__num_steps = 0
        self.__last_state = None
        self.__last_action = None

        self.__random_action = 0
        self.__greedy_action = 0

    def __process_image(self, image):
        image = torch.tensor(image)
        image = image.unsqueeze(0)

        return image

    def __select_action(self, state):
        sample = random.random()
        if self.__epsilon_decay_rate != 0:
            epsilon_value = self.__last_epsilon + (self.__initial_epsilon - self.__last_epsilon) * math.exp(
                -1. * self.__num_steps / self.__epsilon_decay_rate)
        else:
            epsilon_value = self.__initial_epsilon

        self.__cache = epsilon_value

        self.__num_steps += 1
        if sample > epsilon_value:
            with torch.no_grad():
                action = self.__policy_network(
                    state
                ).argmax(dim=1).view(1, -1)

                self.__greedy_action += 1
        else:
            action = torch.tensor([
                [np.random.choice(self.__num_actions)]
            ], device=self.__device)

            self.__random_action += 1

        return action

    def __optimize_policy_network(self):
        if len(self.__memory) < self.__batch_size:
            return

        transitions = self.__memory.sample(self.__batch_size)
        batch = MemoryElement(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.current_state)), device=self.__device,
                                      dtype=torch.bool)
        non_final_next_state = torch.cat(
            [s for s in batch.current_state if s is not None]
        )

        last_state_batch = torch.cat(batch.last_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).to(self.__device)
        # current_states = torch.cat(batch.current_state)

        state_action_values = self.__policy_network(last_state_batch).gather(1, action_batch)

        next_state_value = torch.zeros(self.__batch_size, device=self.__device)
        next_state_value[non_final_mask] = self.__target_network(non_final_next_state).argmax(dim=1).detach().float()
        expected_state_action_values = next_state_value * self.__gamma + reward_batch
        # temp = self.__policy_network(non_final_next_state)
        # print(temp, temp.argmax(dim=1))
        reward_batch.to(self.__cpu)

        criterion = nn.HuberLoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.__optimizer.zero_grad()
        loss.backward()

        self.__loss_cache.append(loss.item())
        self.__optimizer.step()

        # for param in self.__policy_network.parameters():
        #     print(param.grad.data)

        if self.__num_steps % self.__target_update_rate == 0:
            self.__target_network.load_state_dict(self.__policy_network.state_dict())

    def start(self, state):
        super().start(state)

        self.__cache = None
        self.__loss_cache = []
        self.__random_action = 0
        self.__greedy_action = 0

        state = self.__process_image(state)
        action = self.__select_action(state)

        self.__last_state = state
        self.__last_action = action

        return action.item()

    def step(self, reward, state):
        super().step(reward, state)

        state = self.__process_image(state)
        self.__memory.push(
            self.__last_state,
            self.__last_action,
            state,
            torch.tensor([reward], dtype=torch.float32))
        # if self.__num_steps > 10:
        #     print(self.__memory.sample(10))
        action = self.__select_action(state)

        self.__last_state = state
        self.__last_action = action

        if self.__train:
            self.__optimize_policy_network()

        return action.item()

    def end(self, reward):
        super().end(reward)

        self.__memory.push(
            self.__last_state,
            self.__last_action,
            None,
            torch.tensor([reward], dtype=torch.float32))
        # print('here')
        self.__last_state = None
        self.__last_action = None

        return self.__cache, self.__loss_cache, self.__random_action, self.__greedy_action

    def get_message(self):
        super().get_message()

    def save(self, version):
        torch.save(self.__policy_network.state_dict(), f'model_{version}.pth')
