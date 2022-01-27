import math
import random

import torch

from Agent import Agent
from PolicyNetwork import DQN


def get_device(use_gpu):
    if use_gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    return device


class TestingQLearningAgent(Agent):

    def __init__(self, **kwargs):
        self.__num_actions = kwargs.get('num_actions', 3)

        self.__device = get_device(kwargs.get('use_gpu', False))
        self.__policy_network = DQN(self.__num_actions, self.__device).to(self.__device)

        self.__policy_network.load_state_dict(torch.load('base_policy.pth'))

        self.__cpu = 'cpu'

    def __process_image(self, image):
        image = torch.tensor(image)
        image = image.unsqueeze(0)

        return image

    def __select_action(self, state):
        with torch.no_grad():
            action = self.__policy_network(
                state
            ).argmax(dim=1).view(1, -1)

        return action

    def start(self, state):
        super().start(state)

        state = self.__process_image(state)
        action = self.__select_action(state)

        return action.item()

    def step(self, reward, state):
        super().step(reward, state)

        state = self.__process_image(state)
        action = self.__select_action(state)

        return action.item()

    def end(self, reward):
        super().end(reward)

        return None, None, None, None

    def get_message(self):
        super().get_message()

    def save(self, version):
        torch.save(self.__policy_network.state_dict(), f'model_{version}.pth')
