import gym
from gym import Env

from Environment import Environment


class LunarLander(Environment):
    def __init__(self):
        self.__env: Env = gym.make('LunarLander-v2')

    def reset(self):
        super().reset()

        return self.__env.reset()

    def step(self, action):
        super().step()

        self.__env.render()
        return self.__env.step(action)

    def end(self):
        super().end()
        self.__env.close()
