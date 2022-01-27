from Agent import Agent
from Environment import Environment


class Glue:
    def __init__(self, env: Environment, agent: Agent):
        self.env = env
        self.agent = agent

    def __start(self):
        observation = self.env.reset()
        action = self.agent.start(observation)

        return action

    def __step(self, state):
        pass

    def run_episode(self):
        action = self.__start()
        done = False

        reward_sum = 0
        while not done:
            observation, reward, done, message = self.env.step(action)
            action = self.agent.step(reward, observation)

            reward_sum += reward

        cache, loss, random, greedy = self.agent.end(reward)

        return message, reward_sum, cache, loss, random, greedy

    def end(self):
        self.env.end()
