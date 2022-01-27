import gym
from gym import Env

from LunarLanderEnvironment import LunarLander
from Glue import Glue
from QLearningAgent import QLearningAgent

import matplotlib.pyplot as plt
from tqdm import tqdm

from TestAgent import TestingQLearningAgent


def calculate_new_values(numerator, denominator, reward, alpha):
    denominator *= 1 - alpha
    denominator += 1 - alpha

    numerator *= 1 - alpha
    numerator += reward

    return numerator, denominator


def train(num_episodes):
    agent_configs = {
        'gamma': .999,
        'num_actions': 4,
        'initial_epsilon': 1.,
        'last_epsilon': .01,
        'epsilon_decay_rate': 5000,
        'batch_size': 256,
        'target_update_rate': 10,
        'buffer_capacity': 100000,
        'lr': 1e-3,
        'use_gpu': True,
        'load_weights': False
    }

    agent = QLearningAgent(**agent_configs)
    env = LunarLander()

    glue = Glue(env, agent)

    rewards_cache = []
    short_term_ema = []
    long_term_ema = []
    loss_cache = []

    short_term_numerator, short_term_denominator = 0, 0
    long_term_numerator, long_term_denominator = 0, 0
    short_term_alpha = 0.1
    long_term_alpha = 0.01

    fig, axs = plt.subplots(2, 1)

    plt.title('rewards (0.9)')

    for i in tqdm(range(1, num_episodes + 1)):
        _, average, epsilon, loss, random, greedy = glue.run_episode()

        if i % 100 == 0:
            agent.save(i // 100)

        rewards_cache.append(average)

        short_term_numerator, short_term_denominator = calculate_new_values(
            short_term_numerator, short_term_denominator, average, short_term_alpha
        )

        long_term_numerator, long_term_denominator = calculate_new_values(
            long_term_numerator, long_term_denominator, average, long_term_alpha
        )

        short_term_ema.append(short_term_numerator / short_term_denominator)
        long_term_ema.append(long_term_numerator / long_term_denominator)

        loss_cache.extend(loss)

        if len(rewards_cache) > 100:
            rewards_cache.pop(0)
            short_term_ema.pop(0)
            long_term_ema.pop(0)

        if len(loss_cache) > 1000:
            loss_cache = loss_cache[-1000:]

        axs[0].cla()
        axs[1].cla()
        # plt.clf()
        axs[0].plot(rewards_cache, color='blue', label='reward')
        axs[0].plot(short_term_ema, color='red', label='short term EMA')
        axs[0].plot(long_term_ema, color='green', label='long term EMA')

        axs[1].plot(loss_cache, label='loss')

        axs[0].legend()
        axs[1].legend()

        fig.suptitle(f'epsilon = {epsilon} episode: {i}')
        plt.pause(0.01)

    glue.end()
    plt.show()


def test():
    agent_configs = {
        'num_actions': 4,
        'use_gpu': True,
        'load_weights': True,
    }

    agent = TestingQLearningAgent(**agent_configs)
    env = LunarLander()

    glue = Glue(env, agent)

    while True:
        glue.run_episode()



def main():
    # train(500)
    test()
    # env: Env = gym.make('LunarLander-v2')
    # print(env.action_space)
    # print(env.observation_space)


if __name__ == '__main__':
    main()
