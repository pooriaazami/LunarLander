import random
from collections import deque, namedtuple

MemoryElement = namedtuple('Transition', ['last_state', 'action', 'current_state', 'reward'])


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(MemoryElement(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
