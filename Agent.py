class Agent:

    def start(self, state):
        """
        Agent should config it's important variables in this method

        :param state: initial state
        :return: first action that agent should take
        """
        pass

    def step(self, reward, state):
        """

        :param reward: last actions reward
        :param state: current state
        :return: action that should be taken in current state
        """
        pass

    def end(self, reward):
        """
        Agent should perform last updates of it's models in this method based on
        last reward ir receives that is the reward for action that caused episode to end.

        :param reward: last actions reward

        """
        pass

    def get_message(self):
        pass
