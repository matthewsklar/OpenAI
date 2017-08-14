import gym

class Agent:
    def __init__(self, action_space):
        self.action_space = action_space
        assert isinstance(action_space, gym.spaces.discrete.Discrete), 'unsupported action space for now'

    def act(self, state):
        return 0
