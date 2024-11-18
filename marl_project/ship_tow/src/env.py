from gymnasium import gym
from gymnasium import spaces
import numpy as np

class ShipTowEnv(gym.Env):
    def __init__(self):
        """
        Observation:
        Type: None
        
        """
        self.action_space = None
        self.obs_sapce = None

        

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self, mode='human'):
        pass
