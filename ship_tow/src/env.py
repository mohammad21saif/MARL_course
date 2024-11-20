from gymnasium import gym
from gymnasium import spaces
import numpy as np

class ShipTowEnv(gym.Env):
    def __init__(self, grid_size, ship_dim, tugboat_dim, max_rope_length, dock_position, target_position):
        """
        Observation Space:
        Type: Box(11)

        Action Space:
        Type: Box(5)

        """
        self.grid_size = grid_size
        self.ship_dim = ship_dim # [length, breadth]
        self.tugboat_dim = tugboat_dim # [length, breadth]
        self.max_rope_length = max_rope_length
        self.dock_position = dock_position
        self.target_position = target_position

        self.front_offset = None # ship length/2
        self.back_offset = None # -ship length/2

        # [xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, ds, l]
        #TODO: Fix low and high values
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                            high=np.array([100.0, 100.0, 2 * np.pi, 100.0, 100.0, 2 * np.pi, 100.0, 100.0, 2 * np.pi, 100.0, 100.0]),
                                            dtype=np.float32)
        
        #TODO: first try with action space consisting of x, y velocity components of tugboats
        # [Ft1, Ft2, alphat1, alphat2, taus]
        self.action_space = spaces.Box(low=np.array([0.0, 0.0, -np.pi, -np.pi, 0.0]),
                                       high=np.array([100.0, 100.0, np.pi, np.pi, 10.0]),
                                       dtype=np.float32)
        
        self.reset()


    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        #TODO: Fix according to max rope length.
        # [xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, ds, l]
        self.state = np.array([50.0, 50.0, 0.0, 55.0, 55.0, 0.0, 45.0, 45.0, 0.0, 50.0, 3])
        return self.state


    def step(self, action, current_state):
        Ft1, Ft2, alphat1, alphat2, taus = action
        xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, ds, l = current_state

        


    def _get_rewards(self, old_ds, new_ds, thetas, xs, ys):
        """
        +100 reward for reaching at target position in right alignment.
        -1 for each step.
         OR
        +100 reward for reaching at target position in right alignment.
        -2 for each step.
        +1 for reducing distance between dock and ship.
        """
        pass


    def render(self, mode='human'):
        pass
