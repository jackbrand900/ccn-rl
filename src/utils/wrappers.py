import numpy as np
from gymnasium import Wrapper, spaces


class RAMObservationWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(128,), dtype=np.float32)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        ram = self.env.unwrapped.ale.getRAM().astype(np.float32) / 255.0
        return ram, info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM().astype(np.float32) / 255.0
        return ram, reward, terminated, truncated, info
