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


class SeaquestRAMWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # Flatten all RAM indices of interest
        self.indices = (
                list(range(30, 34)) +  # enemy_obstacle_x
                [70, 97] +             # player_x, player_y
                list(range(71, 75)) +  # diver_or_enemy_missile_x
                [86, 87, 102, 103] +   # directions and oxygen
                [57, 58] +             # score
                [59, 62]               # lives and divers
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(self.indices),), dtype=np.float32
        )

    def _get_filtered_ram(self):
        ram = self.env.unwrapped.ale.getRAM()
        return ram[self.indices].astype(np.float32) / 255.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._get_filtered_ram(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._get_filtered_ram(), reward, terminated, truncated, info

class DemonAttackRAMWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # Flatten all RAM indices of interest
        self.indices = (
                [62, 22, 17, 18, 19] +
                [21, 69, 70, 71, 114]
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(self.indices),), dtype=np.float32
        )

    def _get_filtered_ram(self):
        ram = self.env.unwrapped.ale.getRAM()
        return ram[self.indices].astype(np.float32) / 255.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._get_filtered_ram(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._get_filtered_ram(), reward, terminated, truncated, info
