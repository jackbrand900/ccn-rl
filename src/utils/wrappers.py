import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper, spaces


class RAMObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(128,), dtype=np.float32)

    def observation(self, obs):
        return self.unwrapped.ale.getRAM().astype(np.float32) / 255.0

class FreewayFeatureWrapper(ObservationWrapper):
    """
    Extracts symbolic features from Atari RAM for Freeway:
    - Player Y position (1)
    - Crossings (1)
    - Car X positions per lane (up to 2 per lane × 8 lanes = 16)
    - Car speeds per lane (approximate, 8)
    Total: 26 features
    """

    def __init__(self, env):
        super().__init__(env)
        self.num_lanes = 8
        self.max_cars_per_lane = 2  # based on known RAM layout
        self.car_pos_start = 40  # RAM[40-47]: main car x-positions per lane
        self.car_speed_start = 28  # RAM[28-31]: used for car movement
        self.prev_car_positions = np.zeros((self.num_lanes,), dtype=np.float32)

        # Feature vector: [player_y, crossings, car_pos..., car_speeds...]
        dim = 1 + 1 + self.num_lanes * self.max_cars_per_lane + self.num_lanes
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)

    def observation(self, obs):
        ram = self.unwrapped.ale.getRAM()

        # Player Y position (RAM[35]) normalized
        player_y = ram[35] / 255.0

        # Chicken crossings (RAM[83]) normalized
        crossings = ram[83] / 255.0

        # Car positions (RAM[40–47]), possibly RAM[120–127] also relevant for other cars
        car_positions = []
        for lane in range(self.num_lanes):
            base = self.car_pos_start + lane
            # Normalize x-pos by 160 (screen width)
            x1 = ram[base] / 160.0
            x2 = ram[(base + 8) % 128] / 160.0  # RAM[48–55] wrap-around for potential second car
            car_positions.extend([x1, x2])

        # Car speeds (approx): estimate via difference from last frame
        car_speeds = []
        for lane in range(self.num_lanes):
            curr = ram[self.car_pos_start + lane] / 160.0
            prev = self.prev_car_positions[lane]
            speed = (curr - prev) * 10  # scale diff to approx velocity
            car_speeds.append(np.clip(speed, -1.0, 1.0))  # normalize
            self.prev_car_positions[lane] = curr

        features = np.array([player_y, crossings] + car_positions + car_speeds, dtype=np.float32)
        return features
