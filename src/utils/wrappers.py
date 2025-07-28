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

class SeaquestFeatureWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)

    def observation(self, obs):
        ram = self.unwrapped.ale.getRAM()

        # === Core values ===
        player_x = ram[54] / 255.0           # Horizontal position
        player_y = ram[29] / 255.0           # Vertical position
        num_divers = ram[61] / 6.0           # Max 6 divers
        oxygen = ram[102] / 255.0            # Oxygen bar
        ammo = ram[106] / 255.0              # Torpedo count

        # === Binary flags ===
        oxygen_low = 1.0 if ram[102] < 80 else 0.0
        ammo_empty = 1.0 if ram[106] == 0 else 0.0

        # === Player vertical zone ===
        is_near_top = 1.0 if ram[29] < 60 else 0.0
        is_near_bottom = 1.0 if ram[29] > 180 else 0.0
        is_centered = 1.0 if 100 <= ram[29] <= 140 else 0.0

        # === Time ticking (approx) ===
        timer_warning = 1.0 if ram[74] < 50 else 0.0

        # === Approx enemy proximity ===
        enemy_flags = []
        for i in range(42, 46):  # X positions of nearby enemies (RAM[42–45])
            enemy_x = ram[i]
            near = abs(enemy_x - ram[54]) < 20
            enemy_flags.append(1.0 if near else 0.0)

        features = np.array([
            player_x,
            player_y,
            num_divers,
            oxygen,
            ammo,
            *enemy_flags,
            is_near_top,
            is_near_bottom,
            is_centered,
            oxygen_low,
            ammo_empty,
            timer_warning,
        ], dtype=np.float32)

        return features