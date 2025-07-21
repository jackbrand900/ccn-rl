from gymnasium.envs.box2d.car_racing import CarRacing, WINDOW_W, WINDOW_H
import numpy as np
import pygame


class CarRacingWithTrafficLights(CarRacing):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.red_light_active = False
        self.step_counter = 0
        self.toggle_interval = 40  # Steps between toggles
        self.green_priority = 0.9  # 90% of the time, stay green
        self.wheel_on_grass = []
        self.on_grass = False  # Flag for whether the car is on grass or not

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.step_counter = 0
        self._toggle_red_light()
        return obs, info

    def _toggle_red_light(self):
        # Favor green over red
        self.red_light_active = np.random.rand() > self.green_priority

    def step(self, action):
        self.step_counter += 1

        if self.step_counter % self.toggle_interval == 0:
            self._toggle_red_light()

        obs, reward, terminated, truncated, info = super().step(action)
        self.wheel_on_grass = []  # ⬅️ Store booleans for each wheel

        for w in self.car.wheels:
            if not w.tiles:
                self.wheel_on_grass.append(True)
                continue
            on_grass = all(
                getattr(tile.userData, "road_friction", 0) < 0.8
                for tile in w.tiles
            )
            self.wheel_on_grass.append(on_grass)
            # if self.wheel_on_grass.count(True) > 0:
            #     print(f"  Any wheels on grass? {self.wheel_on_grass.count(True) > 0}")

        # print(f"[Step {self.step_counter}] Red Light: {self.red_light_active}")
        # print(f"  Wheels on grass: {self.wheel_on_grass}")
        # print(f"  → All wheels on grass? {self.on_grass}")

        self.on_grass = all(self.wheel_on_grass)
        return obs, reward, terminated, truncated, info

    def render(self):
        frame = super().render()

        if self.render_mode == "human" and self.red_light_active and hasattr(self, "screen"):
            overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            overlay.fill((255, 0, 0, 100))  # Red with 100 alpha
            self.screen.blit(overlay, (0, 0))
            pygame.display.flip()

        return frame
