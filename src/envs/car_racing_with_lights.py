from gymnasium.envs.box2d.car_racing import CarRacing, WINDOW_W, WINDOW_H
import numpy as np
import pygame


class CarRacingWithTrafficLights(CarRacing):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.red_light_active = False
        self.step_counter = 0
        self.next_toggle_step = 0
        self.green_priority = 0.9  # 90% chance of staying green
        self.wheel_on_grass = []
        self.on_grass = False

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.step_counter = 0
        self._toggle_red_light()
        self._set_next_toggle_step()
        return obs, info

    def _toggle_red_light(self):
        self.red_light_active = np.random.rand() > self.green_priority

    def _set_next_toggle_step(self):
        self.next_toggle_step = self.step_counter + np.random.randint(20, 201)

    def step(self, action):
        self.step_counter += 1

        if self.step_counter >= self.next_toggle_step:
            self._toggle_red_light()
            self._set_next_toggle_step()

        obs, reward, terminated, truncated, info = super().step(action)

        self.wheel_on_grass = []
        for w in self.car.wheels:
            if not w.tiles:
                self.wheel_on_grass.append(True)
                continue
            on_grass = all(
                getattr(tile.userData, "road_friction", 0) < 0.8
                for tile in w.tiles
            )
            self.wheel_on_grass.append(on_grass)

        self.on_grass = all(self.wheel_on_grass)
        return obs, reward, terminated, truncated, info

    def render(self):
        frame = super().render()

        if self.render_mode == "human" and self.red_light_active and hasattr(self, "screen"):
            overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            overlay.fill((255, 0, 0, 100))  # Red with transparency
            self.screen.blit(overlay, (0, 0))
            pygame.display.flip()

        return frame
