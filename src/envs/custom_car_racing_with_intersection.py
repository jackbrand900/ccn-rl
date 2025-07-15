import math
import pygame
from pygame import gfxdraw
from gymnasium.envs.box2d.car_racing import CarRacing, TRACK_WIDTH, SCALE, WINDOW_W, WINDOW_H, VIDEO_W, VIDEO_H, STATE_W, STATE_H

class CustomCarRacing(CarRacing):
    def _create_track(self):
        self.track = []
        self.road = []
        self.road_poly = []

        spacing = 10.0
        num_tiles = 20
        width = TRACK_WIDTH
        center_x, center_y = 0.0, 0.0

        def add_connected_tile(x1, y1, x2, y2):
            angle = math.atan2(y2 - y1, x2 - x1)
            dx = width * math.cos(angle)
            dy = width * math.sin(angle)

            # Perpendicular offset
            ox = width * math.sin(angle)
            oy = -width * math.cos(angle)

            road1_l = (x1 - ox, y1 - oy)
            road1_r = (x1 + ox, y1 + oy)
            road2_l = (x2 - ox, y2 - oy)
            road2_r = (x2 + ox, y2 + oy)

            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], self.road_color))

        # Horizontal segment (left to right)
        for i in range(-num_tiles, num_tiles):
            x1 = center_x + i * spacing
            x2 = center_x + (i + 1) * spacing
            y = center_y
            add_connected_tile(x1, y, x2, y)

        # Vertical segment (top down)
        for i in range(0, num_tiles):
            y1 = center_y + i * spacing
            y2 = center_y + (i + 1) * spacing
            add_connected_tile(center_x, y1, center_x, y2)

        # Vertical segment (bottom up)
        for i in range(-num_tiles, 0):
            y1 = center_y + i * spacing
            y2 = center_y + (i + 1) * spacing
            add_connected_tile(center_x, y1, center_x, y2)

        # Car start position: far left of intersection, rotated 90 degrees clockwise
        start_x = center_x - num_tiles * spacing
        start_y = center_y
        start_heading = -math.pi / 2  # 90 degrees clockwise from facing center

        self.track = [(0.0, start_heading, start_x + TRACK_WIDTH, start_y)]

        # Optional: define intersection zone for use in logic
        self.intersection_zone = ((-10, -10), (10, 10))
        return True

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        return obs, info

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        # Zoom out view
        zoom = 1.0 * SCALE  # smaller than default (2.7 * SCALE)
        angle = -self.car.hull.angle

        # Center camera on car
        scroll_x = -self.car.hull.position[0] * zoom
        scroll_y = -self.car.hull.position[1] * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
        self.car.draw(self.surf, zoom, trans, angle, True)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self._render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen
