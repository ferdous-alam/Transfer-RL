__credits__ = ["Carlos Luis"]

from typing import Optional
from os import path

import numpy as np
import pygame
from pygame import gfxdraw

import gym
from gym import spaces
from gym.utils import seeding

# ------------ MODIFICATION ---------
from pathlib import Path
# -----------------------------------

class PendulumEnv(gym.Env):
    """
    ### Description

    The inverted pendulum swingup problem is a classic problem in the control literature. In this
    version of the problem, the pendulum starts in a random position, and the goal is to swing it up so
    it stays upright.

    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.

    ![Pendulum Coordinate System](./diagrams/pendulum.png)

    - `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta`: angle in radians.
    - `tau`: torque in `N * m`. Defined as positive _counter-clockwise_.

    ### Action Space
    The action is the torque applied to the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |


    ### Observation Space
    The observations correspond to the x-y coordinate of the pendulum's end, and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(angle)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |

    ### Rewards
    The reward is defined as:
    ```
    r = -(theta^2 + 0.1*theta_dt^2 + 0.001*torque^2)
    ```
    where `theta` is the pendulum's angle normalized between `[-pi, pi]`.
    Based on the above equation, the minimum reward that can be obtained is `-(pi^2 + 0.1*8^2 +
    0.001*2^2) = -16.2736044`, while the maximum reward is zero (pendulum is
    upright with zero velocity and no torque being applied).

    ### Starting State
    The starting state is a random angle in `[-pi, pi]` and a random angular velocity in `[-1,1]`.

    ### Episode Termination
    An episode terminates after 200 steps. There's no other criteria for termination.

    ### Arguments
    - `g`: acceleration of gravity measured in `(m/s^2)` used to calculate the pendulum dynamics. The default is
    `g=10.0`.

    ```
    gym.make('Pendulum-v1', g=9.81)
    ```

    ### Version History

    * v1: Simplify the math equations, no difference in behavior.
    * v0: Initial versions release (1.0.0)
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0, gravity=0.1, force=0.001, video_path=None):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.screen = None
        self.isopen = True

        self.screen_dim = 500

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # ---------- MODIFICATION --------------------------------------------------
        self.gravity = gravity
        self.force = force
        self.video_path = video_path
        self.screen_id = 0
        #----------------------------------------------------------------------------

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + self.gravity * thdot ** 2 + self.force * (u ** 2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img, (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2)
            )
            is_flip = self.last_u > 0
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        # ------ MODIFICATION ----------------------------------------------
        # pygame screen in a folder
        if self.video_path is not None:
            Path(self.video_path).mkdir(parents=True, exist_ok=True)
            s_val = str(self.screen_id).zfill(5)
            pygame.image.save(self.screen, '{}/scene-{}.png'.format(self.video_path, int(s_val)))
        self.screen_id += 1
        # -------------------------------------------------------------------

        if mode == "human":
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen


    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi