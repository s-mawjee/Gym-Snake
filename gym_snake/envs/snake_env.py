import os, subprocess, time, signal
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_snake.envs.snake import Controller
from gym_snake.envs.snake.view import LocalView

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: see matplotlib documentation for installation https://matplotlib.org/faq/installing_faq.html#installation".format(e))

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=[15,15], unit_size=10, unit_gap=1, snake_size=1, n_snakes=1, n_foods=1, random_init=True):
        self.grid_size = grid_size
        self.unit_size = unit_size
        self.unit_gap = unit_gap
        self.snake_size = snake_size
        self.n_snakes = n_snakes
        self.n_foods = n_foods
        self.viewer = None
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-2, high=2,
            shape=(self.grid_size[0]*2+1, self.grid_size[1]*2+1, 1), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-2, high=2,
        #     shape=([27*27]), dtype=np.int8)
        self.random_init = random_init

    def step(self, action):
        obs, rewards, done, info = self.controller.step(action)
        # print("STEP")
        # print(np.moveaxis(np.asarray(self.last_obs), 0, -1).shape)
        # print(str(rewards))
        # print(str(done))
        # print(str(info))
        return obs, rewards, done, info

    def reset(self):
        self.controller = Controller(self.grid_size, self.unit_size, self.unit_gap, self.snake_size, self.n_snakes, self.n_foods, random_init=self.random_init)

        obs = []

        for snakes in self.controller.snakes:
            lw = LocalView()
            obs.append(lw.get(self.controller.grid, self.controller.snakes[0].head))
        # print("RESET")
        return  np.asarray(obs)

    def render(self, mode='human', close=False):
        if self.viewer is None:
            self.viewer = plt.imshow(np.squeeze(self.controller.grid.grid), interpolation='none')
            #self.viewer = plt.imshow(self.controller.grid.grid, interpolation='none')
        else:
            self.viewer.set_data(np.squeeze(self.controller.grid.grid))
            #self.viewer.set_data(np.squeeze(self.controller.grid.grid))

        plt.pause(0.07)
        plt.draw()

    def seed(self, x):
        pass


