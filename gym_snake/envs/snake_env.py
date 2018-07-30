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

    def __init__(self, grid_size=[15,15], unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, n_foods=1, random_init=True, action_transformer=None):
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
        self.action_transformer = action_transformer

    def step(self, action):
        if self.action_transformer:
            action = + self.action_transformer.transform(action-1)
        self.last_obs, rewards, done, info = self.controller.step(action)
        # print("STEP")
        # print(np.moveaxis(np.asarray(self.last_obs), 0, -1).shape)
        # print(str(rewards))
        # print(str(done))
        # print(str(info))
        return  np.asarray(np.moveaxis(np.asarray(self.last_obs), 0, -1)), rewards, done, info

    def reset(self):
        self.controller = Controller(self.grid_size, self.unit_size, self.unit_gap, self.snake_size, self.n_snakes, self.n_foods, random_init=self.random_init)

        #self.last_obs = self.controller.grid.grid
        lw = LocalView(self.controller.grid)
        self.last_obs = lw.get()
        # print("RESET")
        # print(np.expand_dims(self.last_obs, -1).shape)
        return np.asarray(np.expand_dims(self.last_obs, -1))

    def render(self, mode='human', close=False):
        if self.viewer is None:
            self.viewer = plt.imshow(self.last_obs, interpolation='none')
        else:
            self.viewer.set_data(self.last_obs)
        plt.pause(0.1)
        plt.draw()

    def seed(self, x):
        pass


