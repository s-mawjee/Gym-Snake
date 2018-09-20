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

    def __init__(self, grid_size=[13,13], unit_size=10, unit_gap=1, n_snakes=1, n_foods=1, random_init=True):
        self.grid_size = grid_size
        self.unit_size = unit_size
        self.unit_gap = unit_gap
        self.n_snakes = n_snakes
        self.n_foods = n_foods
        self.viewer = None
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-2, high=2,
            shape=(self.grid_size[0]*2+1, self.grid_size[1]*2+1, 3), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-2, high=2,
        #     shape=([27*27]), dtype=np.int8)
        self.random_init = random_init

        self.num_envs = n_snakes

    def step(self, action):
        obs, rewards, done, _ = self.controller.step(action)
        #print("STEP")
        # print(np.moveaxis(np.asarray(self.last_obs), 0, -1).shape)
        # print(str(rewards))
        # print(str(done))
        # print(rewards)
        info = [{"episode": {"l": 0, "r": np.sum(rewards)}}]
        # self.render()
        return obs, rewards, done, info

    def reset(self):
        self.controller = Controller(self.grid_size, self.unit_size, self.unit_gap, self.n_snakes, self.n_foods, random_init=self.random_init)

        obs = []

        for snakes in self.controller.snakes:
            lw = LocalView()
            obs.append(lw.get(self.controller.grid, self.controller.snakes[0].head))
        return  np.asarray(obs)

    def render(self, mode='human', close=False):
        if self.viewer is None:
            self.viewer = plt.imshow(np.squeeze(self.controller.grid.grid), interpolation='none')
            #self.viewer = plt.imshow(self.controller.grid.grid, interpolation='none')
        else:
            self.viewer.set_data(np.squeeze(self.controller.grid.grid))
            #self.viewer.set_data(np.squeeze(self.controller.grid.grid))

        print(self.controller.grid.grid.shape)
        plt.pause(0.01)
        plt.draw()

    def seed(self, x):
        pass


