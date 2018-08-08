import gym_snake.envs.snake_env
from gym_snake.envs.snake.view import LocalAction
from gym_snake.envs.snake.snake import Snake
import time
import numpy as np

n_snakes = 1
env = gym_snake.envs.SnakeEnv(grid_size=[13, 13], unit_size=1, snake_size=4, unit_gap=0, n_foods=6, n_snakes=n_snakes)
while True:
    obs = env.reset()
    action = [LocalAction.FWD + 1] * n_snakes
    obs, rewards, done, info = env.step(action)
    action = [LocalAction.RIGHT + 1] * n_snakes
    obs, rewards, done, info = env.step(action)
    action = [LocalAction.FWD + 1] * n_snakes
    obs, rewards, done, info = env.step(action)
    while True:
        action = [LocalAction.FWD+1]*n_snakes
        obs, rewards, done, info = env.step(action)
        print(np.asarray(obs).shape)
        #env.render()
        # obs, rewards, done, info = env.step(LocalAction.RIGHT+1)
        # if done:
        #     break
        # obs, rewards, done, info = env.step(LocalAction.LEFT+1)
        # if done:
        #     break
        #
        #


