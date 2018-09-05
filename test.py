import gym_snake.envs.snake_env
from gym_snake.envs.snake.view import LocalAction
from gym_snake.envs.snake.snake import Snake
import time
import numpy as np

n_snakes = 15
env = gym_snake.envs.SnakeEnv(grid_size=[13, 13], unit_size=1, unit_gap=0, n_foods=6, n_snakes=n_snakes)
while True:
    obs = env.reset()
    # action = [LocalAction.FWD + 1] * n_snakes
    # obs, rewards, done, info = env.step(action)
    # # env.render()
    # action = [LocalAction.RIGHT + 1] * n_snakes
    # obs, rewards, done, info = env.step(action)
    # # env.render()
    # action = [LocalAction.FWD + 1] * n_snakes
    # obs, rewards, done, info = env.step(action)
    # env.render()
    # while True:
    #     action = [LocalAction.FWD]*n_snakes
    #     obs, rewards, done, info = env.step(action)
    #     env.render()
    #     if all(done):
    #         time.sleep(1)
    #         print("Done")
    #         break

        # obs, rewards, done, info = env.step(LocalAction.RIGHT+1)
        # if done:
        #     break
        # obs, rewards, done, info = env.step(LocalAction.LEFT+1)
        # if done:
        #     break
        #
        #


