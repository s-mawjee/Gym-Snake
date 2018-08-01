import gym_snake.envs.snake_env
from gym_snake.envs.snake.view import LocalAction
from gym_snake.envs.snake.snake import Snake
import time
import numpy as np
env = gym_snake.envs.SnakeEnv(grid_size=[13, 13], unit_size=1, snake_size=2, unit_gap=0, action_transformer=LocalAction())
while True:
    print("RESET")
    obs = env.reset()
    assert (np.asarray(obs).shape == (27, 27, 1))
    while True:

        obs, rewards, done, info = env.step(LocalAction.RIGHT+1)
        print(np.asarray(obs).shape)
        assert(np.asarray(obs).shape == (27, 27, 1))
        #env.render()
        if done:
            break
        # obs, rewards, done, info = env.step(LocalAction.RIGHT+1)
        # if done:
        #     break
        # obs, rewards, done, info = env.step(LocalAction.LEFT+1)
        # if done:
        #     break
        #
        #


