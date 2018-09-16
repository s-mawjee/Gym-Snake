from gym_snake.envs.snake import Snake
from gym_snake.envs.snake import Grid
import copy
import matplotlib.pyplot as plt
import numpy as np
import time
class BaseView:
    def __init__(self, grid):
        self.grid = grid

    def get(self, action, snake_id):
        return grid

class LocalView:
    def __init__(self):
        pass
    def get_zero(self, grid):
        return np.expand_dims(np.zeros((grid.grid.shape[0]*2+1, grid.grid.shape[1]*2+1), np.float), -1)

    def get(self, grid, offset = (0,0), action = None):
        assert(np.array_equal(np.asarray(grid.grid.shape[0:2]) % 2, np.asarray([1, 1])))
        local_grid = np.zeros((grid.grid.shape[0]*2+1, grid.grid.shape[1]*2+1), np.float)
        grid_size = np.array(grid.grid.shape[0:2], dtype=np.int8)
        offset = np.roll(offset, 1)
        start = grid_size - offset

        assert(start[0] >= 0 and start[1] >= 0)
        end = start + grid_size

        local_grid[start[0]:end[0],  start[1]:end[1]] = grid.grid

        if action is not None:
            pass
            #print("ROT:" + str(action))
            #local_grid = np.rot90(local_grid, action)



        # Set own head to free space color. We do not want to have our own head to have as strong effect:
        local_grid[local_grid.shape[0] // 2, local_grid.shape[1] // 2] = grid.OWN_HEAD_COLOR
        #
        # plt.imshow(local_grid, interpolation='none')
        # plt.show()

        local_grid = np.expand_dims(local_grid, -1)
        return local_grid


class LocalAction:
    FWD = 0
    RIGHT = -1
    LEFT = 1
    def __init__(self):
        self.reset()

    def transform(self, local_action):
        assert(local_action == self.FWD or local_action == self.RIGHT or local_action == self.LEFT)
        action = (self.prev_action - local_action) % 4
        self.prev_action = action

        # print("LA " + str(local_action))
        # print("GA " + str(action))

        return action

    def reset(self):
        self.prev_action = Snake.DOWN