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
    def __init__(self, grid):
        #super.__init__(BaseView, grid)
        self.grid = grid

    def get(self, offset, action):
        #assert(self.grid.grid.shape % 2 == [0, 0])
        print(self.grid)
        local_grid = np.zeros((self.grid.grid.shape[0]*2, self.grid.grid.shape[1]*2, self.grid.grid.shape[2]), np.uint8)
        grid_size = np.array(self.grid.grid.shape[0:2], dtype=np.uint8)
        start = grid_size - offset

        assert(start[0] >= 0 and start[1] >= 0)
        end = start + grid_size
        print(self.grid.grid.shape)
        print(start)
        print(end)
        local_grid[start[0]:end[0],  start[1]:end[1]] = self.grid.grid

        print(local_grid[0, 0])

        #np.roll(local_grid, -offset[0], axis = 0)
        #np.roll(local_grid, offset[1], axis = 1)


        plt.imshow(local_grid, interpolation='none')
        plt.show()
        # plt.imshow(self.grid.grid, interpolation='none')
        # plt.show()

        #

        #if action == Snake.UP:


