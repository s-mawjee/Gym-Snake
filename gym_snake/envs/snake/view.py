from gym_snake.envs.snake import Snake
import numpy as np

class BaseView:
    def __init__(self, grid):
        self.grid = grid
        self.prev_action = Snake.DOWN

    def get(self, offset=(0, 0), action=None):
        self.prev_action = action
        return self.grid.grid

class LocalView:
    def __init__(self, grid):
        self.grid = grid
        self.prev_action = Snake.DOWN

    def get(self, offset = (0,0), action = None ):
        assert(np.array_equal(np.asarray(self.grid.grid.shape[0:2]) % 2, np.asarray([1, 1])))
        local_grid = np.zeros((self.grid.grid.shape[0]*2+1, self.grid.grid.shape[1]*2+1), np.int8)
        grid_size = np.array(self.grid.grid.shape[0:2], dtype=np.int8)
        offset = np.roll(offset, 1)
        start = grid_size - offset

        assert(start[0] >= 0 and start[1] >= 0)
        end = start + grid_size

        local_grid[start[0]:end[0],  start[1]:end[1]] = self.grid.grid

        # print("ACITON "+str(action))
        if(action):
            local_grid = np.rot90(local_grid, -self.get_rotation(action))

        self.prev_action = action
        # plt.imshow(local_grid, interpolation='none')
        # plt.show()

        return local_grid

    def get_rotation(self, action):
        assert(np.abs(self.prev_action - action) != 2)
        return -(self.prev_action - action)

class LocalAction:
    FWD = 0
    RIGHT = -1
    LEFT = 1
    def __init__(self):

        self.prev_action = Snake.DOWN

    def transform(self, local_action):
        assert(local_action == self.FWD or local_action == self.RIGHT or local_action == self.LEFT)
        action = (self.prev_action - local_action) % 4
        self.prev_action = action
        return action