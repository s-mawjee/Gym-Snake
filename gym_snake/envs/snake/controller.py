from gym_snake.envs.snake import Snake
from gym_snake.envs.snake import Grid
from gym_snake.envs.snake.view import LocalView
from gym_snake.envs.snake.view import LocalAction
import numpy as np
import time
import matplotlib.pyplot as plt

class Controller():
    """
    This class combines the Snake, Food, and Grid classes to handle the game logic.
    """

    def __init__(self, grid_size=[30,30], unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, n_foods=1, random_init=True):

        assert n_snakes < grid_size[0]//3
        assert n_snakes < 25
        assert snake_size < grid_size[1]//2
        assert unit_gap >= 0 and unit_gap < unit_size

        self.done = False

        self.n_snakes = n_snakes
        self.snakes_remaining = n_snakes
        self.grid = Grid(grid_size, unit_size, unit_gap)

        self.snakes = []
        self.dead_snakes = []
        self.erased_snakes = []
        self.local_views = []
        self.local_actions = []
        for i in range(1,n_snakes+1):
            start_coord = [i*grid_size[0]//(n_snakes+1), snake_size+1]
            self.snakes.append(Snake(start_coord, snake_size))
            color = self.grid.HEAD_COLOR
            self.snakes[-1].head_color = color
            self.grid.draw_snake(self.snakes[-1], color)
            self.dead_snakes.append(None)
            self.erased_snakes.append(None)

            self.local_views.append(LocalView())
            self.local_actions.append(LocalAction())


        if not random_init:
            for i in range(2,n_foods+2):
                start_coord = [i*grid_size[0]//(n_foods+3), grid_size[1]-5]
                self.grid.place_food(start_coord)
        else:
            for i in range(n_foods):
                self.grid.new_food()

    def move_snake(self, direction, snake_idx):
        """
        Moves the specified snake according to the game's rules dependent on the direction.
        Does not draw head and does not check for reward scenarios. See move_result for these
        functionalities.
        """

        snake = self.snakes[snake_idx]

        # Cover old head position with body
        self.grid.cover(snake.head, self.grid.BODY_COLOR)
        # Erase tail without popping so as to redraw if food eaten
        self.grid.erase(snake.body[0])
        # Find and set next head position conditioned on direction
        snake.action(direction)

    def move_result(self, direction, snake_idx=0):
        """
        Checks for food and death collisions after moving snake. Draws head of snake if
        no death scenarios.
        """

        snake = self.snakes[snake_idx]
        if type(snake) == type(None):
            return 0

        # Check for death of snake
        if self.grid.check_death(snake.head):
            self.dead_snakes[snake_idx] = self.snakes[snake_idx]
            self.snakes[snake_idx] = None
            self.grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
            self.grid.connect(snake.body.popleft(), snake.body[0], self.grid.SPACE_COLOR)
            reward = -1
        # Check for reward
        elif self.grid.food_space(snake.head):
            self.grid.draw(snake.body[0], self.grid.BODY_COLOR) # Redraw tail
            self.grid.connect(snake.body[0], snake.body[1], self.grid.BODY_COLOR)
            self.grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
            reward = 1
            self.grid.new_food()
        else:
            reward = 0
            empty_coord = snake.body.popleft()
            self.grid.connect(empty_coord, snake.body[0], self.grid.SPACE_COLOR)
            self.grid.draw(snake.head, snake.head_color)

        self.grid.connect(snake.body[-1], snake.head, self.grid.BODY_COLOR)

        return reward

    def kill_snake(self, snake_idx):
        """
        Deletes snake from game and subtracts from the snake_count 
        """
        
        assert self.dead_snakes[snake_idx] is not None
        self.grid.erase(self.dead_snakes[snake_idx].head)
        self.grid.erase_snake_body(self.dead_snakes[snake_idx])
        self.erased_snakes[snake_idx] = self.dead_snakes[snake_idx]
        self.dead_snakes[snake_idx] = None
        self.snakes_remaining -= 1

    def step(self, actions):
        """
        Takes an action for each snake in the specified direction and collects their rewards
        and dones.

        directions - tuple, list, or ndarray of directions corresponding to each snake.
        """
        if len(actions) != len(self.snakes):
            print(len(actions))
            print(len(self.snakes))
        assert(len(actions) == len(self.snakes))
        # Ensure no more play until reset
        # if self.snakes_remaining < 1 or self.grid.open_space < 1:
        #     if len(actions) is 1:
        #         return self.grid.grid.copy(), 0, True, {"snakes_remaining":self.snakes_remaining}
        #     else:
        #         return self.grid.grid.copy(), [0]*len(actions), True, {"snakes_remaining":self.snakes_remaining}

        rewards = []
        #
        try:
            actions = np.asarray(actions).tolist()
            len(actions)
            assert (len(actions) == len(self.snakes))
        except TypeError:
            actions = [actions]
            assert (1 == self.snakes_remaining)

        obs = []
        dones = []

        for i, snakes in enumerate(self.snakes):
            if self.snakes[i] is not None:
                direction = actions[i]
                direction = self.local_actions[i].transform(direction - 1)

                self.move_snake(direction,i)
                obs.append(self.local_views[i].get(self.grid, self.snakes[i].head, direction))
                rewards.append(self.move_result(direction, i))
                #print("direction: " + str(direction))
                #print(rewards)
                if self.snakes[i] is not None:
                    #print("ALIVE")

                    dones.append(False)

                if self.snakes[i] is None:
                    #print("DEAD")
                    if self.dead_snakes[i] is not None:
                        self.kill_snake(i)

                    # coord_not_found = True
                    # while coord_not_found:
                    #     coord = (np.random.randint(0, self.grid.grid_size[0]), np.random.randint(0, self.grid.grid_size[1]))
                    #     if np.array_equal(self.grid.color_of(coord), self.grid.SPACE_COLOR):
                    #         coord_not_found = False
                    #
                    # self.snakes[i] = Snake(coord, 2)
                    # color = self.grid.HEAD_COLOR
                    # self.snakes[i].head_color = color
                    # self.grid.draw_snake(self.snakes[i], color)
                    # obs.append(self.local_views[i].get(self.grid, self.snakes[i].head, direction))
                    # self.local_actions[i].reset()
                    dones.append(True)
                    # plt.imshow(np.squeeze(obs[i]), interpolation='none')
                    # plt.show()
            else:
                obs.append(self.local_views[i].get_zero(self.grid))
                dones.append(True)
                rewards.append(0)


        # if self.done:
        #     rewards = [0]
        #     dones = [True]
        #
        # if dones[0]:
        #     self.done = True

        #print(dones)
        #print("*******************")

        #done = self.snakes_remaining <= 1 or self.grid.open_space < 1


        assert(len(obs) == len(self.snakes))
        assert(len(dones) == len(self.snakes))

        #print("CTRL: " + str(obs))

        return obs, rewards, np.asarray(dones), {"snakes_remaining":self.snakes_remaining}
