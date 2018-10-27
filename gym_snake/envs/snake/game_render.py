from pygame.locals import *
import pygame
import time
import numpy as np

class GameRender:
    window_width = 1000
    window_height = 1000

    def __init__(self, grid_size):

        self.grid_size = grid_size
        self._block_size_width = self.window_width / grid_size[0]
        self._block_size_height = self.window_height / grid_size[1]

        self._running = True
        self._display_surf = None
        self._snake_surf = None
        self._apple_surf = None

        self.on_init()

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.window_width, self.window_height), pygame.HWSURFACE)
        pygame.display.set_caption('Gym-Snake')
        self._running = True

        self._snake_head_surf = pygame.Surface([self._block_size_width, self._block_size_height])
        self._snake_head_surf.fill((0, 0, 255))

        self._snake__body_surf = pygame.Surface([self._block_size_width, self._block_size_height])
        self._snake__body_surf.fill((255, 255, 0))

        self._apple_surf = pygame.Surface([self._block_size_width, self._block_size_height])
        self._apple_surf.fill((0, 255, 0))

        self._space_surf = pygame.Surface([self._block_size_width, self._block_size_height])
        self._space_surf.fill((255, 255, 255))

    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

    def render(self, grid):
        self._display_surf.fill((0, 0, 0))

        grid_shape = grid.shape
        M = grid_shape[0]
        N = grid_shape[1]

        for i in range(M):
            for j in range(N):
                if grid[i][j] != 0:
                    if grid[i][j] == 1:
                        self._display_surf.blit(self._space_surf, (i * self._block_size_width, j * self._block_size_height))
                    elif grid[i][j] == -1:
                        self._display_surf.blit(self._snake__body_surf, (i * self._block_size_width, j * self._block_size_height))
                    elif grid[i][j] == -2:
                        self._display_surf.blit(self._snake_head_surf, (i * self._block_size_width, j * self._block_size_height))
                    elif grid[i][j] == 2:
                        self._display_surf.blit(self._apple_surf, (i * self._block_size_width, j * self._block_size_height))
        pygame.display.flip()

    def cleanup(self):
        pygame.quit()

if __name__ == "__main__":

    game = GameRender(grid_size=(10,10))
    grid = np.zeros((10, 10))
    for i in range(10):
        grid[i][i] = 1
    grid[4][7] = 10
    print(grid.shape)
    game.render(grid)
    for i in range(5):
        time.sleep(1)
    game.on_cleanup()
