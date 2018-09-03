import numpy as np
import matplotlib.pyplot as plt


class HeatMap:
    def __init__(self, gridsize, num_maps=1):
        self.hmps = []
        for _ in range(0, num_maps):
            self.hmps.append(np.ones(gridsize, dtype=np.double))

    def visit(self, cords, id):
        print(self.hmps[id][cords[0]][cords[1]])
        self.hmps[id][cords[0]][cords[1]] += 1

    def plot(self, id):
        #total = max(self.hmps[id])
        plt.imshow(self.hmps[id], cmap='hot', interpolation='nearest')
        plt.draw()

        #
        # for heat_map in self.hmps:
        #     plt.imshow(heat_map/total, cmap='hot_r', interpolation='nearest')
        #     plt.pause(0.07)
        #     plt.draw()
