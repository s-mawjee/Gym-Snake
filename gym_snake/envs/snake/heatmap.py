import numpy as np
import matplotlib.pyplot as plt


class HeatMap:
    def __init__(self, gridsize, num_maps=1):
        plt.ion()
        self.num_maps = num_maps
        self.hmps = []
        for _ in range(0, num_maps):
            self.hmps.append(np.ones(gridsize, dtype=np.double))

        self.fig = plt.figure(figsize=(num_maps, 1))


    def visit(self, cords, id):
        self.hmps[id][cords[0]][cords[1]] += 1

    def plot(self, idx = None):
        if idx is None:
            self.fig.clear()
            for i, map in enumerate(self.hmps):


                self.fig.add_subplot(self.num_maps, 1, i + 1)
                plt.imshow(self.hmps[i], cmap='hot', interpolation='nearest')

        else:
            #total = max(self.hmps[id])
            plt.imshow(self.hmps[idx], cmap='hot', interpolation='nearest')

        #plt.show()
        plt.pause(0.01)
        #plt.draw()
        #
        # for heat_map in self.hmps:
        #     plt.imshow(heat_map/total, cmap='hot_r', interpolation='nearest')
        #     plt.pause(0.07)
        #     plt.draw()
