import numpy as np
import matplotlib.pyplot as plt


class HeatMap:
    def __init__(self, gridsize, num_maps=1):
        self.hmps = []
        for _ in range(0, num_maps):
            self.hmps.append(np.zeros(gridsize, dtype=np.double ))

    def visit(self, cords, id):
        self.hmps[id][cords] += 1

    def plot(self):
        total = sum([np.sum(e) for e in self.hmps])


        for heat_map in self.hmps:
            plt.imshow(heat_map/total, cmap='hot_r', interpolation='nearest')
            plt.show()
