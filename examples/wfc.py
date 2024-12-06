import numpy as np
import random
from markov import *

axii = [1, 2, -1, -2, 3, -3]


class Tile:
    def __init__(self, data, prob):
        self.data = data
        self.prob = prob
        self.connections = dict(((axis, set()) for axis in axii))

    def connect(self, other, axis):
        self.connections[axis].add(other)

    def finalize(self, num_tiles):
        for axis in axii:
            values = np.zeros(num_tiles, dtype=bool)
            values[list(self.connections[axis])] = True
            self.connections[axis] = values

    def __repr__(self):
        return f"{self.data}, {self.connections}"


def sign(x):
    return x // abs(x)


class Wfc:
    def __init__(self):
        self.tiles = []
        self.arr = None

    def add(self, data, prob=1.0):
        index = len(self.tiles)
        self.tiles.append(Tile(data, prob))
        return index

    def connect(self, frm, to, axii):
        for axis in axii:
            self.tiles[frm].connect(to, axis)
            self.tiles[to].connect(frm, -axis)

    def setup(self, size):
        self.shape = np.array(size)
        for tile in self.tiles:
            tile.finalize(len(self.tiles))
        self.arr = np.ones((len(self.tiles),) + size, dtype=bool)

    def collapse(self, coord, value):
        values = np.array([i for i in range(len(self.tiles)) if i != value])
        self.set(coord, values)

    def find_lowest_entropy(self):
        counts = np.sum(self.arr, axis=0)
        above_one = counts > 1
        if not above_one.any():
            return None
        min_val = counts[above_one].min()
        coords = np.argwhere(counts == min_val)
        coord = random.choice(coords)
        tiles = self.arr[:, *coord].nonzero()[0]
        tile = random.choice(tiles)
        return coord, tile

    def collapse_status(self):
        return self.arr.sum(0)

    def result(self):
        result = np.zeros(self.arr.shape, dtype=np.uint8)
        for i in range(len(self.tiles)):
            result[i] = i
        result[self.arr == False] = 0
        return result.sum(0).astype(np.uint8)

    def collapse_all(self):
        while True:
            values = self.find_lowest_entropy()
            if not values:
                return
            coords, tile = values
            self.collapse(coords, tile)

    def set(self, coord, negate_values):
        if ((coord < 0) | (self.shape <= coord)).any():
            return
        updated = negate_values[self.arr[:, *coord][negate_values]]

        if len(updated) == 0:
            return

        self.arr[updated, *coord] = False

        remaining = self.arr[:, *coord].nonzero()[0]

        if len(remaining) == 0:
            print(coord)
            return

        for axis in axii:
            invalid = np.ones(len(self.tiles), dtype=bool)

            for i in remaining:
                invalid &= self.tiles[i].connections[axis] == False

            negate = invalid.nonzero()[0]

            delta = [0] * len(self.shape)
            delta[abs(axis) - 1] = sign(axis)
            self.set(coord + delta, negate)


wfc = Wfc()
"""
forest = wfc.add("forest")
grass = wfc.add("grass")
beach = wfc.add("beach")
sea = wfc.add("sea")
wfc.connect(beach, sea, axii)
wfc.connect(beach, grass, axii)
wfc.connect(sea, sea, axii)
wfc.connect(grass, grass, axii)
wfc.connect(forest, grass, axii)
wfc.connect(forest, forest, axii)
wfc.setup((10, 10, 10))

"""
empty = wfc.add("empty")
solid = wfc.add("solid")
wfc.connect(empty, empty, axii)
wfc.connect(solid, solid, axii)
wfc.connect(solid, empty, [3, 2, -3, -2, 1])
wfc.setup((20, 20, 20))

"""
empty=wfc.add("empty")
base = wfc.add("base")
wfc.connect(base,base,[3,-3,2,-2])

def add_stairs(suffix, axis):
    lower = wfc.add("stairs_lower_"+suffix)
    upper= wfc.add("stairs_upper_"+suffix)
    wfc.connect(lower,upper,[1])
    wfc.connect(lower,base,[-axis])
    wfc.connect(upper,base,[axis])
    wfc.connect(lower,upper,[-axis])
    wfc.connect(upper,lower,[axis])
add_stairs("1",3)
add_stairs("-1",-3)
add_stairs("2",2)
add_stairs("-2",-2)
for tile in range(len(wfc.tiles)):
    wfc.connect(empty,tile,axii)
wfc.setup((10,10,10))
print(wfc.tiles)
"""
wfc.collapse_all()

status = wfc.collapse_status()

if (status != 1).any():
    print("!!")
    print(status)


print(wfc.result())

write_usd("res.usdc", wfc.result())
