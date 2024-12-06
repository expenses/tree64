from markov import *

dim = 64

wfc = Wfc((dim, dim, 1))


class Tileset:
    def __init__(self, wfc):
        self.wfc = wfc
        self.tiles = {}
        self.tag_dir_to_tiles = {}
        self.blocklist = set()

    def add(self, prob, tags):
        tile = self.wfc.add(prob)

        for dir, tag in tags.items():
            if type(tag) is not list:
                tags[dir] = [tag]

        self.tiles[tile] = tags

        for dir, dir_tags in tags.items():
            for tag in dir_tags:
                pair = (flip(dir), tag)
                if not pair in self.tag_dir_to_tiles:
                    self.tag_dir_to_tiles[pair] = []
                self.tag_dir_to_tiles[pair].append(tile)
        return tile

    def add_mul(self, prob, rots, tags):
        res = []
        prob /= rots
        for i in range(rots):
            res.append(self.add(prob, tags))
            tags = rot_z(tags)
        return res

    def connect_all(self):
        for frm, tags in self.tiles.items():
            for dir, dir_tags in tags.items():
                for tag in dir_tags:
                    if not (dir, tag) in self.tag_dir_to_tiles:
                        continue

                    for to in self.tag_dir_to_tiles[(dir, tag)]:
                        if (frm, to) in self.blocklist or (to, frm) in self.blocklist:
                            # print("skipping")
                            continue
                        # print(f"connecting {frm} to {to} along {dir}")
                        self.wfc.connect(frm, to, [dir])


tileset = Tileset(wfc)

empty = tileset.add(0.0, {"x": "no", "negx": "no", "y": "no", "negy": "no"})

# straight = tileset.add({"x":("right", "left")})

straight_h, straight_v = tileset.add_mul(
    1.0, 2, {"x": ("line", "right"), "negx": "line", "y": "no", "negy": "no"}
)

edge_dr, edge_dl, edge_ul, edge_ur = tileset.add_mul(
    1.0, 4, {"y": "line", "negy": "no", "x": "line", "negx": "no"}
)


end_r, end_d, end_l, end_u = tileset.add_mul(
    0.01, 4, {"y": "no", "negy": "no", "x": "line", "negx": "no"}
)

cross = tileset.add(2.5, {"y": "line", "negy": "line", "x": "line", "negx": "line"})

tileset.blocklist.add((cross, cross))

tileset.connect_all()
"""

tiles = {}
for i in range(cross + 1):
    tiles[i] = np.zeros((3, 3), dtype=np.uint8)

tiles[straight_h][1] = 1
tiles[straight_v][:, 1] = 1

tiles[cross][1] = 1
tiles[cross][:, 1] = 1

tiles[edge_ur][1, 1:] = 1
tiles[edge_ur][:2, 1] = 1

tiles[edge_ul][1, :2] = 1
tiles[edge_ul][:1, 1] = 1

tiles[edge_dr][1, 1:] = 1
tiles[edge_dr][1:, 1] = 1

tiles[edge_dl][1, :1] = 1
tiles[edge_dl][1:, 1] = 1

arr = np.zeros((dim * 3, dim * 3), dtype=np.uint8)


def draw():
    values = wfc.values()[0]
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            val = values[y, x]
            arr[y * 3 : y * 3 + 3, x * 3 : x * 3 + 3] = tiles[val]
    return arr


wfc.setup_state()

writer = FfmpegWriter("out.avi", (dim * 3, dim * 3))

i = 0
while True:
    value = wfc.find_lowest_entropy()
    if value is None:
        break
    index, tile = value
    wfc.collapse(index, tile)
    if i % 3 == 0:
        writer.write(draw())
    i += 1
print(i)
# wfc.collapse_all()
assert wfc.all_collapsed()
"""
