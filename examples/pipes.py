from markov import *

axises = ["X", "Y"]

dim = 64

wfc = Wfc((dim, dim, 1))

tiles = {}
tag_dir_to_tiles = {}

blocklist = set()


def flip(d):
    if d == "x":
        return "negx"
    if d == "y":
        return "negy"
    if d == "negx":
        return "x"
    if d == "negy":
        return "y"


def rot(d):
    if d == "x":
        return "y"
    if d == "y":
        return "negx"
    if d == "negx":
        return "negy"
    if d == "negy":
        return "x"


def rot_m(m):
    n = {}
    for d, v in m.items():
        n[rot(d)] = v
    return n


def add(prob, tags):
    tile = wfc.add(prob)

    tiles[tile] = tags

    for dir, tag in tags.items():
        pair = (flip(dir), tag)
        if not pair in tag_dir_to_tiles:
            tag_dir_to_tiles[pair] = []
        tag_dir_to_tiles[pair].append(tile)
    return tile


def add_mul(prob, rots, tags):
    res = []
    prob /= rots
    for i in range(rots):
        res.append(add(prob, tags))
        tags = rot_m(tags)
    return res


def connect_all():
    for frm, tags in tiles.items():
        for dir, tag in tags.items():
            for to in tag_dir_to_tiles[(dir, tag)]:
                if (frm, to) in blocklist or (to, frm) in blocklist:
                    # print("skipping")
                    continue
                # print(f"connecting {frm} to {to} along {dir}")
                wfc.connect(frm, to, [dir])


empty = add(0.0, {"x": "no", "negx": "no", "y": "no", "negy": "no"})

straight_h, straight_v = add_mul(
    1.0, 2, {"x": "line", "negx": "line", "y": "no", "negy": "no"}
)

edge_dr, edge_dl, edge_ul, edge_ur = add_mul(
    1.0, 4, {"y": "line", "negy": "no", "x": "line", "negx": "no"}
)


end_r, end_d, end_l, end_u = add_mul(
    0.01, 4, {"y": "no", "negy": "no", "x": "line", "negx": "no"}
)

cross = add(2.5, {"y": "line", "negy": "line", "x": "line", "negx": "line"})

blocklist.add((cross, cross))

print(blocklist)
print(tag_dir_to_tiles)

connect_all()


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
