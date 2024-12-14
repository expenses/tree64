from markov import *
import sys
from pprint import pprint

d = sys.argv[1]

dim = 9

tileset = TaggingTileset()

empty = tileset.add(2.0, "empty", symmetry="X_3d")
i = tileset.add_mul(
    1.0,
    2,
    {
        "x": "i",
        "y": Tags(outgoing="empty"),
        "z": Tags("i", outgoing="empty"),
        "negz": Tags("i", outgoing="empty"),
    },
    symmetry="I",
)
t = tileset.add_mul(
    1.0,
    4,
    {
        "x": Tags("t_x", outgoing="i"),
        "negx": Tags("t_neg_x", outgoing="i"),
        "y": Tags(outgoing="i"),
        "negy": "empty",
        "z": Tags("empty", "t"),
        "negz": Tags("empty", "t"),
    },
)

wfc = tileset.tileset.create_wfc((dim, dim, dim))

tiles = np.zeros((wfc.num_tiles(), 3, 6, 6), dtype=np.uint8)

line_vox = load_vox(f"{d}/I.vox")

t_vox = load_vox(f"{d}/T.vox")
empty_vox = load_vox(f"{d}/Nothing.vox")


# tiles[line_z] = np.rot90(line_vox)
tiles[i[0], :3, :3, :3] = np.rot90(line_vox, axes=(1, 2))
tiles[i[1], :3, :3, :3] = line_vox
tiles[empty, :3, :3, :3] = empty_vox

for i, t in enumerate(t):
    tiles[t, :3, :3, :3] = np.rot90(np.rot90(t_vox, axes=(0, 2)), axes=(1, 2), k=-i - 1)

"""
for i, t in enumerate(t_vertical):
    tiles[t] = np.rot90(np.rot90(t_vox), axes=(1,2), k=2-i)


tiles[t_up[1]] = t_vox
tiles[t_up[0]] = np.rot90(t_vox, axes=(1,2), k=-1)

tiles[t_down[1]] = np.rot90(t_vox, k=2)
tiles[t_down[0]] = np.rot90(np.rot90(t_vox, k=2), axes=(1,2), k=-1)
"""

print(wfc.collapse_all())

print(t_vox)


output = np.zeros((dim * 3, dim * 6, dim * 6), dtype=np.uint8)

writer = UsdWriter("house.usdc")

arr = map_3d(wfc.values(), output, tiles)

unpadded = np.zeros((dim * 3, dim * 6 - 3, dim * 6 - 3), dtype=np.uint8)
unpadded[:] = arr[:, :-3, :-3]
arr = unpadded

writer.write(arr)

rep(arr, "LBBBL=*LSL*")

writer.write(arr)

house_flips = [[False, False, False], [True, False, False]]
house_shuffles = [[0, 1, 2], [1, 0, 2]]


def house_pat(pattern):
    return Pattern(pattern, flips=house_flips, shuffles=house_shuffles)


def house_rep(arr, pattern, flips=None):
    if type(pattern) is str:
        pattern = Pattern(pattern, flips=flips or house_flips, shuffles=house_shuffles)
    rep(arr, pattern)
    writer.write(arr)


house_rep(
    arr,
    """
    V***,
    ****,
    ****,
    ***B=
    ****,
    ****,
    ****,
    ***O
""",
)

house_rep(
    arr,
    """
    *****,
    *****,
    **V**,
    *****,
    *****=
    *****,
    *****,
    **v**,
    *****,
    *****
    """,
)
house_rep(arr, ("V*****v", "******n"))
house_rep(arr, "nBBBBBn=*aaaaa*")
# spawn orange things
house_rep(arr, ("a**O**S", "***o***"))
house_rep(arr, ("V***,****,****,***O", "****,****,****,***o"))
# Extend edges
house_rep(arr, ("anBBBBBV*", "**aaaaa*a"))
# Recolour
house_rep(arr, "a=L")
house_rep(arr, "n=V")
house_rep(arr, "v=V")
house_rep(arr, "VLLLLLV=***S***")
house_rep(arr, "VBBBBBV=***S***")

new = np.zeros((dim, dim * 2 - 1, dim * 2 - 1), dtype=np.uint8)

for z in range(dim):
    for y in range(dim * 2 - 1):
        for x in range(dim * 2 - 1):
            center = arr[z * 3 + 1, y * 3 + 1, x * 3 + 1]
            if center == index_for_colour("O"):
                new[z, y, x] = index_for_colour("t")
            elif center == index_for_colour("o"):
                new[z, y, x] = index_for_colour("o")
            elif center == index_for_colour("V"):
                new[z, y, x] = index_for_colour("V")
            else:
                middle_layer = arr[z * 3 + 1, y * 3 : (y + 1) * 3, x * 3 : (x + 1) * 3]
                S_layer = np.zeros((3, 3), dtype=np.uint8)
                S_layer[1, 1] = index_for_colour("S")
                if (middle_layer == S_layer).all():
                    new[z, y, x] = index_for_colour("s")
                else:
                    new[z, y, x] = index_for_colour("S")

arr = new
writer.write(new)

WILDCARD = 255

rep(arr, Pattern("*t=*y", shuffles=[[2, 1, 0]], flips=[[True, False, False]]))
writer.write(arr)
house_rep(arr, "***,*S*,***=***,*C*,***")
house_rep(arr, "***,*s*,***=***,*c*,***")
house_rep(
    arr,
    Markov(
        house_pat("R*y=**r"),
        house_pat("r*y=**r"),
        Pattern(
            ("*******,y*y*y*y,y*y*y*y", "*******,*******,**R****"),
            flips=Y_IS_Z_TOGGLE_X,
            shuffles=Y_IS_Z,
        ),
        Pattern(
            ("*******,r*r*r*r,r*y*r*r", "*******,*******,**R****"),
            flips=Y_IS_Z_TOGGLE_X,
            shuffles=Y_IS_Z,
        ),
    ),
)
writer.write(arr)
