
from markov import *
from markov.wfc import TaggingTileset, Tags, wave_from_tiles, collapse_all_with_callback
from pprint import pprint

dim = 16

tileset = TaggingTileset()

empty = tileset.add(2.0, "empty", symmetry="X_3d")
line = tileset.add_mul(
    1.0, 2, {"x": "line", "y": Tags(outgoing="empty"), "z": "empty"}, symmetry="I_3d"
)
turn = tileset.add_mul(
    1.0, 4, {"x": "line", "negx": "empty", "z": "empty"}, symmetry="L_3d"
)
x = tileset.add(
    0.1, {"x": Tags(incoming="cross", outgoing="line"), "z": "empty"}, symmetry="X_3d"
)

down = tileset.add_mul(
    4.0,
    4,
    {
        "x": "empty",
        "negx": Tags(incoming="down", outgoing=["line", "up", "cross"]),
        "y": "empty",
        "negy": "empty",
        "z": lambda i: f"stairs_{i}",
        "negz": "empty",
    },
)
up = tileset.add_mul(
    4.0,
    4,
    {
        "x": Tags(incoming="up", outgoing=["line", "down", "cross"]),
        "negx": "empty",
        "y": "empty",
        "negy": "empty",
        "negz": lambda i: f"stairs_{i}",
        "z": "empty",
    },
)

tiles = np.zeros((tileset.tileset.num_tiles(), 5, 5, 5), dtype=np.uint8)
line_vox = load_mkjr_vox("MarkovJunior/resources/tilesets/Paths/Line.vox")
corner_vox = load_mkjr_vox("MarkovJunior/resources/tilesets/Paths/Turn.vox")
down_vox = load_mkjr_vox("MarkovJunior/resources/tilesets/Paths/Down.vox")
up_vox = load_mkjr_vox("MarkovJunior/resources/tilesets/Paths/Up.vox")


tiles[x] = load_mkjr_vox("MarkovJunior/resources/tilesets/Paths/X.vox")
#tiles[empty] = load_mkjr_vox("MarkovJunior/resources/tilesets/Paths/Empty.vox")

for i, slot in enumerate(line):
    tiles[slot] = np.rot90(line_vox, axes=(1, 2), k=i)

for i, slot in enumerate(turn):
    tiles[slot] = np.rot90(corner_vox, axes=(1, 2), k=i)

for i, slot in enumerate(down):
    tiles[slot] = np.rot90(down_vox, axes=(1, 2), k=i)

for i, slot in enumerate(up):
    tiles[slot] = np.rot90(up_vox, axes=(1, 2), k=i + 2)

output = np.zeros((dim * 5, dim * 5, dim * 5), dtype=np.uint8)


iters = 0
while True:
    iters += 1
    wfc = tileset.tileset.create_wfc((dim, dim, dim))
    for j in range(dim):
        for i in range(dim):
            #wfc.collapse((j, i, 0), empty)
            wfc.collapse((i, j, dim - 1), empty)
            wfc.collapse((i, 0, j), empty)
            wfc.collapse((i, dim - 1, j), empty)
            wfc.collapse((0, i, j), empty)
            wfc.collapse((dim - 1, i, j), empty)

            wfc.partial_collapse(
                (i, j, 0), wave_from_tiles([empty, x] + line + down + turn)
            )
    found_contradiction = wfc.collapse_all()
    if not found_contradiction:
        print(iters)
        break


output = map_3d(wfc.values(), output, tiles)
output[0, :, :] = index_for_colour("N")
output[1, :, :] = index_for_colour("E")
write_usd("pipes.usdc", output)
'''

rep(output, Prl(Pattern("Y=C", chance=0.25), settings=ONCE))
rep(output, "Y=B")
print(line_vox, index_for_colour("D"))
print(list(PICO8_PALETTE.srgb[index_for_colour("D")]))
#print(list(PICO8_PALETTE.srgb[80]))

print(chr(PALETTE_CHARS_TO_INDEX.index(40)))

rep(output, Pattern('DB=*F', shuffles=[[2,1,0]], flips=[[True, False, False]]))
rep(output, Pattern('FB=FF', shuffles=[[2,1,0]], flips=[[True, False, False]]))
#writer.write(output)

def chr_for_val(val):
    return chr(PALETTE_CHARS_TO_INDEX.index(val))

rep(output, One(
    "FF,EE=BB,EE",
    "pF=*B",
    Pattern('FB=BB', shuffles=[[2,1,0]], flips=TOGGLE_X)
))

print(up_vox, chr_for_val(18))

#rep(output, Pattern('FF,EE=BB,EE'))
#writer.write(output)
#rep(output, Pattern('FF,FD,EE=FF,BD,EE'))
#writer.write(output)
#rep(output, Pattern('FB=BB', shuffles=[[2,1,0]], flips=[[True, False, False]]))
#writer.write(output)
#rep(output, Pattern('BF=BB', shuffles=[[2,1,0]], flips=[[True, False, False]]))
writer.write(output)


rep(output, Pattern((
    np.array([0, 42], dtype=np.uint8).reshape((2, 1, 1)),
    np.array([index_for_colour("F"), 42], dtype=np.uint8).reshape((2, 1, 1))
), shuffles=NO_SHUFFLES, flips = NO_FLIPS))

rep(output, Pattern((
    np.array([0, index_for_colour("F")], dtype=np.uint8).reshape((2, 1, 1)),
    np.array([index_for_colour("F"), index_for_colour("F")], dtype=np.uint8).reshape((2, 1, 1))
), shuffles=NO_SHUFFLES, flips = NO_FLIPS))
'''

#write_usd("pipes.usdc", output)
