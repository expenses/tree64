from markov import *
from markov.interpreter import parse_pattern
from markov.wfc import (
    TaggingTileset,
    Tags,
    wave_from_tiles,
    collapse_all_with_callback,
    bitwise_not,
)
from pprint import pprint

dims = (40, 40, 40)

tileset = TaggingTileset()

empty = tileset.add(0.1, "empty", symmetry="X_3d")
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
# tiles[empty] = load_mkjr_vox("MarkovJunior/resources/tilesets/Paths/Empty.vox")

for i, slot in enumerate(line):
    tiles[slot] = np.rot90(line_vox, axes=(1, 2), k=i)

for i, slot in enumerate(turn):
    tiles[slot] = np.rot90(corner_vox, axes=(1, 2), k=i)

for i, slot in enumerate(down):
    tiles[slot] = np.rot90(down_vox, axes=(1, 2), k=i)

for i, slot in enumerate(up):
    tiles[slot] = np.rot90(up_vox, axes=(1, 2), k=i + 2)

output = np.zeros((dims[0] * 5, dims[1] * 5, dims[2] * 5), dtype=np.uint8)

initial_state = np.zeros(dims, dtype=np.uint64)
initial_state[0] = bitwise_not(wave_from_tiles(up))
initial_state[-1] = wave_from_tiles([empty])
initial_state[:, 0] = wave_from_tiles([empty])
initial_state[:, -1] = wave_from_tiles([empty])
initial_state[:, :, 0] = wave_from_tiles([empty])
initial_state[:, :, -1] = wave_from_tiles([empty])

wfc = tileset.tileset.create_wfc_with_initial_state(initial_state, entropy="linear")
print(wfc.collapse_all())

output = map_3d(wfc.values(), output, tiles)
output[0, :, :] = index_for_colour("N")
output[1, :, :] = index_for_colour("E")

writer = UsdWriter("pipes.usdc")

# writer.write(output)


def vertical(pattern):
    return Pattern(pattern, shuffles=Y_IS_Z, flips=Y_IS_Z_TOGGLE_X)


rep(output, Prl(Pattern("Y=C", chance=0.25), settings=ONCE))
rep(output, "Y=B")
# Extend down
rep(output, Pattern("DB=*F", shuffles=[[2, 1, 0]], flips=[[True, False, False]]))
rep(output, Pattern("FB=FF", shuffles=[[2, 1, 0]], flips=[[True, False, False]]))
# writer.write(output)
rep(
    output,
    All(
        vertical("FF,EE=BB,EE"),
        vertical("FF,FD,EE=FF,BD,EE"),
        vertical("F,B=B,B"),
        vertical("B,F=B,B"),
        vertical("F,P=B,P"),
        vertical("P*,*F=**,*B"),
        # "FF,EE=BB,EE",
        # "PF=*B",
        # Pattern('FB=BB', shuffles=[[2,1,0]], flips=TOGGLE_X)
    ),
)
# writer.write(output)
rep(
    output,
    All(
        Pattern(
            (
                parse_pattern("*A*/ADB/*B* ***/*B*/***"),
                parse_pattern("***/***/*** ***/*F*/***"),
            ),
            shuffles=ROT_AROUND_Z,
            flips=[[False, False, True], [True, False, True]],
        ),
        vertical("F,B=F,F"),
    ),
)
# writer.write(output)
rep(
    output,
    All(vertical("FBBBF,*AAA*=*RRR*,*****"), vertical("FBBBF,*RRR*=*RRR*,*****")),
)
# writer.write(output)
rep(output, All(vertical("RFFR,RFDA=U**U,U***"), vertical("RFDA,RFFR=U***,U**U")))
rep(output, All(vertical("R,U=U,*"), vertical("U,R=*,U")))
# writer.write(output)
rep(output, All("RFFR,BBBB=*RR*,****"))
# writer.write(output)
rep(output, All("U=R"))
# writer.write(output)
rep(output, All("RR,DA,FR=UU,**,*U"))
rep(output, All(vertical("RU=U*"), vertical("UR=*U")))
rep(
    output,
    All(
        vertical("RB,AB,RB=RB,RB,RB"),
        vertical("RB,DB,RB=RB,RB,RB"),
    ),
)
rep(output, Prl("F=A", "D=A", "P=A", "U=R"))
# writer.write(output)
rep(
    output,
    Prl(
        Pattern(
            (
                parse_pattern("AAAAA ARRRA ARRRA ARRRA ARRRA AAAAA"),
                parse_pattern("AAAAA AAAAA AAAAA AAAAA AAAAA AAAAA"),
            ),
            shuffles=ROT_AROUND_Z,
            flips=[[False, False, True], [True, False, True]],
        ),
    ),
)
writer.write(output)
