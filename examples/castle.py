from markov import *

dim = 350

wfc = Wfc((dim, dim, 1))

tileset = Tileset(wfc)

riverside = Tags(incoming="riverside", outgoing=["ground", "wallside", "roadside"])
roadside = Tags(incoming="roadside", outgoing=["wallside", "ground", "riverside"])

river_prob = 0.002

ground = tileset.add(1.0, Tags("ground", "forest"), symmetry="X")
tree = tileset.add(1.0, Tags("forest", outgoing="riverside"), symmetry="X")
# hill = tileset.add(0.8, Tags("hill", "mountain", outgoing="ground"), symmetry="X")
# mountain = tileset.add(0.1, Tags("mountain"), symmetry="X")


bridge = tileset.add_mul(
    river_prob * 0.25, 2, {"x": Tags(outgoing="river"), "y": "road"}, symmetry="I"
)
river = tileset.add_mul(river_prob, 2, {"x": "river", "y": riverside}, symmetry="I")

road = tileset.add_mul(
    1.0,
    2,
    {
        "x": "road",
        "y": roadside,
    },
    symmetry="I",
)
riverturn = tileset.add_mul(
    river_prob,
    4,
    {
        "x": Tags(outgoing="river"),
        "negx": riverside,
    },
    symmetry="L",
)
roadturn = tileset.add_mul(
    1.0,
    4,
    {
        "x": Tags(outgoing="road"),
        "negx": roadside,
    },
    symmetry="L",
)
roadfork = tileset.add_mul(
    1.0,
    4,
    {
        "x": Tags(outgoing="road"),
        "y": Tags(outgoing="road"),
        "negy": roadside,
    },
    symmetry="T",
)

wall_left = Tags(incoming="wall_left", outgoing="wall_right")
wall_right = Tags(incoming="wall_right", outgoing="wall_left")
wall_out = Tags(incoming="wallside", outgoing=["ground", "riverside", "roadside"])
wall_inner = Tags(outgoing="courtyard")

wall = tileset.add_mul(
    1.0,
    4,
    {
        "x": wall_right,
        "negx": wall_left,
        "y": wall_out,
        "negy": wall_inner,
    },
)
courtyard = tileset.add(
    0.15,
    "courtyard",
    symmetry="X",
)
wallturn = tileset.add_mul(
    1.0,
    4,
    {
        "x": Tags(outgoing="wall_right"),
        "y": Tags(outgoing="wall_left"),
        "negx": wall_out,
        "negy": wall_out,
    },
)
wallturn_inner = tileset.add_mul(
    1.0,
    4,
    {
        "y": Tags(outgoing="wall_right"),
        "x": Tags(outgoing="wall_left"),
        "negx": wall_inner,
        "negy": wall_inner,
    },
)
park = tileset.add_mul(
    0.0,
    4,
    {
        "x": Tags(
            incoming="roadside", outgoing=["wallside", "ground", "riverside", "tree"]
        ),
        "y": "road",
        "negy": roadside,
    },
    symmetry="T",
)

wfc.setup_state()


for x in range(dim):
    wave = wave_from_tiles(river + riverturn + [ground, tree])
    wfc.partial_collapse(x, wave)
    wfc.partial_collapse(x * dim, wave)
    wfc.partial_collapse(dim * dim - 1 - x, wave)
    wfc.partial_collapse(dim * dim - 1 - (x * dim), wave)


def pretty():
    values = wfc.values()[0]

    out = np.copy(values)

    for tile in river + riverturn:
        out[values == tile] = index_for_colour("U")

    for tile in bridge + [courtyard]:
        out[values == tile] = index_for_colour("D")

    for tile in road + roadturn + roadfork:
        out[values == tile] = index_for_colour("A")

    for tile in wall + wallturn + wallturn_inner:
        out[values == tile] = index_for_colour("P")

    out[values == ground] = index_for_colour("E")

    for tile in park + [tree]:
        out[values == tile] = index_for_colour("G")

    out[value == 64] = 0

    return out


writer = FfmpegWriter("out.avi", (dim, dim))


i = 0
while True:
    value = wfc.find_lowest_entropy()
    if value is None:
        break
    index, tile = value
    wfc.collapse(index, tile)
    if i % 32 == 0:
        writer.write(pretty())
    i += 1

print(wfc.all_collapsed())

save_image("castle.png", pretty())
