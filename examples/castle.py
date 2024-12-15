from markov import *
from markov.wfc import TaggingTileset, Tags, wave_from_tiles, collapse_all_with_callback

dim = 350

tileset = TaggingTileset()

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

values = np.zeros((dim, dim), dtype=np.uint8)
wfc = tileset.tileset.create_wfc((dim, dim, 1))

for x in range(dim):
    wave = wave_from_tiles(river + riverturn + [ground, tree])
    wfc.partial_collapse((x, 0, 0), wave)
    wfc.partial_collapse((0, x, 0), wave)
    wfc.partial_collapse((x, dim - 1, 0), wave)
    wfc.partial_collapse((dim - 1, x, 0), wave)


def pretty():
    wfc.set_values(values)

    return replace_values(
        values,
        [
            (river + riverturn, "U"),
            (bridge + [courtyard], "D"),
            (road + roadturn + roadfork, "A"),
            (wall + wallturn + wallturn_inner, "P"),
            ([ground], "E"),
            (park + [tree], "G"),
            ([64], 0),
        ],
    )


writer = FfmpegWriter("out.avi", (dim, dim))

any_contradictions = collapse_all_with_callback(
    wfc, lambda: writer.write(pretty()), skip=32
)

print(any_contradictions)

save_image("castle.png", pretty())
