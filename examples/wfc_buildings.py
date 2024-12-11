from markov import *

dim = 64

tileset = TaggingTileset()

empty = tileset.add(0.5, {"x": "e", "negx": "e", "y": "e", "negy": "e"})
floor = tileset.add(1.0, {"x": "f", "negx": "f", "y": "f", "negy": "f"})
edge = tileset.add_mul(
    1.0 / 3.0,
    4,
    {
        "x": Tags(incoming="right", outgoing="left"),
        "negx": Tags(incoming="left", outgoing="right"),
        "y": Tags(outgoing="e"),
        "negy": Tags(outgoing="f"),
    },
)
corner = tileset.add_mul(
    25.0,
    4,
    {
        "x": Tags(outgoing="left"),
        "negy": Tags(outgoing="right"),
        "negx": Tags(outgoing="e"),
        "y": Tags(outgoing="e"),
    },
)
inner = tileset.add_mul(
    0.000025,
    4,
    {
        "negx": Tags(outgoing="right"),
        "y": Tags(outgoing="left"),
        "x": Tags(outgoing="f"),
        "negy": Tags(outgoing="f"),
    },
)

wfc = tileset.tileset.create_wfc((dim, dim, 1))

for i in range(dim):
    wfc.collapse(i, empty)
    wfc.collapse(i * dim, empty)
    wfc.collapse(i * dim + (dim - 1), empty)
    wfc.collapse(i + (dim * (dim - 1)), empty)


writer = FfmpegWriter("out.avi", (dim, dim))

any_contradictions = collapse_all_with_callback(
    wfc, lambda: writer.write(wfc.values()[0]), skip=3
)

assert not any_contradictions
palette = Palette(PICO8_PALETTE.srgb + [(128, 128, 128)] * 100)
save_image("out.png", wfc.values()[0], palette=palette)
