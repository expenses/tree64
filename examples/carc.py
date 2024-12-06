from markov import *

dim = 64

wfc = Wfc((dim, dim, 1))

tileset = Tileset(wfc)

empty = tileset.add(1.0, {"x": "e", "negx": "e", "y": "e", "negy": "e"})
floor = tileset.add(4.0, {"x": "f", "negx": "f", "y": "f", "negy": "f"})
edge = tileset.add_mul(1.0,4, {"x": ("right","left"), "negx": ("left","right"), "y": (None,"e"), "negy": (None, "f")})
corner = tileset.add_mul(100.0, 4,{"x": (None,"left"), "negy": (None,"right"), "negx": (None,"e"), "y": (None,"e")})
inner = tileset.add_mul(0.001, 4,{"negx": (None,"right"), "y": (None,"left"), "x": (None,"f"), "negy": (None,"f")})


wfc.setup_state()

for i in range(dim):
    wfc.collapse(i, empty)
    wfc.collapse(i * dim, empty)
    wfc.collapse(i * dim + (dim - 1), empty)
    wfc.collapse(i + (dim * (dim - 1)), empty)


wfc.collapse_all()
assert wfc.all_collapsed()
palette = Palette(PICO8_PALETTE.srgb + [(128, 128, 128)] * 100)
save_image("out.png", wfc.values()[0], palette=palette)
