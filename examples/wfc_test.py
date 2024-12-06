# coding: utf-8
from PIL.Image import Palette
from markov import *

axises = ["X", "Y", "Z"]

wfc = Wfc((20, 20, 20))

empty = wfc.add(0.0)
ground = wfc.add(10.0)
wfc.connect(ground, ground, ["x", "y"])

for forward, back, left_right in [
    ("x", "negx", ["y", "negy"]),
    ("y", "negy", ["x", "negx"]),
    ("negx", "x", ["y", "negy"]),
    ("negy", "y", ["x", "negx"]),
]:
    stairs_top = wfc.add(1.0)
    stairs_bottom = wfc.add(5.0)

    wfc.connect(stairs_top, stairs_bottom, [forward, "negz"])
    wfc.connect(stairs_top, ground, [forward])
    wfc.connect(stairs_bottom, ground, [back])

    wfc.connect(empty, stairs_top, left_right + [back, "negz"])
    wfc.connect(empty, stairs_bottom, left_right + [forward, "z"])


wfc.connect(empty, empty, axises)
wfc.connect(empty, ground, axises)


# wfc.connect_to_all(empty)

"""
air = wfc.add(1.0)
solid = wfc.add(1.0)
wfc.connect(solid,air,["x","y","z","negx","negy"])
wfc.connect(solid,solid,axises)
wfc.connect(air,air,axises)
"""
wfc.setup_state()

palette = Palette(PICO8_PALETTE.srgb + [(128, 128, 128)] * 100)

writer = FfmpegWriter("out.avi", (100, 80))

while True:
    value = wfc.find_lowest_entropy()
    if value is None:
        break
    index, tile = value
    wfc.collapse(index, tile)
    writer.write(wfc.values().reshape(100, 80), palette=palette)

wfc.collapse_all()
# assert wfc.all_collapsed()

values = wfc.values()

arr = np.zeros(np.array(values.shape) * 3, dtype=np.uint8)

for x in range(values.shape[0]):
    for y in range(values.shape[1]):
        for z in range(values.shape[2]):
            value = values[z, y, x]

            v = np.zeros((3, 3, 3), dtype=np.uint8)

            if value == empty:
                pass
            else:  # value == ground:
                v[:] = 1
            """
            elif value % 2 == 0:
                pass
            elif value == 5:
                v[2,2] = 1
                v[1,1:] = 1
                v[0] = 1
            elif value == 3:
                v[2,:,2] = 1
                v[1,:,1:] = 1
                v[0] = 1
            """

            arr[z * 3 : z * 3 + 3, y * 3 : y * 3 + 3, x * 3 : x * 3 + 3] = v

write_usd("xxx.usdc", values, palette=palette)
