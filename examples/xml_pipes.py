import sys
from markov import *

dim = 128

t, tileset = read_xml(sys.argv[1])

wfc = tileset.tileset.create_wfc((dim, dim, 1))

tiles = np.zeros((wfc.num_tiles(), 3, 3), dtype=np.uint8)

tiles[t["line"][0]][1] = 1
tiles[t["line"][1]][:, 1] = 1

tiles[t["cross"][0]][1] = 1
tiles[t["cross"][0]][:, 1] = 1

tiles[t["cross"][1]][1] = 1
tiles[t["cross"][1]][:, 1] = 1

tiles[t["corner"][0]][1, 1:] = 1
tiles[t["corner"][0]][1:, 1] = 1

tiles[t["corner"][1]][1, :2] = 1
tiles[t["corner"][1]][1:, 1] = 1

tiles[t["corner"][2]][1, :2] = 1
tiles[t["corner"][2]][:2, 1] = 1

tiles[t["corner"][3]][1, 1:] = 1
tiles[t["corner"][3]][:2, 1] = 1


tiles[t["t"][0]][1] = 1
tiles[t["t"][0]][:2, 1] = 1

tiles[t["t"][2]][1] = 1
tiles[t["t"][2]][1:, 1] = 1

tiles[t["t"][3]][:, 1] = 1
tiles[t["t"][3]][1, :2] = 1

tiles[t["t"][1]][:, 1] = 1
tiles[t["t"][1]][1, 1:] = 1

writer = FfmpegWriter("out.avi", (dim * 3, dim * 3))
arr = np.zeros((dim * 3, dim * 3), dtype=np.uint8)

collapse_all_with_callback(
    wfc, lambda: writer.write(map_2d(wfc.values()[0], arr, tiles)), skip=8
)

assert wfc.all_collapsed()
