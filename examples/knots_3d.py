from markov import *

dim = 8

tileset = TaggingTileset()

empty = tileset.add(0.0, "empty", symmetry="X_3d")

line_tags = apply_symmetry({"x": "line", "y": "empty", "z": "empty"}, "I_3d")

line_x = tileset.add(1.0 / 3.0, line_tags)
line_y = tileset.add(1.0 / 3.0, rot_z(line_tags))
line_z = tileset.add(1.0 / 3.0, rot_y(line_tags))

l_tags = apply_symmetry({"x": "line", "negy": "empty", "z": "empty"}, "L_3d")

l_x = tileset.add_mul(1.0 / 3.0, 4, l_tags)
l_up = tileset.add_mul(1.0 / 3.0, 4, rot_y(l_tags))
l_down = tileset.add_mul(1.0 / 3.0, 4, rot_y(rot_y(rot_y(l_tags))))

tileset = tileset.tileset

tiles = np.zeros((tileset.num_tiles(), 3, 3, 3), dtype=np.uint8)

tiles[line_x][1, 1, :] = 1
tiles[line_y][1, :, 1] = 1
tiles[line_z][:, 1, 1] = 1

tiles[l_x[0]][1, 1, 1:] = 1
tiles[l_x[0]][1, 1:, 1] = 1

tiles[l_x[1]][1, 1, :2] = 1
tiles[l_x[1]][1, 1:, 1] = 1

tiles[l_x[2]][1, 1, :2] = 1
tiles[l_x[2]][1, :2, 1] = 1

tiles[l_x[3]][1, 1, 1:] = 1
tiles[l_x[3]][1, :2, 1] = 1

tiles[l_up[0]][1:, 1, 1] = 1
tiles[l_up[0]][1, 1:, 1] = 1

tiles[l_up[1]][1:, 1, 1] = 1
tiles[l_up[1]][1, 1, :2] = 1


tiles[l_up[3]][1:, 1, 1] = 1
tiles[l_up[3]][1, 1, 1:] = 1

tiles[l_up[2]][1:, 1, 1] = 1
tiles[l_up[2]][1, :2, 1] = 1


tiles[l_down[0]][:2, 1, 1] = 1
tiles[l_down[0]][1, 1:, 1] = 1

tiles[l_down[1]][:2, 1, 1] = 1
tiles[l_down[1]][1, 1, :2] = 1

tiles[l_down[3]][:2, 1, 1] = 1
tiles[l_down[3]][1, 1, 1:] = 1

tiles[l_down[2]][:2, 1, 1] = 1
tiles[l_down[2]][1, :2, 1] = 1


output = np.zeros((dim * 3, dim * 3, dim * 3), dtype=np.uint8)

wfc = tileset.create_wfc((dim, dim, dim))

writer = UsdWriter("pipes2.usdc")

any_contradictions = collapse_all_with_callback(
    wfc, lambda: writer.write(map_3d(wfc.values(), output, tiles)), skip=10
)

assert not any_contradictions
