from markov.markov import *
import numpy as np

ONCE = NodeSettings(count=1)


def all(pattern, **kwargs):
    return Pattern(pattern, apply_all=True, **kwargs)


def rep(array, node, callback=None, writer=None, inplace=True):
    if not inplace:
        array = array.copy()
    if writer != None:
        callback = lambda index: writer.write(array)
    markov.rep(array, node, callback)
    return array


NO_FLIPS = [[False, False, False]]
NO_SHUFFLES = [[0, 1, 2]]

Y_IS_Z = [[0, 2, 1], [2, 0, 1]]
Y_IS_Z_FLIP = [[False, True, False]]
Y_IS_Z_TOGGLE_X = [[False, True, False], [True, True, False]]

TOGGLE_X = [
    [False, False, False],
    [True, False, False],
]
TOGGLE_XY = [
    [False, False, False],
    [True, False, False],
    [False, True, False],
    [True, True, False],
]
ROT_AROUND_Z = [[0, 1, 2], [1, 0, 2]]


# https://pico-8.fandom.com/wiki/Palette
PICO8_PALETTE = Palette(
    [
        [0, 0, 0],
        [255, 241, 232],
        [255, 0, 7],
        [29, 43, 83],
        [126, 37, 83],
        [0, 135, 81],
        [171, 82, 54],
        [95, 87, 79],
        [194, 195, 199],
        [255, 163, 0],
        [255, 236, 39],
        [0, 228, 54],
        [41, 173, 255],
        [131, 118, 156],
        [255, 119, 168],
        [255, 204, 170],
    ]
    + [[128, 128, 128]] * 100
)


class FfmpegWriter:
    def __init__(self, filename, dims, skip=1, framerate=60):
        import ffmpeg

        width, height = dims

        self.process = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(width, height),
                framerate=framerate,
            )
            .output(filename, crf=0, vcodec="libx264", preset="ultrafast")
            .global_args("-hide_banner")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        self.buffer = np.zeros((width, height, 3), dtype=np.uint8)
        self.skip = skip
        self.index = 0

    def write(self, array, palette=PICO8_PALETTE):
        if self.index % self.skip == 0:
            colour_image(self.buffer, array, palette)
            self.process.stdin.write(self.buffer)
        self.index += 1


def save_image(filename, arr, palette=PICO8_PALETTE):
    from PIL import Image

    width, height = arr.shape
    buffer = np.zeros((width, height, 3), dtype=np.uint8)
    colour_image(buffer, arr, palette)
    Image.fromarray(buffer).save(filename)


def add_to_usd_stage(prim_path, stage, arr, time=1, palette=PICO8_PALETTE):
    from pxr import Sdf, UsdGeom

    positions, colours, indices = mesh_voxels(np.pad(arr, 1))
    colours = [palette.linear[x] for x in colours]
    prim = stage.DefinePrim(prim_path, "Mesh")
    prim.CreateAttribute("points", Sdf.ValueTypeNames.Float3Array).Set(positions, time)

    colours_attr = prim.CreateAttribute(
        "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
    )
    colours_attr.Set(colours, time)
    UsdGeom.Primvar(colours_attr).SetInterpolation("uniform")

    num_faces = len(positions) // 4

    prim.CreateAttribute("faceVertexCounts", Sdf.ValueTypeNames.IntArray).Set(
        [4] * num_faces, time
    )

    prim.CreateAttribute("faceVertexIndices", Sdf.ValueTypeNames.IntArray).Set(
        indices, time
    )


def write_usd(filename, arr, palette=PICO8_PALETTE):
    from pxr import Usd

    stage = Usd.Stage.CreateNew(filename)
    stage.SetMetadata("upAxis", "Z")
    add_to_usd_stage("/mesh", stage, arr, palette=palette)
    stage.Save()


class UsdWriter:
    def __init__(self, filename, skip=1):
        from pxr import Usd

        self.stage = Usd.Stage.CreateNew(filename)
        self.stage.SetMetadata("upAxis", "Z")
        self.stage.SetStartTimeCode(1)
        self.frameindex = 0
        self.index = 0
        self.skip = skip

    def write(self, arr):
        index = self.index
        self.index += 1
        if index % self.skip != 0:
            return
        self.frameindex += 1
        print(self.frameindex)
        add_to_usd_stage("/arr", self.stage, arr, time=self.frameindex)
        self.stage.SetEndTimeCode(self.frameindex)
        self.stage.Save()


def rot_z(d):
    if type(d) is dict:
        m = d
        n = {}
        for d, v in m.items():
            n[rot_z(d)] = v
        return n

    if d == "x":
        return "y"
    if d == "y":
        return "negx"
    if d == "negx":
        return "negy"
    if d == "negy":
        return "x"
    return d


def rot_y(d):
    if type(d) is dict:
        m = d
        n = {}
        for d, v in m.items():
            n[rot_y(d)] = v
        return n

    if d == "x":
        return "z"
    if d == "z":
        return "negx"
    if d == "negx":
        return "negz"
    if d == "negz":
        return "x"
    return d


class Tags:
    def __init__(self, *tags, incoming=[], outgoing=[]):
        if type(incoming) is not list:
            incoming = [incoming]
        if type(outgoing) is not list:
            outgoing = [outgoing]

        self.incoming = set(incoming + list(tags))
        self.outgoing = set(outgoing + list(tags))

    def merge(self, other):
        self.incoming |= other.incoming
        self.outgoing |= other.outgoing
        other.incoming = self.incoming
        other.outgoing = self.outgoing

    def __repr__(self):
        return f"in: {self.incoming} out: {self.outgoing}"


def wave_from_tiles(tiles):
    wave = 0
    for tile in tiles:
        wave |= 1 << tile
    return wave


def apply_symmetry(tags, symmetry):
    if type(tags) is not dict:
        tags = {"x": tags}

    for dir in ["x", "y", "negx", "negy", "z", "negz"]:
        if not dir in tags:
            tags[dir] = Tags()
        if type(tags[dir]) is str:
            tags[dir] = Tags(tags[dir])

    if symmetry == "X":
        for dir in ["x", "y", "negx", "negy"]:
            for other in tags.values():
                tags[dir].merge(other)
    if symmetry == "I":
        tags["x"].merge(tags["negx"])
        tags["y"].merge(tags["negy"])
    if symmetry == "L":
        tags["x"].merge(tags["y"])
        tags["negx"].merge(tags["negy"])
    if symmetry == "T":
        tags["x"].merge(tags["negx"])
    if symmetry == "X_3d":
        for dir in ["x", "y", "negx", "negy", "z", "negz"]:
            for other in tags.values():
                tags[dir].merge(other)
    if symmetry == "I_3d":
        tags["x"].merge(tags["negx"])
        tags["y"].merge(tags["negy"])
        tags["z"].merge(tags["negz"])
    if symmetry == "L_3d":
        tags["x"].merge(tags["y"])
        tags["negx"].merge(tags["negy"])
        tags["z"].merge(tags["negz"])
    return tags


class TaggingTileset:
    def __init__(self):
        self.tileset = Tileset()
        self.tiles = {}
        self.tag_dir_to_tiles = {}
        # self.blocklist = set()

    def add(self, prob, tags, symmetry=""):
        tile = self.tileset.add(prob)

        tile_tags = {}
        connect_to_tags = {}

        tags = apply_symmetry(tags, symmetry)

        for dir, tags in tags.items():
            if type(tags) is str:
                tags = Tags(tags)

            tile_tags[dir] = tags.incoming
            connect_to_tags[dir] = tags.outgoing

        self.tiles[tile] = connect_to_tags

        for dir, dir_tags in tile_tags.items():
            for tag in dir_tags:
                pair = (FLIPPED[dir], tag)
                if not pair in self.tag_dir_to_tiles:
                    self.tag_dir_to_tiles[pair] = []
                self.tag_dir_to_tiles[pair].append(tile)

        # No point in not doing this unless we're using a blocklist
        self.connect_all()

        return tile

    def add_mul(self, prob, rots, tags, symmetry=""):
        res = []
        prob /= rots
        tags = apply_symmetry(tags, symmetry)
        for i in range(rots):
            res.append(self.add(prob, tags))
            tags = rot_z(tags)
        return res

    def connect_all(self):
        for frm, tags in self.tiles.items():
            for dir, dir_tags in tags.items():
                for tag in dir_tags:
                    if not (dir, tag) in self.tag_dir_to_tiles:
                        # print(f"missing ({dir}, {tag})")
                        continue

                    for to in self.tag_dir_to_tiles[(dir, tag)]:
                        """
                        if (frm, to) in self.blocklist or (to, frm) in self.blocklist:
                            # print("skipping")
                            continue
                        """
                        # print(f"connecting {frm} to {to} along {dir}")
                        self.tileset.connect(frm, to, [dir])


def map_2d(values, output, tiles):
    markov.map_2d(values, output, tiles)
    return output


def map_3d(values, output, tiles):
    markov.map_3d(values, output, tiles)
    return output


FLIPPED = {"x": "negx", "y": "negy", "negx": "x", "negy": "y", "z": "negz", "negz": "z"}


def split_xml_tile(string, tiles):
    split = string.split(" ")
    if len(split) == 1:
        return split[0], "x"
    axis = ["x", "negy", "negx", "y"]

    if split[0] in tiles:
        return split[0], axis[int(split[1])]
    else:
        zs = split[0].count("z")
        xs = split[0].count("x")
        ys = split[0].count("y")
        return split[1], axis[zs]


def read_xml(filename, symmetry_override={}):
    import xmltodict

    tiles = {}
    d = xmltodict.parse(open(filename).read())

    if "set" in d:
        d = d["set"]
    elif "tileset" in d:
        d = d["tileset"]

    for tile in d["tiles"]["tile"]:
        name = tile["@name"]
        tiles[name] = {
            "symmetry": (
                tile["@symmetry"]
                if "@symmetry" in tile
                else (symmetry_override[name] if name in symmetry_override else "")
            ),
            "weight": float(tile["@weight"]) if "@weight" in tile else 1.0,
            "conns": {"x": Tags(), "y": Tags(), "negx": Tags(), "negy": Tags()},
        }

    for neighbor in d["neighbors"]["neighbor"]:
        left, left_axis = split_xml_tile(neighbor["@left"], tiles)
        right, right_axis = split_xml_tile(neighbor["@right"], tiles)
        right_axis = FLIPPED[right_axis]

        incoming = f"{left}_{left_axis}"
        outgoing = f"{right}_{right_axis}"
        tiles[left]["conns"][left_axis].incoming.add(incoming)
        tiles[right]["conns"][right_axis].outgoing.add(incoming)

        tiles[left]["conns"][left_axis].outgoing.add(outgoing)
        tiles[right]["conns"][right_axis].incoming.add(outgoing)

    tileset = TaggingTileset()

    for name, tile in tiles.items():
        times = 4
        if tile["symmetry"] == "X":
            times = 1
        elif tile["symmetry"] == "I":
            times = 2

        tiles[name] = tileset.add_mul(
            tile["weight"], times, tile["conns"], symmetry=tile["symmetry"]
        )

    return tiles, tileset


def collapse_all_with_callback(wfc, callback, skip=1):
    i = 0
    any_contradictions = False
    while True:
        value = wfc.find_lowest_entropy()
        if value is None:
            break
        index, tile = value
        any_contradictions |= wfc.collapse(index, tile)
        if i % skip == 0:
            callback()
        i += 1
    return any_contradictions


def replace_values(input, tiles, output=None):
    if output is None:
        output = np.copy(input)

    for tiles, value in tiles:
        if type(value) is str:
            value = index_for_colour(value)

        for tile in tiles:
            output[input == tile] = value

    return output
