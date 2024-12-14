from markov.markov import *
import numpy as np
import argparse
import os

ONCE = NodeSettings(count=1)


def do_times(pattern, count, **kwargs):
    return Pattern(pattern, node_settings=NodeSettings(count=count), **kwargs)


def all(pattern, **kwargs):
    return Pattern(pattern, apply_all=True, **kwargs)


def all_once(pattern, **kwargs):
    return Pattern(pattern, apply_all=True, node_settings=ONCE, **kwargs)


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

    if len(arr.shape) == 2:
        arr = arr.reshape((1, arr.shape[0], arr.shape[1]))

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

    def write(self, arr, palette=PICO8_PALETTE):
        index = self.index
        self.index += 1
        if index % self.skip != 0:
            return
        self.frameindex += 1
        print(self.frameindex)
        add_to_usd_stage("/arr", self.stage, arr, time=self.frameindex, palette=palette)
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


def rot_x(d):
    if type(d) is dict:
        m = d
        n = {}
        for d, v in m.items():
            n[rot_x(d)] = v
        return n

    if d == "y":
        return "z"
    if d == "z":
        return "negy"
    if d == "negy":
        return "negz"
    if d == "negz":
        return "y"
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
        return f"(in: {self.incoming} out: {self.outgoing})"


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
    if symmetry == "T_3d":
        tags["x"].merge(tags["negx"])
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


def split_xml_tile(string, tiles, top_to_bottom=False):
    split = string.split(" ")

    default_dir = "negz" if top_to_bottom else "x"

    if len(split) == 1:
        return split[0], default_dir
    axis = ["x", "negy", "negx", "y"]

    if split[0] in tiles:
        return split[0], axis[int(split[1])]
    else:
        zs = split[0].count("z")
        xs = split[0].count("x")
        ys = split[0].count("y")

        if top_to_bottom:
            assert xs == 0 and ys == 0
            return split[1], default_dir

        return split[1], axis[zs]


def rot_z_symmetrical(voxels, rots):
    return (voxels == np.rot90(voxels, axes=(1, 2), k=rots)).all()


class XmlTileset:
    def __init__(self, filename, old=False):
        import xmltodict

        self.tiles = {}
        self.tile_ids = {}
        self.tileset = TaggingTileset()
        self.filename = filename

        parsed = xmltodict.parse(open(filename).read())

        if "set" in parsed:
            self.parsed = parsed["set"]
        elif "tileset" in parsed:
            self.parsed = parsed["tileset"]

        if not old:
            self.read_xml()

    def create_wfc(self, dims):
        return self.tileset.tileset.create_wfc(dims)

    def setup_connections(self):
        for neighbor in self.parsed["neighbors"]["neighbor"]:
            if "@top" in neighbor and "@bottom" in neighbor:
                left, left_axis = split_xml_tile(
                    neighbor["@top"], self.tiles, top_to_bottom=True
                )
                right, right_axis = split_xml_tile(
                    neighbor["@bottom"], self.tiles, top_to_bottom=True
                )
            else:
                left, left_axis = split_xml_tile(neighbor["@left"], self.tiles)
                right, right_axis = split_xml_tile(neighbor["@right"], self.tiles)
            right_axis = FLIPPED[right_axis]

            incoming = f"{left}_{left_axis}"
            outgoing = f"{right}_{right_axis}"
            self.tiles[left].conns[left_axis].incoming.add(incoming)
            self.tiles[right].conns[right_axis].outgoing.add(incoming)

            self.tiles[left].conns[left_axis].outgoing.add(outgoing)
            self.tiles[right].conns[right_axis].incoming.add(outgoing)

        for name, tile in self.tiles.items():
            times = 4
            if tile.symmetry == "X":
                times = 1
            elif tile.symmetry == "I":
                times = 2

            self.tile_ids[name] = self.tileset.add_mul(
                tile.weight, times, tile.conns, symmetry=tile.symmetry
            )

    def create_array_from_models(self, tile_dims):
        array = np.zeros(
            (self.tileset.tileset.num_tiles(),) + tile_dims, dtype=np.uint8
        )
        for name, ids in self.tile_ids.items():
            model = self.tiles[name].model
            add_model_variants_to_model_array(self.tiles[name].model, array, ids)
        return array

    def run_wfc_and_map(self, dims, model_padding=(0, 0, 0)):
        model_shape = next(iter(self.tiles.values())).model.shape
        model_shape = tuple(model_shape[i] + model_padding[i] for i in range(3))
        # dimensions in numpy order for consistency.
        z, y, x = dims
        wfc = self.tileset.tileset.create_wfc((x, y, z))
        model_array = self.create_array_from_models(model_shape)
        wfc.collapse_all_reset_on_contradiction()
        output = np.zeros(
            tuple(dims[i] * model_shape[i] for i in range(3)),
            dtype=np.uint8,
        )
        map_3d(wfc.values(), output, model_array)
        if model_padding != (0, 0, 0):
            output = output[
                : -model_padding[0], : -model_padding[1], : -model_padding[2]
            ]
        return output

    def read_xml_old(self, symmetry_override={}):
        for tile in self.parsed["tiles"]["tile"]:
            name = tile["@name"]
            self.tiles[name] = MutableTile(
                symmetry=(
                    tile["@symmetry"]
                    if "@symmetry" in tile
                    else (symmetry_override[name] if name in symmetry_override else "")
                ),
                weight=float(tile["@weight"]) if "@weight" in tile else 1.0,
            )

        self.setup_connections()

    def read_xml(self):
        for tile in self.parsed["tiles"]["tile"]:
            name = tile["@name"]

            model = load_mkjr_vox(f"{os.path.splitext(self.filename)[0]}/{name}.vox")

            symmetry = ""
            if rot_z_symmetrical(model, 1):
                symmetry = "X"
            elif rot_z_symmetrical(model, 2):
                symmetry = "I"

            self.tiles[name] = MutableTile(
                symmetry=symmetry,
                weight=float(tile["@weight"]) if "@weight" in tile else 1.0,
                model=model,
            )

        self.setup_connections()


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


def run_example(name, default_size, callback, is_default_3d=False, disable_3d=False):
    parser = argparse.ArgumentParser()
    if not disable_3d:
        parser.add_argument(
            "-3", "--3d", dest="three_d", action="store_true", default=is_default_3d
        )
    parser.add_argument("-d", "--dims", nargs="+", type=int, default=[default_size])
    parser.add_argument("-i", "--image", action=argparse.BooleanOptionalAction)
    parser.add_argument("-m", "--model", action=argparse.BooleanOptionalAction)
    parser.add_argument("-v", "--video", action=argparse.BooleanOptionalAction)
    parser.add_argument("--animated-model", action=argparse.BooleanOptionalAction)
    parser.add_argument("-s", "--skip", type=int, default=1)
    args = parser.parse_args()
    print(args)
    is_3d = False if disable_3d else (args.three_d or len(args.dims) >= 3)
    args.dims = args.dims[:3]
    if len(args.dims) == 1:
        args.dims = args.dims * (3 if is_3d else 2)
    if is_3d and len(args.dims) == 2:
        args.dims.append(default_size)

    if args.image and is_3d:
        raise "can't write 3d image"

    if args.video and is_3d:
        raise "can't write 3d video"

    if args.video and args.animated_model:
        raise "Pick between video and animated model"

    if args.video == True and args.image == None:
        args.image = False

    if args.image == None and not is_3d:
        args.image = True

    if args.model == None and is_3d:
        args.model = True

    print(args)

    array = np.zeros(args.dims, dtype=np.uint8)

    writer = None

    if args.video:
        writer = FfmpegWriter(f"{name}.avi", args.dims, skip=args.skip)

    if args.animated_model:
        writer = UsdWriter(f"{name}-animated.usdc", skip=args.skip)

    callback(array, writer)

    if args.image:
        filename = f"{name}.png"
        print(f"writing {filename}")
        save_image(filename, array)

    if args.model:
        filename = f"{name}.usdc"
        print(f"writing {filename}")
        write_usd(filename, array)


def put_middle(array, value):
    array[*(np.array(array.shape) // 2)] = value


# https://stackoverflow.com/a/29643643
def hex2rgb(h):
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def load_vox(filename):
    from voxypy.models import Entity

    vox = Entity().from_file(filename=filename)

    # MarkovJunior voxel models often have the palette ordered differently
    # to the internal palette so we have to fix that.
    replacements = []
    for i, (r, g, b, a) in enumerate(vox.get_palette()):
        if bytes([r, g, b]) in PICO8_PALETTE.srgb:
            new_index = PICO8_PALETTE.srgb.index(bytes([r, g, b]))
            # Make sure we're not changing opaque voxels to transparent.
            if a == 255 and new_index == 0:
                new_index = index_for_colour("@")
            if i == new_index:
                continue
            replacements.append(([i], new_index))

    palette = Palette([(r, g, b) for (r, g, b, a) in vox.get_palette()])

    return replace_values(vox.get_dense(), replacements)


def load_mkjr_vox(filename):
    return np.rot90(
        load_vox(filename),
        axes=(0, 2),
    )


def add_model_variants_to_model_array(model, array, ids):
    for rot, id in enumerate(ids):
        rotated_model = np.rot90(model, axes=(1, 2), k=-rot)
        array[
            id,
            : rotated_model.shape[0],
            : rotated_model.shape[1],
            : rotated_model.shape[2],
        ] = rotated_model


class MutableTile:
    def __init__(self, weight=1.0, symmetry="", model=None):
        self.weight = weight
        self.symmetry = symmetry
        self.conns = {
            "x": Tags(),
            "y": Tags(),
            "negx": Tags(),
            "negy": Tags(),
            "z": Tags(),
            "negz": Tags(),
        }
        self.model = model

    def apply_symmetry(self):
        self.conns = apply_symmetry(self.conns, self.symmetry)
