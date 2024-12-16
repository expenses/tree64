from markov.markov import *
import numpy as np
import argparse

ONCE = NodeSettings(count=1)


def do_times(pattern, count, **kwargs):
    return Pattern(pattern, node_settings=NodeSettings(count=count), **kwargs)


def all_once(pattern, **kwargs):
    return All(pattern, settings=ONCE, **kwargs)


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

        height, width = dims

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


def map_2d(values, output, tiles):
    markov.map_2d(values, output, tiles)
    return output


def map_3d(values, output, tiles):
    markov.map_3d(values, output, tiles)
    return output


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
    args.dims = list(reversed(args.dims))

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

    array = vox.get_dense()

    limited_palette = {"B", "Y", "D", "A", "W", "P", "R", "F", "U", "E", "N", "C"}
    limited_palette = [PICO8_PALETTE.srgb[index_for_colour(c)] for c in limited_palette]
    rgb_palette = [(r,g,b) for r,g,b,a in vox.get_palette()]
    print(list(find_closest_pairs(rgb_palette, limited_palette)))

    replacements = []
    for i, c in enumerate(find_closest_pairs(rgb_palette, limited_palette)):
        if not (array == i).any():
            continue
        if rgb_palette[i] == tuple(limited_palette[c]):
            continue
        print(rgb_palette[i], c, limited_palette[c])
        replacements.append(([i], PICO8_PALETTE.srgb.index(limited_palette[c])))

    '''
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
    '''
    palette = Palette([(r, g, b) for (r, g, b, a) in vox.get_palette()])

    return replace_values(array, replacements)


def load_mkjr_vox(filename):
    return np.rot90(
        load_vox(filename),
        axes=(0, 2),
    )
