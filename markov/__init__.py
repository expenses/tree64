from markov import markov
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


def write_usd(filename, arr):
    from pxr import Usd

    stage = Usd.Stage.CreateNew(filename)
    stage.SetMetadata("upAxis", "Z")
    add_to_usd_stage("/mesh", stage, arr)
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
