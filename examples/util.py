from markov import *
import subprocess
import time
import numpy as np
from PIL import Image


NO_FLIPS = [[False, False, False]]
NO_SHUFFLES = [[0, 1, 2]]

Y_IS_Z = [[0, 2, 1], [2, 0, 1]]
TOGGLE_X = [
    [False, False, False],
    [True, False, False],
]
ROT_AROUND_Z = [[0, 1, 2], [1, 0, 2]]

PALETTE_LETTERS = [
    'B', # Black
    'W', # White
    'R', # Red
    'I', # Dark blue
    'P', # Dark purple
    'E', # Dark green
    'N', # Brown
    'D', # Dark grey (Dead)
    'A', # Light grey (Alive)
    'O', # Orange
    'Y', # Yellow
    'G', # Green
    'U', # Blue
    'S', # Lavender
    'K', # Pink
    'F', # Light peach
]

# https://pico-8.fandom.com/wiki/Palette
PALETTE = Palette(
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

def array_from_chars(chars):
    width = None
    array = []
    for char in chars:
        if char == ' ' or char == '\n':
            continue
        elif char == ',':
            if width == None:
                width = len(array)
        elif char == '*':
            array.append(255)
        else:
            try:
                array.append(int(char))
            except ValueError:
                array.append(PALETTE_LETTERS.index(char))

    if width == None:
        return np.array(array)
    else:
        return np.reshape(array, (-1, width))

def save_image(filename, arr):
    width, height = arr.shape
    buffer = np.zeros((width, height, 3), dtype=np.uint8)
    colour_image(buffer, arr, PALETTE)
    Image.fromarray(buffer).save(filename)


def rep2(arr, *args, **kwargs):
    arr = arr.copy()
    callback = None
    if "ffmpeg" in kwargs:
        ffmpeg = kwargs["ffmpeg"]
        callback = lambda index: ffmpeg.write(arr)
        del kwargs["ffmpeg"]
    rep(arr, *args, callback=callback, **kwargs)
    return arr


def spawn_tev():
    subprocess.Popen("tev", stdout=subprocess.PIPE)
    time.sleep(1.0 / 60.0)
    return TevClient()


class FfmpegOutput:
    def __init__(self, filename, width, height, skip=1, framerate=60):
        import ffmpeg

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

    def write(self, array):
        if self.index % self.skip == 0:
            colour_image(self.buffer, array, PALETTE)
            self.process.stdin.write(self.buffer)
        self.index += 1

    def close():
        self.process.stdin.close()


def save_as_voxels(filename, arr):
    from voxypy.models import Entity

    entity = Entity(data=arr.astype(int))

    # Copied from source code.
    palette = [
        (r, g, b, 255 if i > 0 else 0) for i, (r, g, b) in enumerate(PALETTE.srgb)
    ]

    entity.set_palette(palette)
    entity.save(filename)


class CompressedVoxelsOutput:
    def __init__(self, filename):
        import zstandard as zstd

        self.ctx = zstd.ZstdCompressor()
        self.file = open(filename, "wb")
        self.writer = self.ctx.stream_writer(self.file)

    def write(self, array):
        np.save(self.writer, array)

    def close(self):
        self.writer.close()
        self.file.close()


def add_to_usd_stage(prim_path, stage, arr, time=1):
    from pxr import Sdf, UsdGeom

    positions, colours, indices = mesh_voxels(np.pad(arr, 1))
    colours = [PALETTE.linear[x] for x in colours]
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
