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

PALETTE = [
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


def save_image(filename, arr):
    width, height = arr.shape
    buffer = np.zeros((width, height, 3), dtype=np.uint8)
    colour_image(buffer, arr)
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
            colour_image(self.buffer, array)
            self.process.stdin.write(self.buffer)
        self.index += 1

    def close():
        self.process.stdin.close()


def save_as_voxels(filename, arr):
    from voxypy.models import Entity

    entity = Entity(data=arr.astype(int))

    # Copied from source code.
    palette = [(r, g, b, 255 if i > 0 else 0) for i, (r, g, b) in enumerate(PALETTE)]

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


def add_to_usd_stage(prim_path, stage, arr):
    from pxr import Sdf, UsdGeom

    positions, colours, indices = mesh_voxels(np.pad(arr, 1))
    colours = [[v / 255.0 for v in PALETTE[x]] for x in colours]
    prim = stage.DefinePrim(prim_path, "Mesh")
    prim.CreateAttribute("points", Sdf.ValueTypeNames.Float3Array).Set(positions)

    colours_attr = prim.CreateAttribute(
        "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
    )
    colours_attr.Set(colours)
    UsdGeom.Primvar(colours_attr).SetInterpolation("uniform")

    num_faces = len(positions) // 4

    prim.CreateAttribute("faceVertexCounts", Sdf.ValueTypeNames.IntArray).Set(
        [4] * num_faces
    )

    prim.CreateAttribute("faceVertexIndices", Sdf.ValueTypeNames.IntArray).Set(indices)


def write_usd(filename, arr):
    from pxr import Usd

    stage = Usd.Stage.CreateNew(filename)
    stage.SetMetadata("upAxis", "Z")
    add_to_usd_stage("/mesh", stage, arr)
    stage.Save()
