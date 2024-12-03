from markov import *
import subprocess
import time
import numpy as np
from PIL import Image

PALETTE_LETTERS = [
    "B",  # Black
    "W",  # White
    "R",  # Red
    "I",  # Dark blue
    "P",  # Dark purple
    "E",  # Dark green
    "N",  # Brown
    "D",  # Dark grey (Dead)
    "A",  # Light grey (Alive)
    "O",  # Orange
    "Y",  # Yellow
    "G",  # Green
    "U",  # Blue
    "S",  # Lavender
    "K",  # Pink
    "F",  # Light peach
]


def array_from_chars(chars):
    width = None
    array = []
    for char in chars:
        if char == " " or char == "\n":
            continue
        elif char == ",":
            if width == None:
                width = len(array)
        elif char == "*":
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


def spawn_tev():
    subprocess.Popen("tev", stdout=subprocess.PIPE)
    time.sleep(1.0 / 60.0)
    return TevClient()


def save_as_voxels(filename, arr):
    from voxypy.models import Entity

    entity = Entity(data=arr.astype(int))

    # Copied from source code.
    palette = [
        (r, g, b, 255 if i > 0 else 0) for i, (r, g, b) in enumerate(PICO8_PALETTE.srgb)
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
