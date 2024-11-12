from markov import TevClient, colour_image
import subprocess
import time
import numpy as np

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


def save_as_voxels(filename, arr):
    from voxypy.models import Entity

    entity = Entity(data=arr.astype(int))

    # Copied from source code.
    palette = [
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
    palette = [(r, g, b, 255 if i > 0 else 0) for i, (r, g, b) in enumerate(palette)]

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
