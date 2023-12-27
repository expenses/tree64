from markov import TevClient, colour_image
import subprocess
import time
import ffmpeg
import numpy as np


def spawn_tev():
    subprocess.Popen("tev", stdout=subprocess.PIPE)
    time.sleep(1.0 / 60.0)
    return TevClient()

class FfmpegOutput:
    def __init__(self, filename, width, height, skip=1, framerate=60):
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
