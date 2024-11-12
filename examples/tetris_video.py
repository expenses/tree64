from markov import rep, index_for_colour, Markov, One, rep_all
from util import spawn_tev, FfmpegOutput
import numpy as np
import subprocess
import sys

tev_client = spawn_tev()

dim = 128
arr = np.zeros((dim, dim), dtype=np.uint8)

output = FfmpegOutput("tetris.mp4", dim, dim, skip=10)

callback = lambda index: output.write(arr)

rep(
    arr,
    Markov(
        One("RW=RR", "GW=GG", "UW=UU"),
        One("W=R", "W=G", "W=U"),
        One(
            """
            BBB*,
            BBB*,
            BBBB,
            BBBB,
            BBBB
            =
            ****,
            *W**,
            *W**,
            *WW*
            ****
        """
        ),
    ),
    callback=callback
)
tev_client.send_image("tetris", arr)
