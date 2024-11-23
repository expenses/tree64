# coding: utf-8
from util import *

dim = 256
arr = np.zeros((dim, dim), dtype=np.uint8)
ffmpeg = FfmpegOutput("out.avi", width=dim, height=dim, skip=8)
rep2(
    arr,
    Markov(
        """
111,
000,
000=
101,
101,
121
""",
        """
000,
000,
000=
000,
111,
000
""",
    ),
    ffmpeg=ffmpeg,
)
