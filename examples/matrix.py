# coding: utf-8
from util import *

dim = 128
arr = np.zeros((5, dim, dim), dtype=np.uint8)

ffmpeg = FfmpegOutput("out.avi", width=dim, height=dim, skip=40)

arr[0, dim // 2, dim // 2] = 1

rep(
    arr,
    Markov(
        """
        *2*,
        212,
        *2*=
        *2*,
        222,
        *2*
        """,
        "101=121",
        "100=121",
    ),
    callback=lambda index: ffmpeg.write(arr[2]),
)
write_usd("matrix.usdc", arr)
