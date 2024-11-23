# coding: utf-8
from util import *

arr = np.zeros((128, 128), dtype=np.uint8)
ffmpeg = FfmpegOutput("out.avi", width=128, height=128)
rep2(
    arr,
    Markov(
        """
100,
010,
000=
100,
010,
001
                 """,
        """
*000*,
00000,
00*00
=
**2**,
*1*1*,
1***1
""",
    ),
    ffmpeg=ffmpeg,
)
