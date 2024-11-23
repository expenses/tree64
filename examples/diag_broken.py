# coding: utf-8
from util import *

arr = np.zeros((128, 128), dtype=np.uint8)
ffmpeg = FfmpegOutput("out.avi", width=128, height=128, skip=3)
rep2(
    arr,
    Markov(
        """
10*,
010,
*00=
10*,
010,
*01
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
