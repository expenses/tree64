# coding: utf-8
from util import *

dim = 128
ffmpeg = FfmpegOutput("out.avi", width=dim, height=dim, skip=8)
arr = np.zeros((dim, dim), dtype=np.uint8)
arr = rep2(
    arr,
    Markov(
        One(
            PatternWithOptions(
                "10=11", allow_dimension_shuffling=False, allow_flip=False
            ),
            PatternWithOptions(
                "01=11", allow_dimension_shuffling=False, allow_flip=False
            ),
        ),
        PatternWithOptions(
            "0,0,0,0,0=0,0,1,0,0", allow_dimension_shuffling=False, allow_flip=False
        ),
    ),
    ffmpeg=ffmpeg,
)

rep2(
    arr,
    Markov(
        PatternWithOptions(
            "0,1,1=1,1,1", allow_dimension_shuffling=False, allow_flip=False
        ),
        PatternWithOptions(
            "0,1=1,1", allow_dimension_shuffling=False, allow_flip=False
        ),
    ),
    ffmpeg=ffmpeg,
)
