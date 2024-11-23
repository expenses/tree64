# coding: utf-8
from util import *

dim = 128
arr = np.zeros((dim, dim), dtype=np.uint8)
ffmpeg = FfmpegOutput("out.avi", width=dim, height=dim, skip=300)


def sand(arr, c):
    global ffmpeg
    rep_all(arr, (f"0={c}",), chance=0.1)
    return rep2(
        arr,
        Markov(
            One(
                PatternWithOptions(
                    f"{c}0,{c}0=00,{c}{c}",
                    allow_dimension_shuffling=False,
                    allow_flip=False,
                ),
                PatternWithOptions(
                    f"0{c},0{c}=00,{c}{c}",
                    allow_dimension_shuffling=False,
                    allow_flip=False,
                ),
            ),
            PatternWithOptions(
                f"{c},0=0,{c}", allow_dimension_shuffling=False, allow_flip=False
            ),
        ),
        ffmpeg=ffmpeg,
    )


def rainbow(arr):
    arr = sand(arr, "R")
    arr = sand(arr, "O")
    arr = sand(arr, "Y")
    arr = sand(arr, "G")
    arr = sand(arr, "U")
    arr = sand(arr, "I")
    arr = sand(arr, "P")
    return arr


for i in range(1000):
    arr = rainbow(arr)
