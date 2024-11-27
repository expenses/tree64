# coding: utf-8
from util import *

dim = 64
arr = np.zeros((7, dim, dim), dtype=np.uint8)

ffmpeg = FfmpegOutput("out.avi", width=dim * 2, height=dim * 2)

arr[0, dim // 2, dim // 2] = 2

arr2 = np.zeros((dim * 2, dim * 2), dtype=np.uint8)


def display(index):
    global arr, ffmpeg, arr2
    arr2[:dim, :dim] = arr[0]
    arr2[:dim, dim:] = arr[2]
    arr2[dim:, :dim] = arr[4]
    arr2[dim:, dim:] = arr[6]
    ffmpeg.write(arr2)


rep(
    arr,
    Markov(
        PatternWithOptions(
            """
  2**,
  0**,
  000
  =
  1**,
  1**,
  2**
                    """,
            flips=[
                [False, True, False],
                [True, True, False],
            ],
            shuffles=Y_IS_Z,
        ),
        PatternWithOptions(
            "200=112",
            flips=TOGGLE_X,
            shuffles=ROT_AROUND_Z,
        ),
        PatternWithOptions("200=112", flips=NO_FLIPS, shuffles=[[2, 1, 0]]),
    ),
    callback=display,
)
save_as_voxels("out.vox", arr)
