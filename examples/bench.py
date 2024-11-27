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
save_image("0.png", arr[0])
save_image("1.png", arr[2])
save_image("2.png", arr[4])
save_image("3.png", arr[6])

'''
PatternWithOptions(
            """
  200,
  ***,
  ,,1
  =
  112,
  ***,
  ***
                    """,
            flips=[
                [False, False, False],
                [True, False, False],
            ],
            shuffles=[[0, 2, 1], [2, 0, 1]],
        ),
'''
