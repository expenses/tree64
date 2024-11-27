# coding: utf-8
from util import *

dim = 128
arr = np.zeros((dim, dim), dtype=np.uint8)
ffmpeg = FfmpegOutput("out.avi", width=dim, height=dim, skip=300)


def sand(c):
    return (
        PatternWithOptions(
            f"0={c}",
            apply_all=True,
            chance=0.1,
            node_settings=NodeSettings(count=1),
        ),
        Markov(
            PatternWithOptions(
                f"{c}0,{c}0=00,{c}{c}",
                shuffles=NO_SHUFFLES,
                flips=[[False, False, False], [True, False, False]],
            ),
            PatternWithOptions(
                f"{c},0=0,{c}",
                flips=NO_FLIPS,
                shuffles=NO_SHUFFLES,
            ),
        ),
    )


rep2(
    arr,
    Sequence(
        *(
            sand("R")
            + sand("O")
            + sand("Y")
            + sand("G")
            + sand("U")
            + sand("I")
            + sand("P")
        )
    ),
    ffmpeg=ffmpeg,
)
