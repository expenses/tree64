from markov import rep, index_for_colour, PatternWithOptions, TevClient, One, Markov
import numpy as np
import sys
from util import spawn_tev, save_image

width = int(sys.argv[1])
height = int(sys.argv[2])
arr = np.zeros((height, width), dtype=np.uint8)
arr[: height - 4] = index_for_colour("U")
arr[height - 4] = index_for_colour("G")
arr[height - 3 :] = index_for_colour("N")

rep(
    arr,
    Markov(
        One(
            PatternWithOptions(
                """
                    UUU,
                    UUU,
                    UPU
                    =
                    ***,
                    *P*,
                    *E*
                """,
                allow_dimension_shuffling=False,
                allow_flip=False,
            ),
            PatternWithOptions(
                """
                    UUU,
                    UUU,
                    UUU,
                    PUU,
                    **U
                    =
                    ***,
                    *P*,
                    *E*,
                    EE*,
                    ***
                """,
                allow_dimension_shuffling=False,
            ),
            PatternWithOptions(
                """
                    UUUUU,
                    UUUUU,
                    UUUUU,
                    UUPUU,
                    U***U
                    =
                    *****,
                    *P*P*,
                    *E*E*,
                    *EEE*,
                    *****
                """,
                allow_dimension_shuffling=False,
                allow_flip=False,
            ),
            PatternWithOptions(
                """
                UUU,
                UPU,
                UEU,
                UEU,
                =
                *Y*,
                YEY,
                *Y*,
                ***,
            """,
                allow_dimension_shuffling=False,
                allow_flip=False,
            ),
        ),
        PatternWithOptions(
            """
            UUUUU,
            UUUUU,
            UUUUU,
            GGGGG,
            NNNNN
            =
            UUUUU,
            UUPUU,
            UUEUU,
            GGEGG,
            NNENN
        """,
            allow_dimension_shuffling=False,
            allow_flip=False,
        ),
    ),
)
rep(
    arr,
    PatternWithOptions(
        """
            ***,
            *P*,
            ***,
            =
            *Y*,
            YEY,
            *Y*
        """,
        allow_dimension_shuffling=False,
        allow_flip=False,
    ),
)
save_image("flowers.png", arr)
