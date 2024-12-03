import numpy as np
import sys
from util import *

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
            Pattern(
                """
                    UUU,
                    UUU,
                    UPU
                    =
                    ***,
                    *P*,
                    *E*
                """,
                shuffles=NO_SHUFFLES,
                flips=NO_FLIPS,
            ),
            Pattern(
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
                shuffles=NO_SHUFFLES,
                flips=[[False, False, False], [True, False, False]],
            ),
            Pattern(
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
                shuffles=NO_SHUFFLES,
                flips=NO_FLIPS,
            ),
            Pattern(
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
                shuffles=NO_SHUFFLES,
                flips=NO_FLIPS,
            ),
        ),
        Pattern(
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
            shuffles=NO_SHUFFLES,
            flips=NO_FLIPS,
        ),
    ),
)
rep(
    arr,
    """
    ***,
    *P*,
    ***,
    =
    *Y*,
    YEY,
    *Y*
""",
)
save_image("flowers.png", arr)
