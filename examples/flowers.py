from markov import rep, index_for_colour, PatternWithOptions, TevClient
import numpy as np
import subprocess
import time

subprocess.Popen("tev", stdout=subprocess.PIPE)
time.sleep(1.0 / 60.0)
client = TevClient()

width = 1924
height = 1924
arr = np.zeros((1924, 1924), dtype=np.uint8)
arr[: height - 4] = index_for_colour("U")
arr[height - 4] = index_for_colour("G")
arr[height - 3 :] = index_for_colour("N")

client.send_image("initial", arr)
rep(
    arr,
    [
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
            allow_rot90=False,
        ),
        PatternWithOptions("""
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
    allow_rot90 = False),
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
            allow_rot90=False,
        ),

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
    ],
    priority_after=4,
)
rep(
    arr,
    [
        """
            ***,
            *P*,
            ***,
            =
            *Y*,
            YEY,
            *Y*
        """
    ]
)
client.send_image("final", arr)
