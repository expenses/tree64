from markov import *


def run(arr, writer):
    arr[0, arr.shape[1] // 2, arr.shape[2] // 2] = 2

    rep(
        arr,
        Markov(
            Pattern(
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
            Pattern(
                "200=112",
                flips=TOGGLE_X,
                shuffles=ROT_AROUND_Z,
            ),
            Pattern("200=112", flips=NO_FLIPS, shuffles=[[2, 1, 0]]),
        ),
        writer=writer,
    )


run_example("bench", 32, run, is_default_3d=True)
