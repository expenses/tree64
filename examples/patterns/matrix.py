from markov import *


def run(arr, writer):
    arr[0, arr.shape[1] // 2, arr.shape[2] // 2] = 1

    rep(
        arr,
        Markov(
            """
            *2*,
            212,
            *2*=
            *2*,
            222,
            *2*
            """,
            "101=121",
            "100=121",
        ),
        writer=writer,
    )


run_example("matrix", 32, run, is_default_3d=True)
