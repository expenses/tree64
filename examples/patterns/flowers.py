from markov import *


def run(arr, writer):
    width = arr.shape[1]
    height = arr.shape[0]

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
        writer=writer,
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
        writer=writer,
    )


run_example("flowers", 128, run, disable_3d=True)
