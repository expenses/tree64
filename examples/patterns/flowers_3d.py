from markov import *


def run(arr, writer):
    arr[3] = index_for_colour("G")
    arr[:3] = index_for_colour("N")

    base = np.zeros((5, 5, 5), dtype=np.uint8)
    base[0] = index_for_colour("N")
    base[1] = index_for_colour("G")
    base_from = base.copy()
    base[0:3, 2, 2] = index_for_colour("E")
    base[3, 2, 2] = index_for_colour("P")

    rep(
        arr,
        Markov(
            One(
                Pattern(
                    """
                        000,
                        000,
                        0P0
                        =
                        ***,
                        *P*,
                        *E*
                    """,
                    shuffles=Y_IS_Z,
                    flips=Y_IS_Z_FLIP,
                ),
                Pattern(
                    """
                        00000,
                        00000,
                        00000,
                        00P00,
                        0***0
                        =
                        *****,
                        *P*P*,
                        *E*E*,
                        *EEE*,
                        *****
                    """,
                    shuffles=Y_IS_Z,
                    flips=Y_IS_Z_FLIP,
                ),
                Pattern(
                    """
                        000,
                        000,
                        000,
                        P00,
                        **0
                        =
                        ***,
                        *P*,
                        *E*,
                        EE*,
                        ***
                    """,
                    shuffles=Y_IS_Z,
                    flips=Y_IS_Z_TOGGLE_X,
                ),
                Pattern(
                    """
                    000,
                    0P0,
                    0E0,
                    0E0,
                    =
                    *Y*,
                    YEY,
                    *Y*,
                    ***,
                """,
                    shuffles=Y_IS_Z,
                    flips=Y_IS_Z_FLIP,
                ),
            ),
            Pattern(
                (base_from, base),
                shuffles=NO_SHUFFLES,
                flips=NO_FLIPS,
            ),
        ),
        writer=writer,
    )
    rep(
        arr,
        Pattern(
            """
        ***,
        *P*,
        ***,
        =
        *Y*,
        YEY,
        *Y*
    """,
            shuffles=Y_IS_Z,
            flips=Y_IS_Z_FLIP,
        ),
        writer=writer,
    )


run_example("flowers_3d", 128, run, is_default_3d=True)
