from markov import *


def run(arr, writer):
    rep(
        arr,
        Markov(
            One("RW=RR", "GW=GG", "UW=UU"),
            One("W=R", "W=G", "W=U"),
            One(
                """
                BBB*,
                BBB*,
                BBBB,
                BBBB,
                BBBB
                =
                ****,
                *W**,
                *W**,
                *WW*
                ****
            """
            ),
        ),
        writer=writer,
    )


run_example("l_pieces", 128, run)
