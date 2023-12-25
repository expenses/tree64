from markov import rep, index_for_colour, Markov, One, rep_all
from util import spawn_tev
import numpy as np
import subprocess
import sys

tev_client = spawn_tev()

dim = 4096
arr = np.zeros((dim, dim), dtype=np.uint8)
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
)
tev_client.send_image("tetris", arr)
