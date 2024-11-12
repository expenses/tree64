from markov import rep, index_for_colour, Markov, One, rep_all
from util import spawn_tev
import numpy as np
import subprocess
import sys

tev_client = spawn_tev()

dim = 400
arr = np.zeros((dim, dim), dtype=np.uint8)
arr[:, :] = index_for_colour("U")
arr[1 : dim - 1, 1 : dim - 1] = 0
arr[200, 200] = index_for_colour("W")
tev_client.send_image("init", arr)
rep(arr, "BW=WW", times=1)
rep(
    arr,
    """
        BWB,
        BBB
        =
        BWB,
        BWB
    """,
    times=200,
)
tev_client.send_image("backbone", arr)
rep(
    arr,
    """
        BWB,
        BWB,
        BBB
        =
        BWB,
        BRB,
        BBB
    """,
)
tev_client.send_image("edge blockers", arr)
rep(
    arr,
    """
        BBB,
        WWW
        =
        BWB,
        WWW
    """,
    times=20,
)
tev_client.send_image("offshoots", arr)
rep(
    arr,
    """
        BBB,
        BWB
        =
        BWB,
        BWB
    """,
    times=500,
)
rep(arr, "R=W")
tev_client.send_image("grown offshoots", arr)
rep(arr, "B=U", times=20)
tev_client.send_image("sea points", arr)
rep(arr, One("WB=WW", "UB=UU"))
tev_client.send_image("island", arr)
