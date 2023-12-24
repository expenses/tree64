from markov import rep, index_for_colour, Markov
from util import spawn_tev
import numpy as np
import subprocess
import sys

tev_client = spawn_tev()

dim = int(sys.argv[1])
middle = dim // 2

p = index_for_colour("P")

arr = np.zeros((dim + 1, dim + 1), dtype=np.uint8)
# Place a point in the exact middle
arr[middle + 1, middle + 1] = p
# Generate a grid
rep(arr, "PBB=PBP")
tev_client.send_image("grid", arr)
# Place rooms
rep(
    arr,
    """
        PBPBPBPBP,
        BBBBBBBBB,
        PBPBPBPBP,
        BBBBBBBBB,
        PBPBPBPBP,
        BBBBBBBBB,
        PBPBPBPBP
        =
        WWWWWWWWW,
        WWWWWWWWW,
        WWWWWWWWW,
        WWWWWWWWW,
        WWWWWWWWW,
        WWWWWWWWW,
        WWWWWWWWW
    """,
)
tev_client.send_image("placed rooms", arr)
# Create mazes in the empty spaces using backtracing, place walkers when a maze is complete
rep(arr, Markov("RBP=UUR", "UUR=RWW", "P=R"))
# Replace one walker, turn the rest into paths
rep(arr, "R=A", times=1)
rep(arr, "R=W")
tev_client.send_image("mazes", arr)
# Spread a grid out, making sure all rooms are connected
rep(arr, Markov("AWW=AWA", "ABW=AWA"))
rep(arr, "A=W")
tev_client.send_image("connected rooms", arr)
# Create a few random connections
rep(arr, "WBW=WWW", times=5)
tev_client.send_image("random connections", arr)
# Remove deadends
rep(
    arr,
    """
        BBB,
        BWB
        =
        BBB,
        BBB
   """,
)
tev_client.send_image("removed dead ends", arr)
