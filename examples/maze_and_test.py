from markov import rep, Markov
from util import spawn_tev
import numpy as np
import subprocess

tev_client = spawn_tev()

dim = 512
arr = np.zeros((dim, dim), dtype=np.uint8)

rep(arr, "B=R", times=1)
rep(arr, Markov("RBB=GGR", "GGR=RWW"))
rep(arr, "R=W")
tev_client.send_image("maze", arr)
rep(arr, "BWW=BRW", times=1)
rep(arr, "BWW=BGW", times=1)
tev_client.send_image("points", arr)
rep(arr, Markov("RWG=UUU", "RWW=UUR", "UUR=RFF"))
tev_client.send_image("maze walked", arr)
rep(arr, "F=W")
tev_client.send_image("path", arr)
