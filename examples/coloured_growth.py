from markov import rep, index_for_colour, Markov, One, rep_all
from util import spawn_tev
import numpy as np
import subprocess
import sys

tev_client = spawn_tev()

dim = 1024
arr = np.zeros((dim, dim), dtype=np.uint8)
arr[dim // 2, dim // 2] = index_for_colour("R")
rep(arr, One("RB=RO", "OB=OY", "YB=YG", "GB=GU", "UB=UI", "IB=IP", "PB=PR"))
tev_client.send_image("growth", arr)
