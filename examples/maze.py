from markov import rep, Markov
import numpy as np
import subprocess
from util import spawn_tev

tev_client = spawn_tev()

dim = 1024
arr = np.zeros((dim, dim), dtype=np.uint8)

rep(arr, "0=2", times=1)
rep(arr, Markov("200=332", "332=211"))
tev_client.send_image("maze", arr)
