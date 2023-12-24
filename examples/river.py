from markov import rep, rep_all, One
import numpy as np
import subprocess
from util import spawn_tev

tev_client = spawn_tev()

arr = np.zeros((2048, 1024), dtype=np.uint8)
rep(arr, "B=W", times=1)
rep(arr, "B=R", times=1)
tev_client.send_image("voronoi points", arr)
rep(arr, One("BW=WW", "BR=RR"))
tev_client.send_image("voronoi", arr)
rep_all(arr, ("WR=UU", "W=B", "R=B"))
tev_client.send_image("river", arr)
rep_all(arr, ("UB=UU", "UB=UG"))
tev_client.send_image("wider river and shore", arr)
rep(arr, "B=E", times=13)
tev_client.send_image("forest points", arr)
rep(arr, One("BE=EE", "BG=GG"))
tev_client.send_image("final", arr)
