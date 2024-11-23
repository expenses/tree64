# coding: utf-8
from util import *

dim = 128
ffmpeg = FfmpegOutput("out.avi", width=dim, height=dim, skip=200)
arr = np.zeros((dim, dim), dtype=np.uint8)
rep_all(arr, ("B=W",), chance=0.015)
rep(arr, "W=R", times=1)
rep2(arr, Markov("21=22", One("01=10", "20=02")), times=600000, ffmpeg=ffmpeg)
