# coding: utf-8
from util import *

arr = np.zeros((128, 128), dtype=np.uint8)
arr[:, :] = 1
arr[1:-1, 1:-1] = 0
ffmpeg = FfmpegOutput("out.avi", width=128, height=128, skip=5)
rep2(arr, Markov("1100=1110", "100=110"), ffmpeg=ffmpeg)
