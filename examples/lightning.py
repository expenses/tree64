# coding: utf-8
from util import *

arr = np.zeros((128, 128), dtype=np.uint8)
arr[:, :] = 1
arr[1:-1, 1:-1] = 0
ffmpeg = FfmpegOutput("out.avi", width=128, height=128, skip=600)
rep2(arr, "11,10=01,11", times=2000000, ffmpeg=ffmpeg)
# get_ipython().run_line_magic('s', 'scriot.py')
