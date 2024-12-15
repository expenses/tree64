from markov import *

arr = np.zeros((64, 64), dtype=np.uint8)

arr[32, 32] = 1

rep(arr, Prl("0=1", "1=0", settings=NodeSettings(count=1)))
save_image("out.png", arr)
