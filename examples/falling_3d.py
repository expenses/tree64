from util import *
import numpy as np
from pxr import Usd

output = UsdWriter("go2.usdc", skip=10)
arr = np.zeros((16, 16, 16), dtype=np.uint8)
rep(
    arr,
    Pattern(
        "0=2", apply_all=True, chance=1.0 / 3.0, node_settings=NodeSettings(count=1)
    ),
)
rep(
    arr,
    Pattern("02=20", flips=NO_FLIPS, shuffles=[[1, 2, 0]]),
    callback=lambda index: output.write(arr),
)
output.stage.Save()
