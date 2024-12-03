from markov import *
import numpy as np

arr = np.zeros((64, 64, 64), dtype=np.uint8)
arr[0] = 1

start = np.zeros((5, 5, 5), dtype=np.uint8)
start[0] = 1
start_from = start.copy()
start[:, 2, 2] = 1
start[4, 2, 2] = 1

start2 = np.zeros((7, 2, 1), dtype=np.uint8)
start2[:3, 0] = 1
start2[2, 1] = 1
start2_from = start2.copy()
start2[:, 0] = 1
print(start2)

rep(
    arr,
    Markov(
        # One(
        # Sequence(
        Pattern(
            """
        10000,
        10000,
        10000,
        10000,
        11111=
        11111,
        ****1,
        ****1,
        ****1,
        *****,
        """,
            flips=[[False, True, False], [True, True, False]],
            shuffles=Y_IS_Z,
            # ))
        ),
        Pattern((start_from, start), flips=NO_FLIPS, shuffles=NO_SHUFFLES),
        Pattern(
            (start2_from, start2),
            flips=TOGGLE_XY,
            shuffles=NO_SHUFFLES,
            node_settings=ONCE,
        ),
    ),
)

output = UsdWriter("corrode.usdc", skip=500)

rep(
    arr,
    Markov(
        "21,1*=02,2*",
        "21=02",
        "2=0",
        "111=211",
        "1=0",
        # settings=NodeSettings(count=1000)
    ),
    writer=output,
)

# write_usd("towers.usdc", arr)
