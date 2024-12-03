from markov import *
import numpy as np

dim = 64

arr = np.zeros((dim, dim), dtype=np.uint8)
rep(arr, all("B=W", node_settings=ONCE))
rep(arr, all("B=R", node_settings=ONCE))
rep(arr, One("BW=WW", "BR=RR"))
rep(arr, all("WR=UU"))
rep(arr, all("W=B"))
rep(arr, all("R=B"))

rep(arr, all("UB=UU", node_settings=ONCE))
rep(arr, all("UB=UG", node_settings=ONCE))

# rep(arr, Pattern("B=E", node_settings=NodeSettings(count=1)))
# tev_client.send_image("forest points", arr)
rep(arr, "BG=GG")

bottom = arr
arr = np.zeros((dim, dim, dim), dtype=np.uint8)
arr[0] = bottom

output = UsdWriter("falling.usdc")

output.write(arr)


rep(
    arr,
    Pattern(
        f"0=G",
        apply_all=True,
        chance=0.1,
        node_settings=NodeSettings(count=1),
    ),
)

output.write(arr)


rep(
    arr,
    Markov(
        Pattern("UG=U0", shuffles=[[2, 1, 0]], flips=NO_FLIPS),
        Pattern(
            f"G0,G0=00,GG",
            shuffles=[[2, 0, 1], [0, 2, 1]],
            flips=[[False, False, False], [True, False, False]],
        ),
        Pattern("0G=G0", shuffles=[[2, 1, 0]], flips=NO_FLIPS),
    ),
    # writer=output
)

output.write(arr)


house = np.zeros((4, 2, 2), dtype=np.uint8)
house[0] = index_for_colour("G")

house_from = house.copy()

house[1:3] = index_for_colour("N")
house[3, 0, 0] = index_for_colour("D")

rep(arr, Pattern((house_from, house), flips=TOGGLE_XY, shuffles=NO_SHUFFLES))

output.write(arr)


# rep(arr, Pattern('D0=DA', flips=NO_FLIPS, shuffles=[[2,1,0]], apply_all=True, chance=0.75))

# output.write(arr)


# rep(arr, Pattern('A0=0A', flips=NO_FLIPS, shuffles=[[2,1,0]], node_settings=NodeSettings(count=5000)))

rep(
    arr,
    One(
        Pattern("D0=DA", flips=NO_FLIPS, shuffles=[[2, 1, 0]]),
        Pattern("A0=0A", flips=NO_FLIPS, shuffles=[[2, 1, 0]]),
        settings=NodeSettings(count=20000),
    ),
)


output.write(arr)


# rep(arr, Pattern('''
# BBB,
# BBB,
# GGG=
# RRR,
# R*R,
# ***
#''', flips=[[False, True, False]], shuffles=Y_IS_Z))


# write_usd("river.usdc", arr)

# save_image("river.png", arr)
# tev_client.send_image("final", arr)
