from markov import *


def run(arr, writer):
    arr_2d = np.zeros((arr.shape[1], arr.shape[2]), dtype=np.uint8)
    rep(arr_2d, do_times("B=W", 1))
    rep(arr_2d, do_times("B=R", 1))
    rep(arr_2d, One("BW=WW", "BR=RR"))
    rep(arr_2d, All("WR=UU"))
    rep(arr_2d, All("W=B"))
    rep(arr_2d, All("R=B"))

    rep(arr_2d, all_once("UB=UU"))
    rep(arr_2d, all_once("UB=UG"))

    rep(arr_2d, "BG=GG")

    arr[0] = arr_2d

    rep(
        arr,
        all_once(
            Pattern(
                f"0=G",
                chance=0.1,
            )
        ),
        writer=writer,
    )

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
        writer=writer,
    )

    house = np.zeros((4, 2, 2), dtype=np.uint8)
    house[0] = index_for_colour("G")

    house_from = house.copy()

    house[1:3] = index_for_colour("N")
    house[3, 0, 0] = index_for_colour("D")

    rep(
        arr,
        Pattern((house_from, house), flips=TOGGLE_XY, shuffles=NO_SHUFFLES),
        writer=writer,
    )

    # rep(arr, Pattern('D0=DA', flips=NO_FLIPS, shuffles=[[2,1,0]], apply_all=True, chance=0.75))

    # output.write(arr)

    # rep(arr, Pattern('A0=0A', flips=NO_FLIPS, shuffles=[[2,1,0]], node_settings=NodeSettings(count=5000)))

    rep(
        arr,
        Markov(
            Pattern("D0=DA", flips=NO_FLIPS, shuffles=[[2, 1, 0]]),
            Pattern(
                "0A,0A=00,AA",
                shuffles=[[2, 0, 1], [0, 2, 1]],
                flips=[[False, False, False], [True, False, False]],
            ),
            Pattern("A0=0A", flips=NO_FLIPS, shuffles=[[2, 1, 0]]),
            # settings=NodeSettings(count=20000),
        ),
        writer=writer,
    )

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


run_example("town", 32, run, is_default_3d=True)
