from markov import *


def run(arr, writer):
    arr[:, :] = index_for_colour("U")
    arr[1 : arr.shape[0] - 1, 1 : arr.shape[1] - 1] = 0
    arr[arr.shape[0] // 2, arr.shape[1] // 2] = index_for_colour("W")
    rep(arr, do_times("BW=WW", 1), writer=writer)

    backbone_length = 400
    num_branches = 20
    length_of_branches = 2300
    lake_chance = 0.00003 * 10

    rep(
        arr,
        do_times(
            """
            BWB,
            BBB
            =
            BWB,
            BWB
        """,
            backbone_length,
        ),
        writer=writer,
    )
    # tev_client.send_image("backbone", arr)
    rep(
        arr,
        """
            BWB,
            BWB,
            BBB
            =
            BWB,
            BRB,
            BBB
        """,
        writer=writer,
    )
    # tev_client.send_image("edge blockers", arr)
    rep(
        arr,
        do_times(
            """
            BBB,
            WWW
            =
            BWB,
            WWW
        """,
            num_branches,
        ),
        writer=writer,
    )
    # tev_client.send_image("offshoots", arr)
    rep(
        arr,
        do_times(
            """
            BBB,
            BWB
            =
            BWB,
            BWB
        """,
            length_of_branches,
        ),
        writer=writer,
    )
    rep(arr, "R=W", writer=writer)
    # tev_client.send_image("grown offshoots", arr)
    rep(arr, all("W=U", chance=lake_chance), writer=writer)
    # tev_client.send_image("sea points", arr)
    rep(arr, One("WB=WW", "UB=UU"), writer=writer)
    # tev_client.send_image("island", arr)

    # rep(arr, all_once("U=I"), writer=writer)
    # rep(arr, all_once("IW=UW"), writer=writer)
    # rep(arr, all_once("UI=UU"), writer=writer)
    rep(arr, all_once("WU=WR"), writer=writer)
    rep(
        arr,
        all("RU=RR", chance=0.5, node_settings=NodeSettings(count=30)),
        writer=writer,
    )
    rep(arr, all_once("U=I"))
    rep(arr, all_once("R=U"))

    rep(arr, all("UW=UG"))
    rep(arr, all("GW=GG", chance=0.5, node_settings=NodeSettings(count=80)))
    rep(arr, all_once("WW=WI", chance=0.000015))
    rep(arr, all("G=W"))


run_example("island", 800, run, disable_3d=True)
