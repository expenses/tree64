from markov import *


def run(arr, writer):
    rep(arr, do_times("B=W", 1), writer=writer)
    rep(arr, do_times("B=R", 1), writer=writer)
    # tev_client.send_image("voronoi points", arr)
    rep(arr, One("BW=WW", "BR=RR"), writer=writer)
    # tev_client.send_image("voronoi", arr)
    for pattern in ["WR=UU", "W=B", "R=B", "UB=UU", "UB=UG"]:
        rep(arr, all_once(pattern), writer=writer)
    # tev_client.send_image("river", arr)
    # tev_client.send_image("wider river and shore", arr)
    rep(arr, do_times("B=E", 13), writer=writer)
    # tev_client.send_image("forest points", arr)
    rep(arr, One("BE=EE", "BG=GG"), writer=writer)
    # tev_client.send_image("final", arr)


run_example("river", 128, run)
