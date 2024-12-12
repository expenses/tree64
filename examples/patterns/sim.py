from markov import *


def run(arr, writer):
    rep(arr, all("B=W", chance=0.015, node_settings=ONCE), writer=writer)
    rep(arr, do_times("W=R", 1), writer=writer)
    rep(
        arr,
        Markov("21=22", One("01=10", "20=02"), settings=NodeSettings(count=600000)),
        writer=writer,
    )


run_example("sim", 128, run)
