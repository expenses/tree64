from markov import *


def run(arr, writer):
    rep(arr, all_once(Pattern("B=W", chance=0.015)), writer=writer)
    rep(arr, do_times("W=R", 1), writer=writer)
    rep(
        arr,
        Markov("21=22", One("01=10", "20=02"), settings=NodeSettings(count=600000)),
        writer=writer,
    )


run_example("sim", 128, run)
