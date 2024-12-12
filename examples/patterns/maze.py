from markov import *


def run(arr, writer):
    rep(arr, Pattern("0=2", node_settings=ONCE), writer=writer)
    rep(arr, Markov("200=332", "332=211"), writer=writer)


run_example("maze", 128, run)
