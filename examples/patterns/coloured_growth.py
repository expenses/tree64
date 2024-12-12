from markov import *


def run(arr, writer):
    put_middle(arr, index_for_colour("R"))
    rep(
        arr,
        One("RB=RO", "OB=OY", "YB=YG", "GB=GU", "UB=UI", "IB=IP", "PB=PR"),
        writer=writer,
    )
    # tev_client.send_image("growth", arr)


run_example("coloured_growth", 128, run)
