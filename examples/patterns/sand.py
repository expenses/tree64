from markov import *


def sand(c):
    return (
        All(
            Pattern(
                f"0={c}",
                chance=0.1,
            ),
            settings=NodeSettings(count=1),
        ),
        Markov(
            Pattern(
                f"{c}0,{c}0=00,{c}{c}",
                shuffles=NO_SHUFFLES,
                flips=[[False, False, False], [True, False, False]],
            ),
            Pattern(
                f"{c},0=0,{c}",
                flips=NO_FLIPS,
                shuffles=NO_SHUFFLES,
            ),
        ),
    )


def run(arr, writer):
    rep(
        arr,
        Sequence(
            *(
                sand("R")
                + sand("O")
                + sand("Y")
                + sand("G")
                + sand("U")
                + sand("I")
                + sand("P")
            )
        ),
        writer=writer,
    )


run_example("sand", 128, run)
