from markov import *


def run(arr, writer):
    # Place a point in the exact middle
    put_middle(arr, index_for_colour("P"))
    # Generate a grid
    rep(arr, "PBB=PBP", writer=writer)
    # tev_client.send_image("grid", arr)
    # Place rooms
    rep(
        arr,
        """
            PBPBPBPBP,
            BBBBBBBBB,
            PBPBPBPBP,
            BBBBBBBBB,
            PBPBPBPBP,
            BBBBBBBBB,
            PBPBPBPBP
            =
            WWWWWWWWW,
            WWWWWWWWW,
            WWWWWWWWW,
            WWWWWWWWW,
            WWWWWWWWW,
            WWWWWWWWW,
            WWWWWWWWW
        """,
        writer=writer,
    )
    # tev_client.send_image("placed rooms", arr)
    # Create mazes in the empty spaces using backtracing, place walkers when a maze is complete
    rep(arr, Markov("RBP=UUR", "UUR=RWW", "P=R"), writer=writer)
    # Replace one walker, turn the rest into paths
    rep(arr, Pattern("R=A", node_settings=ONCE), writer=writer)
    rep(arr, "R=W", writer=writer)
    # tev_client.send_image("mazes", arr)
    # Spread a grid out, making sure all rooms are connected
    rep(arr, Markov("AWW=AWA", "ABW=AWA"), writer=writer)
    rep(arr, "A=W", writer=writer)
    # tev_client.send_image("connected rooms", arr)
    # Create a few random connections
    rep(arr, Pattern("WBW=WWW", node_settings=NodeSettings(count=5)), writer=writer)
    # tev_client.send_image("random connections", arr)
    # Remove deadends
    rep(
        arr,
        """
            BBB,
            BWB
            =
            BBB,
            BBB
    """,
        writer=writer,
    )
    # tev_client.send_image("removed dead ends", arr)


run_example("dungeon", 128, run, disable_3d=True)
