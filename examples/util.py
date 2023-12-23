from markov import TevClient
import subprocess
import time


def spawn_tev():
    subprocess.Popen("tev", stdout=subprocess.PIPE)
    time.sleep(1.0 / 60.0)
    return TevClient()
