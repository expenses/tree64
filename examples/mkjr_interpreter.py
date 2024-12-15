import sys
from markov.interpreter import interpret
import numpy as np

arr = np.zeros((128, 128), dtype=np.uint8)
interpret(open(sys.argv[1]).read(), arr)
