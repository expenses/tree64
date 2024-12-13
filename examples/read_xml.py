import sys
from markov import *

filename = sys.argv[1]

tileset = XmlTileset(filename)

output = tileset.run_wfc_and_map((3, 9, 9), (3, 3, 3))
write_usd("output.usdc", output)
