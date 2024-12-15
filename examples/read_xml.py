import sys
from markov import *
from markov.wfc import XmlTileset

filename = sys.argv[1]

tileset = XmlTileset(filename)
output = tileset.run_wfc_and_map((10, 10, 10))
write_usd("output.usdc", output)
