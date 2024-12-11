import sys
from markov import *

dim = 128

t, tileset = read_xml(
    sys.argv[1],
    symmetry_override={"Empty": "X", "Cross": "X", "Line": "I", "Turn": "L"},
)

wfc = tileset.tileset.create_wfc((dim, dim, 1))

print(t)

writer = FfmpegWriter("out.avi", (dim, dim))

any_contradictions = collapse_all_with_callback(
    wfc,
    lambda: writer.write(
        replace_values(
            wfc.values()[0],
            [
                # Castle
                # (t["ground"], "E"),
                # (t["river"] + t["riverturn"], "U"),
                # (t["bridge"] + t["tower"], "D"),
                # (t["road"] + t["roadturn"] + t["t"], "A"),
                # (t["wall"] + t["wallriver"] + t["wallroad"], "P")
                # Summer
                # (t["grass"] + t["grasscorner"], "E"),
                # (t["road"] + t["roadturn"], "A"),
                # (t["water_a"] + t["water_b"] + t["water_c"] + t["watercorner"] + t["waterside"] + t["waterturn"], "U"),
                # (t["cliff"] + t["cliffcorner"] + t["cliffturn"], "D")
                # Floorplan
                # (t["empty"], 0),
                # (t["wall"], "P"),
                # (t["floor"], "A")
                # Circles
                # (t["b"] + t["b_i"] + t["b_half"] + t["b_quarter"], 0),
                # (t["w"] + t["w_i"] + t["w_half"] + t["w_quarter"], 1),
                # Circuit
                # (t["component"], "D")
                # Knots2D
                # (t["Empty"], 0),
                # (t["Line"], )
            ],
        )
    ),
    skip=1,
)

assert not any_contradictions
