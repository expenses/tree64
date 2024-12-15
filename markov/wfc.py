from markov import Tileset, load_mkjr_vox, map_3d
import os
import numpy as np

FLIPPED = {"x": "negx", "y": "negy", "negx": "x", "negy": "y", "z": "negz", "negz": "z"}


def rot_z(d):
    if type(d) is dict:
        m = d
        n = {}
        for d, v in m.items():
            n[rot_z(d)] = v
        return n

    if d == "x":
        return "y"
    if d == "y":
        return "negx"
    if d == "negx":
        return "negy"
    if d == "negy":
        return "x"
    return d


def rot_y(d):
    if type(d) is dict:
        m = d
        n = {}
        for d, v in m.items():
            n[rot_y(d)] = v
        return n

    if d == "x":
        return "z"
    if d == "z":
        return "negx"
    if d == "negx":
        return "negz"
    if d == "negz":
        return "x"
    return d


def rot_x(d):
    if type(d) is dict:
        m = d
        n = {}
        for d, v in m.items():
            n[rot_x(d)] = v
        return n

    if d == "y":
        return "z"
    if d == "z":
        return "negy"
    if d == "negy":
        return "negz"
    if d == "negz":
        return "y"
    return d


class Tags:
    def __init__(self, *tags, incoming=[], outgoing=[]):
        if type(incoming) is not list:
            incoming = [incoming]
        if type(outgoing) is not list:
            outgoing = [outgoing]

        self.incoming = set(incoming + list(tags))
        self.outgoing = set(outgoing + list(tags))

    def merge(self, other):
        self.incoming |= other.incoming
        self.outgoing |= other.outgoing
        other.incoming = self.incoming
        other.outgoing = self.outgoing

    def __repr__(self):
        return "{}" if len(self.outgoing) == 0 else str(self.outgoing)
        bidirectional = self.incoming & self.outgoing
        string = "("
        if bidirectional != set():
            string += f"{bidirectional}"
        incoming = self.incoming - self.outgoing
        if incoming != set():
            if string != "(":
                string += " "
            string += f"in: {incoming}"
        outgoing = self.outgoing - self.incoming
        if outgoing != set():
            if string != "(":
                string += " "
            string += f"out: {outgoing}"
        string += ")"
        return string


def wave_from_tiles(tiles):
    wave = 0
    for tile in tiles:
        wave |= 1 << tile
    return wave


def apply_symmetry(tags, symmetry):
    if type(tags) is not dict:
        tags = {"x": tags}

    for dir in ["x", "y", "negx", "negy", "z", "negz"]:
        if not dir in tags:
            tags[dir] = Tags()
        if type(tags[dir]) is str:
            tags[dir] = Tags(tags[dir])

    if symmetry == "X":
        for dir in ["x", "y", "negx", "negy"]:
            for other in tags.values():
                tags[dir].merge(other)
    if symmetry == "I":
        tags["x"].merge(tags["negx"])
        tags["y"].merge(tags["negy"])
    if symmetry == "L":
        tags["x"].merge(tags["y"])
        tags["negx"].merge(tags["negy"])
    if symmetry == "T":
        tags["y"].merge(tags["negy"])
    if symmetry == "X_3d":
        for dir in ["x", "y", "negx", "negy", "z", "negz"]:
            for other in tags.values():
                tags[dir].merge(other)
    if symmetry == "I_3d":
        tags["x"].merge(tags["negx"])
        tags["y"].merge(tags["negy"])
        tags["z"].merge(tags["negz"])
    if symmetry == "L_3d":
        tags["x"].merge(tags["y"])
        tags["negx"].merge(tags["negy"])
        tags["z"].merge(tags["negz"])
    if symmetry == "T_3d":
        tags["y"].merge(tags["negy"])
        tags["z"].merge(tags["negz"])
    return tags


class TaggingTileset:
    def __init__(self):
        self.tileset = Tileset()
        self.tiles = {}
        self.tag_dir_to_tiles = {}
        # self.blocklist = set()

    def add(self, prob, tags, symmetry=""):
        tile = self.tileset.add(prob)

        tile_tags = {}
        connect_to_tags = {}

        tags = apply_symmetry(tags, symmetry)

        for dir, tags in tags.items():
            if type(tags) is str:
                tags = Tags(tags)

            tile_tags[dir] = tags.incoming
            connect_to_tags[dir] = tags.outgoing

        self.tiles[tile] = connect_to_tags

        for dir, dir_tags in tile_tags.items():
            for tag in dir_tags:
                pair = (FLIPPED[dir], tag)
                if not pair in self.tag_dir_to_tiles:
                    self.tag_dir_to_tiles[pair] = []
                self.tag_dir_to_tiles[pair].append(tile)

        # No point in not doing this unless we're using a blocklist
        self.connect_all()

        return tile

    def add_mul(self, prob, rots, tags, symmetry=""):
        res = []
        prob /= rots
        tags = apply_symmetry(tags, symmetry)
        for i in range(rots):
            res.append(self.add(prob, tags))
            tags = rot_z(tags)
        return res

    def connect_all(self):
        for frm, tags in self.tiles.items():
            for dir, dir_tags in tags.items():
                for tag in dir_tags:
                    if not (dir, tag) in self.tag_dir_to_tiles:
                        # print(f"missing ({dir}, {tag})")
                        continue

                    for to in self.tag_dir_to_tiles[(dir, tag)]:
                        """
                        if (frm, to) in self.blocklist or (to, frm) in self.blocklist:
                            # print("skipping")
                            continue
                        """
                        # print(f"connecting {frm} to {to} along {dir}")
                        self.tileset.connect(frm, to, [dir])


def split_xml_tile(string, tiles, top_to_bottom=False):
    split = string.split(" ")

    default_dir = "negz" if top_to_bottom else "x"

    if len(split) == 1:
        return split[0], default_dir
    axis = ["x", "negy", "negx", "y"]

    if split[0] in tiles:
        return split[0], axis[int(split[1])]
    else:
        zs = split[0].count("z")
        xs = split[0].count("x")
        ys = split[0].count("y")

        if top_to_bottom:
            assert xs == 0 and ys == 0
            return split[1], default_dir

        return split[1], axis[zs]


def rot_z_symmetrical(voxels, rots):
    return (voxels == np.rot90(voxels, axes=(1, 2), k=rots)).all()


class XmlTileset:
    def __init__(self, filename, old=False):
        import xmltodict

        self.tiles = {}
        self.tile_ids = {}
        self.tileset = TaggingTileset()
        self.filename = filename

        parsed = xmltodict.parse(open(filename).read(), force_list=["tile", "neighbor"])

        if "set" in parsed:
            self.parsed = parsed["set"]
        elif "tileset" in parsed:
            self.parsed = parsed["tileset"]

        if not old:
            self.read_xml()

    def create_wfc(self, dims):
        return self.tileset.tileset.create_wfc(dims)

    def setup_connections(self):
        for neighbor in self.parsed["neighbors"]["neighbor"]:
            if "@top" in neighbor and "@bottom" in neighbor:
                left, left_axis = split_xml_tile(
                    neighbor["@top"], self.tiles, top_to_bottom=True
                )
                right, right_axis = split_xml_tile(
                    neighbor["@bottom"], self.tiles, top_to_bottom=True
                )
            else:
                left, left_axis = split_xml_tile(neighbor["@left"], self.tiles)
                right, right_axis = split_xml_tile(neighbor["@right"], self.tiles)
            right_axis = FLIPPED[right_axis]

            incoming = f"{left}_{left_axis}"
            outgoing = f"{right}_{right_axis}"
            self.tiles[left].conns[left_axis].incoming.add(incoming)
            self.tiles[right].conns[right_axis].outgoing.add(incoming)

            self.tiles[left].conns[left_axis].outgoing.add(outgoing)
            self.tiles[right].conns[right_axis].incoming.add(outgoing)

        for name, tile in self.tiles.items():
            times = 4
            if tile.symmetry == "X":
                times = 1
            elif tile.symmetry == "I":
                times = 2

            self.tile_ids[name] = self.tileset.add_mul(
                tile.weight, times, tile.conns, symmetry=tile.symmetry
            )

    def create_array_from_models(self, tile_dims):
        array = np.zeros(
            (self.tileset.tileset.num_tiles(),) + tile_dims, dtype=np.uint8
        )
        for name, ids in self.tile_ids.items():
            model = self.tiles[name].model
            add_model_variants_to_model_array(self.tiles[name].model, array, ids)
        return array

    def run_wfc_and_map(self, dims, model_padding=(0, 0, 0)):
        model_shape = next(iter(self.tiles.values())).model.shape
        model_shape = tuple(model_shape[i] + model_padding[i] for i in range(3))
        # dimensions in numpy order for consistency.
        z, y, x = dims
        wfc = self.tileset.tileset.create_wfc((x, y, z))
        model_array = self.create_array_from_models(model_shape)
        wfc.collapse_all_reset_on_contradiction()
        output = np.zeros(
            tuple(dims[i] * model_shape[i] for i in range(3)),
            dtype=np.uint8,
        )
        map_3d(wfc.values(), output, model_array)
        if model_padding != (0, 0, 0):
            output = output[
                : -model_padding[0], : -model_padding[1], : -model_padding[2]
            ]
        return output

    def read_xml_old(self, symmetry_override={}):
        for tile in self.parsed["tiles"]["tile"]:
            name = tile["@name"]
            self.tiles[name] = MutableTile(
                symmetry=(
                    tile["@symmetry"]
                    if "@symmetry" in tile
                    else (symmetry_override[name] if name in symmetry_override else "")
                ),
                weight=float(tile["@weight"]) if "@weight" in tile else 1.0,
            )

        self.setup_connections()

    def read_xml(self):
        for tile in self.parsed["tiles"]["tile"]:
            name = tile["@name"]

            model = load_mkjr_vox(f"{os.path.splitext(self.filename)[0]}/{name}.vox")

            symmetry = ""
            if rot_z_symmetrical(model, 1):
                symmetry = "X"
            elif rot_z_symmetrical(model, 2):
                symmetry = "I"

            self.tiles[name] = MutableTile(
                symmetry=symmetry,
                weight=float(tile["@weight"]) if "@weight" in tile else 1.0,
                model=model,
            )

        self.setup_connections()


def collapse_all_with_callback(wfc, callback, skip=1):
    i = 0
    any_contradictions = False
    while True:
        value = wfc.find_lowest_entropy()
        if value is None:
            break
        index, tile = value
        any_contradictions |= wfc.collapse(index, tile)
        if i % skip == 0:
            callback()
        i += 1
    return any_contradictions


def add_model_variants_to_model_array(model, array, ids):
    for rot, id in enumerate(ids):
        rotated_model = np.rot90(model, axes=(1, 2), k=-rot)
        array[
            id,
            : rotated_model.shape[0],
            : rotated_model.shape[1],
            : rotated_model.shape[2],
        ] = rotated_model


def connect_2(left_tile, right_tile, left_edge, right_edge):
    left_tile.incoming.add(left_edge)
    left_tile.outgoing.add(right_edge)
    right_tile.incoming.add(right_edge)
    right_tile.outgoing.add(left_edge)


class MutableTile:
    def __init__(self, weight=1.0, symmetry="", model=None):
        self.weight = weight
        self.symmetry = symmetry
        self.conns = {
            "x": Tags(),
            "y": Tags(),
            "negx": Tags(),
            "negy": Tags(),
            "z": Tags(),
            "negz": Tags(),
        }
        self.model = model

    def apply_symmetry(self):
        self.conns = apply_symmetry(self.conns, self.symmetry)


VARIANTS_FOR_SYMMETRY = {"T_3d": 4, "T": 4, "X_3d": 1, "X": 1, "I_3d": 2, "I": 2}


class MkJrConnector:
    def __init__(self):
        self.tiles = {}

    def add(self, name, symmetry):
        self.tiles[name] = [
            MutableTile(symmetry=symmetry)
            for _ in range(VARIANTS_FOR_SYMMETRY[symmetry])
        ]

    def connect(self, left, left_v, right, right_v, on_z=False):
        axes = ["negz"] if on_z else ["x", "negy", "negx", "y"]

        if len(self.tiles[right]) > len(self.tiles[left]):
            left, right = right, left
            left_v, right_v = right_v, left_v
            left_v += 2

        for variant in range(len(self.tiles[left])):
            left_v_mod = (variant + left_v) % len(self.tiles[left])
            right_v_mod = (variant + right_v) % len(self.tiles[right])
            left_axis = axes[left_v_mod % len(axes)]
            # print(f"{left} {left_v_mod} {left_axis} -> {right} {right_v_mod}")

            right_axis = FLIPPED[left_axis]
            connect_2(
                self.tiles[left][left_v_mod].conns[left_axis],
                self.tiles[right][right_v_mod].conns[right_axis],
                f"{left}_{left_axis}_{left_v_mod}",
                f"{right}_{right_axis}_{right_v_mod}",
            )
