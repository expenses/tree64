from markov import Tileset, load_mkjr_vox, map_3d
import os
import numpy as np

FLIPPED = {"x": "negx", "y": "negy", "negx": "x", "negy": "y", "z": "negz", "negz": "z"}


def rot_z_times(d, times):
    for _ in range(times):
        d = rot_z(d)
    return d


def rot_z(d):
    if type(d) is dict:
        m = d
        n = {}
        for d, v in m.items():
            n[rot_z(d)] = v
        return n

    if d == "x":
        return "negy"
    if d == "negy":
        return "negx"
    if d == "negx":
        return "y"
    if d == "y":
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

    def __eq__(self, other):
        return self.incoming == other.incoming and self.outgoing == other.outgoing


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

    if len(split) == 1:
        return split[0], 0

    if split[0] in tiles:
        return split[0], int(split[1])
    else:
        zs = split[0].count("z")
        xs = split[0].count("x")
        ys = split[0].count("y")
        return split[1], zs


def rot_z_symmetrical(voxels, rots):
    return (voxels == np.rot90(voxels, axes=(1, 2), k=rots)).all()


class XmlTileset:
    def __init__(self, filename):
        import xmltodict

        connector = MkJrConnector()

        parsed = xmltodict.parse(open(filename).read(), force_list=["tile", "neighbor"])

        if "set" in parsed:
            parsed = parsed["set"]
        elif "tileset" in parsed:
            parsed = parsed["tileset"]

        for tile in parsed["tiles"]["tile"]:
            name = tile["@name"]

            model = load_mkjr_vox(f"{os.path.splitext(filename)[0]}/{name}.vox")

            symmetry = ""
            if rot_z_symmetrical(model, 1):
                symmetry = "X_3d"
            elif rot_z_symmetrical(model, 2):
                symmetry = "I_3d"
            elif (model == np.flip(model, axis=(0, 1))).all():
                symmetry = "T_3d"

            connector.add(
                name,
                symmetry,
                weight=float(tile["@weight"]) if "@weight" in tile else 1.0,
                model=model,
            )

        for neighbor in parsed["neighbors"]["neighbor"]:
            top_to_bottom = False

            if "@top" in neighbor and "@bottom" in neighbor:
                top_to_bottom = True
                left, left_v = split_xml_tile(
                    neighbor["@top"], connector.tiles, top_to_bottom=top_to_bottom
                )
                right, right_v = split_xml_tile(
                    neighbor["@bottom"], connector.tiles, top_to_bottom=top_to_bottom
                )
            else:
                left, left_v = split_xml_tile(neighbor["@left"], connector.tiles)
                right, right_v = split_xml_tile(neighbor["@right"], connector.tiles)

            connector.connect(left, left_v, right, right_v, on_z=top_to_bottom)

        self.tileset = TaggingTileset()
        self.tile_ids = connector.add_and_get_tile_ids(self.tileset)
        self.tiles = connector.tiles

    def create_wfc(self, dims):
        return self.tileset.tileset.create_wfc(dims)

    def create_array_from_models(self, tile_dims):
        array = np.zeros(
            (self.tileset.tileset.num_tiles(),) + tile_dims, dtype=np.uint8
        )
        for name, ids in self.tile_ids.items():
            model = self.tiles[name][0].model
            add_model_variants_to_model_array(model, array, ids)
        return array

    def run_wfc_and_map(self, dims, model_padding=(0, 0, 0)):
        model_shape = next(iter(self.tiles.values()))[0].model.shape
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
        rotated_model = np.rot90(model, axes=(1, 2), k=rot)
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

    def apply_symmetry_to_variant(self, variant_num):
        self.conns = rot_z_times(self.conns, 4 - variant_num)
        self.apply_symmetry()
        self.conns = rot_z_times(self.conns, variant_num)


VARIANTS_FOR_SYMMETRY = {"T_3d": 4, "T": 4, "X_3d": 1, "X": 1, "I_3d": 2, "I": 2, "": 4}


class MkJrConnector:
    def __init__(self):
        self.tiles = {}

    def add(self, name, symmetry, model=None, weight=1.0):
        self.tiles[name] = [
            MutableTile(
                symmetry=symmetry,
                weight=weight / VARIANTS_FOR_SYMMETRY[symmetry],
                model=model,
            )
            for _ in range(VARIANTS_FOR_SYMMETRY[symmetry])
        ]

    def connect(self, left, left_v, right, right_v, on_z=False):
        axes = ["negz"] if on_z else ["x", "negy", "negx", "y"]

        if len(self.tiles[right]) > len(self.tiles[left]):
            left, right = right, left
            left_v, right_v = right_v, left_v
            axes = ["negz"] if on_z else ["negx", "y", "x", "negy"]

        diff = left_v - right_v

        conns = [
            (
                axes[(left_v + i) % len(axes)],
                (i) % len(self.tiles[left]),
                (i + diff) % len(self.tiles[right]),
            )
            for i in range(len(self.tiles[left]))
        ]

        for left_axis, left_v, right_v in conns:
            right_axis = FLIPPED[left_axis]
            connect_2(
                self.tiles[left][left_v].conns[left_axis],
                self.tiles[right][right_v].conns[right_axis],
                f"{left}_{left_axis}_{left_v}",
                f"{right}_{right_axis}_{right_v}",
            )

    def add_and_get_tile_ids(self, tileset):
        tile_ids = {}

        for name, variants in self.tiles.items():
            for i, variant in enumerate(variants):
                variant.apply_symmetry_to_variant(i)

            if True:
                print(name, len(variants))
                for variant in variants:
                    for dir in sorted(list(variant.conns.keys()))[::-1]:
                        print("    ", dir, variant.conns[dir])
                    print()


            tile_ids[name] = [
                tileset.add(variant.weight, variant.conns, symmetry="")
                for variant in variants
            ]
        return tile_ids
