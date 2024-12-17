from markov import Tileset, load_mkjr_vox, map_3d
import os
import numpy as np
import types

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
        # return "{}" if len(self.outgoing) == 0 else str(self.outgoing)
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


# By default python does a bitwise not on signed values which is not what we want.
def bitwise_not(value):
    return ~np.uint64(value)


def apply_symmetry(tags, symmetry):
    if type(tags) is not dict:
        tags = {"x": tags, "y": tags, "z": tags}

    for dir in ["x", "y", "negx", "negy", "z", "negz"]:
        if not dir in tags:
            tags[dir] = Tags()
        if type(tags[dir]) is str:
            tags[dir] = Tags(tags[dir])

    if symmetry.startswith("X"):
        for dir in ["x", "y", "negx", "negy"]:
            for other in ["x", "y", "negx", "negy"]:
                tags[dir].merge(tags[other])
    if symmetry.startswith("I"):
        tags["x"].merge(tags["negx"])
        tags["y"].merge(tags["negy"])
    if symmetry.startswith("L"):
        tags["x"].merge(tags["y"])
        tags["negx"].merge(tags["negy"])
    if symmetry.startswith("T"):
        tags["y"].merge(tags["negy"])

    if symmetry.endswith("_3d"):
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
            resolved_tags = {}
            for dir, tag in tags.items():
                if isinstance(tag, types.FunctionType):
                    resolved_tags[dir] = tag(i)
                else:
                    resolved_tags[dir] = tag

            res.append(self.add(prob, resolved_tags))
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


def rot_z_symmetrical(voxels, rots):
    return (voxels == np.rot90(voxels, axes=(1, 2), k=rots)).all()


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
