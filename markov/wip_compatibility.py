
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
