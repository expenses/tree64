import sys
from markov import *

import xml.etree.ElementTree as ET


def xml_to_dict(element):
    node = {}

    for key, value in element.attrib.items():
        node[f"@{key}"] = value

    children = list(element)
    if children:
        node["inner"] = [(child.tag, xml_to_dict(child)) for child in children]
    else:
        # If no children, add the text content
        if element.text and element.text.strip():
            node["text"] = element.text.strip()

    return node


def parse_xml_to_dict(xml_string):
    """
    Parses an XML string into an OrderedDict.
    """
    root = ET.fromstring(xml_string)
    return (root.tag, xml_to_dict(root))


# Convert XML to dictionary
d_name, d = parse_xml_to_dict(open(sys.argv[1]).read())


arr = np.zeros((128, 128), dtype=np.uint8)

base = d["@values"][0]
symmetry = "(xy)"

if "@symmetry" in d:
    symmetry = d["@symmetry"]

background, foreground = d["@values"][:2]

arr[:] = index_for_colour(background)

if "@origin" in d and d["@origin"]:
    put_middle(arr, index_for_colour(foreground))

unions = {}


def parse_pattern(string):
    width = None
    height = None
    array = []

    for union in unions.keys():
        if union in string:
            assert False

    for char in string:
        if char == "/":
            if width == None:
                width = len(array)
        elif char == " ":
            if height == None:
                if width == None:
                    width = len(array)
                height = len(array) // width
        elif char == "*":
            array.append(255)
        else:
            try:
                array.append(int(char))
            except ValueError:
                array.append(PALETTE_CHARS.index(char))

    if width == None:
        return np.reshape(array, (1, -1)).astype(np.uint8)
    elif height == None:
        return np.reshape(array, (-1, width)).astype(np.uint8)
    else:
        return np.reshape(array, (-1, height, width)).astype(np.uint8)


def flatten(iterator):
    return [item for sublist in iterator for item in sublist]


def parse_node(ty, data, symmetry, steps=None, all=False):
    print(ty, data)

    symmetry = data["@symmetry"] if "@symmetry" in data else symmetry

    if symmetry == "(xy)":
        flips = TOGGLE_XY
        shuffles = ROT_AROUND_Z
    elif symmetry == "()":
        flips = NO_FLIPS
        shuffles = NO_SHUFFLES
    elif symmetry == "(x)":
        flips = TOGGLE_X
        shuffles = NO_SHUFFLES
    else:
        assert False

    steps = int(data["@steps"]) if "@steps" in data else steps

    node_settings = NodeSettings(count=steps) if steps else None

    if ty == "all" or ty == "prl" or all:
        if ty == "prl" and "@d" in data:
            assert False
        if ty == "prl" and "inner" in data:
            assert False

        if "inner" in data:
            return flatten(
                parse_node(ty, data, symmetry, steps, all=True)
                for ty, data in data["inner"]
            )

        print(parse_pattern(data["@in"]))
        print(parse_pattern(data["@out"]))
        return [
            Pattern(
                (parse_pattern(data["@in"]), parse_pattern(data["@out"])),
                apply_all=True,
                flips=flips,
                shuffles=shuffles,
                node_settings=node_settings,
            )
        ]
    if ty == "markov":
        nodes = flatten(parse_node(ty, data, symmetry) for ty, data in data["inner"])
        if None in nodes:
            print("bad node")
            return []
        return [Markov(*nodes, settings=node_settings)]
    if ty == "sequence":
        nodes = flatten(parse_node(ty, data, symmetry) for ty, data in data["inner"])
        if None in nodes:
            print("bad node")
            return []
        return [Sequence(*nodes, settings=node_settings)]
    if ty == "one" and "inner" in data:
        nodes = flatten(parse_node(ty, data, symmetry) for ty, data in data["inner"])
        if None in nodes:
            print("bad node")
            return []
        return [One(*nodes, settings=node_settings)]
    if ty == "rule" or ty == "one":
        print(parse_pattern(data["@in"]))
        print(parse_pattern(data["@out"]))
        return [
            Pattern(
                (parse_pattern(data["@in"]), parse_pattern(data["@out"])),
                flips=flips,
                shuffles=shuffles,
                node_settings=node_settings,
            )
        ]
    if ty == "union":
        unions[data["@symbol"]] = set(c for c in data["@values"])
        print(unions)
        return True
    assert False, ty


nodes_to_parse = [(d_name, d)]
if d_name == "sequence" and "inner" in d:
    nodes_to_parse = d["inner"]

for ty, data in nodes_to_parse:
    nodes = parse_node(ty, data, symmetry, all=d_name == "all")
    if nodes == [] or nodes == None:
        break
    if nodes == True:
        continue
    for node in nodes:
        print(node)
        rep(arr, node)
    save_image("out.png", arr)
