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


def parse_pattern(string, unions):
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
            array.append(index_for_colour(char))

    if width == None:
        return np.reshape(array, (1, -1)).astype(np.uint8)
    elif height == None:
        return np.reshape(array, (-1, width)).astype(np.uint8)
    else:
        return np.reshape(array, (-1, height, width)).astype(np.uint8)


def flatten(iterator):
    return [item for sublist in iterator for item in sublist]


def parse_node(ty, data, symmetry, unions, steps=None):
    print(ty, data)

    unimplemented = {"convolution", "wfc", "field", "path"}

    if ty in unimplemented or "@temperature" in data or "@file" in data:
        assert False
    if ty == "union":
        unions[data["@symbol"]] = set(c for c in data["@values"])
        print(unions)
        return True

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
    chance = float(data["@p"]) if "@p" in data else 10.0

    node_settings = NodeSettings(count=steps)

    nodes = []

    if "@in" in data and "@out" in data:
        nodes = [
            Pattern(
                (
                    parse_pattern(data["@in"], unions),
                    parse_pattern(data["@out"], unions),
                ),
                flips=flips,
                shuffles=shuffles,
                chance=chance,
                node_settings=node_settings,
            )
        ]

    if "inner" in data:
        nodes += flatten(
            parse_node(ty, data, symmetry, unions) for ty, data in data["inner"]
        )

    if ty == "all":
        return [All(*nodes, settings=node_settings)]
    if ty == "prl":
        return [Prl(*nodes, settings=node_settings)]
    if ty == "markov":
        return [Markov(*nodes, settings=node_settings)]
    if ty == "sequence":
        return [Sequence(*nodes, settings=node_settings)]
    if ty == "one":
        return [One(*nodes, settings=node_settings)]
    if ty == "rule":
        return nodes
    assert False


def interpret(xml, arr):
    d_name, d = parse_xml_to_dict(xml)

    base = d["@values"][0]
    symmetry = "(xy)"

    if "@symmetry" in d:
        symmetry = d["@symmetry"]

    background, foreground = d["@values"][:2]

    arr[:] = index_for_colour(background)

    if "@origin" in d and d["@origin"]:
        put_middle(arr, index_for_colour(foreground))

    unions = {}

    nodes_to_parse = [(d_name, d)]
    if d_name == "sequence" and "inner" in d:
        nodes_to_parse = d["inner"]

    for ty, data in nodes_to_parse:
        nodes = parse_node(ty, data, symmetry, unions)
        if nodes == [] or nodes == None:
            break
        if nodes == True:
            continue
        for node in nodes:
            print(node)
            rep(arr, node)
        save_image("out.png", arr)
