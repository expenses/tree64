from markov.wfc import *
import unittest


def mk_conns(updates):
    conns = {
        "x": Tags(),
        "negx": Tags(),
        "z": Tags(),
        "negz": Tags(),
        "y": Tags(),
        "negy": Tags(),
    }
    conns.update(updates)
    return conns


class Tester(unittest.TestCase):
    def test_t_i(self):
        c = MkJrConnector()
        c.add("T", "T_3d")
        c.add("I", "I_3d")

        c.connect("I", 1, "T", 1)
        c.tiles["T"][0].apply_symmetry()
        self.assertEqual(
            c.tiles["T"][0].conns,
            mk_conns(
                {
                    "negy": Tags(incoming="T_y_0", outgoing="I_negy_0"),
                    "y": Tags(incoming="T_y_0", outgoing="I_negy_0"),
                }
            ),
            # "{'x': (), 'y': (in: {'T_y_0'} out: {'I_negy_0'}), 'negx': (), 'negy': (in: {'T_y_0'} out: {'I_negy_0'}), 'z': (), 'negz': ()}"
        )

    def test_i_i(self):
        c = MkJrConnector()
        c.add("I", "I_3d")

        c.connect("I", 1, "I", 1)
        c.tiles["I"][0].apply_symmetry()
        self.assertEqual(
            c.tiles["I"][0].conns,
            mk_conns(
                {
                    "negy": Tags("I_y_0", "I_negy_0"),
                    "y": Tags("I_y_0", "I_negy_0"),
                }
            ),
        )

    def test_i_i_flipped(self):
        c = MkJrConnector()
        c.add("I", "I_3d")

        c.connect("I", 1, "I", 0, on_z=True)
        self.assertEqual(
            c.tiles["I"][0].conns,
            mk_conns(
                {
                    "z": Tags(incoming="I_z_0", outgoing="I_negz_1"),
                    "negz": Tags(incoming="I_negz_0", outgoing="I_z_1"),
                }
            ),
        )

    """def test_b(self):
        c = MkJrConnector()
        c.add("I", "I_3d")

        c.connect("I", 0, "I", 1)
        self.assertEqual(
            str(c.tiles["I"][0].conns),
            "{'x': (in: {'I_x_0'} out: {'I_negx_0'}), 'y': (), 'negx': (in: {'I_negx_0'} out: {'I_x_0'}), 'negy': (), 'z': (), 'negz': ()}"
        )"""


unittest.main()
