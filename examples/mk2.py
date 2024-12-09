axis=["x","y","negx","negy"]

MIRRORED={
        "x":"y",
        "y":"x",

        "negx":"negy",
        "negy":"negx",
        }

FLIPPED={
        "x":"negx",
        "y":"negy",
        "negx":"x",
        "negy":"y",

        }

VARIANTS={
        "X":1,
        "I":2,
        "L":4,
        "T":4
        "":4
        }

def mirror_diag(dir):
    pass

def rots_for_sym(dir,sym):
    ret = [dir]
    if sym=="X":
        return axis
    if sym == "I":
        ret.append(FLIPPED[dir])
    if sym == "L":
        ret.append(MIRRORED[dir])
    if sym == "T":
        if dir == "x" or dir == "negx":
            ret.append(FLIPPED[dir])
    return ret

class Tile:
    def __init__(self,symmetry="", probability=1.0):
        self.symmetry=symmetry.upper()
        variants = VARIANTS[self.symmetry]
        self.probability=probability / variants

        self.tiles = [0] * variants
    def connect(self,rot,right,right_rot):
        for rot in rots_for_sym(rot,self.symmetry):
          for i,variant in enumerate(self.variants):
              wfc.connect(variant,
        pass

class Tileset:
    def __init__(self):
        self.tiles=[]
    def add(self,tile):
        index = len(self.tiles)
        self.tiles.append(tile)
        return index
    def connect(self,left,*rights,rot=0):
        for right in rights:
            if right is not tuple:
                right = (right,0)
            right,right_rot=right
            self.tiles[left].connect(rot,self.tiles[right],right_rot)

