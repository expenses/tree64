# coding: utf-8
from pxr import Usd
import numpy as np
from markov import *
from voxypy.models import *

vox = Entity().from_file(filename="./torii2.vox")
palette = vox.get_palette()
palette = Palette([(r, g, b) for (r, g, b, a) in palette])

tori = np.array(vox.get_dense())
tori = np.pad(tori, 4)

tori_blank = tori.copy()
tori_blank[:, :, :] = 0

arr = np.zeros((128, 128, 128), dtype=np.uint8)

rep(arr, Pattern((tori_blank, tori), flips=NO_FLIPS, shuffles=[[2, 1, 0], [1, 2, 0]]))
# arr2 = np.minimum(arr, 1)
stage = Usd.Stage.CreateNew("gates.usdc")
stage.SetMetadata("upAxis", "Z")
add_to_usd_stage("/arr", stage, arr, palette=palette)
stage.Save()
