# coding: utf-8
from markov import *
get_ipython().run_line_magic('cd', 'examples/')
from markov import *
axises = ["X","Y","Z"]
air = wfc.add(1.0)
wfc = Wfc((50,50,50))
air = wfc.add(1.0)
solid = wfc.add(1.0)
wfc.connect(solid,air,["x","y","z","negx","negy"])
wfc.connect(solid,solid,axises)
wfc.connect(air,air,axises)
wfc.setup_state()
