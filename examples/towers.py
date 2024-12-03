from markov import *
import numpy as np

arr = np.zeros((64,64,64),dtype=np.uint8)
arr[0] = 1

start = np.zeros((3,3,3),dtype=np.uint8)
start[0]=1
start_from = start.copy()
start[1,1,1]=1
start[2,1,1]=2

output = UsdWriter("towers.usdc", skip=500)

rep(arr, Markov(
    Pattern('''
        200,
        1*0,
        1*1=
        112,
        **1,
        ***
    ''', flips=[[False,True,False], [True, True,False]], shuffles=Y_IS_Z),
    Pattern("200=112", shuffles=[[2,1,0]]),
    '2=1',
    Pattern((start_from, start), flips=NO_FLIPS, shuffles=NO_SHUFFLES),
),writer=output)

write_usd("bridges.usdc", arr)
