from markov import *
axises = ["X","Y"]

dim=100

wfc = Wfc((dim,dim,1))

categories = {}

def add(prob,cats, blocklist=set(), block_self=False):
    tile = wfc.add(prob)
    if block_self:
        blocklist.add(tile)
    for cat in cats:
        if not cat in categories:
            categories[cat] =[]
        categories[cat].append((tile,blocklist))
    return tile

def mass_connect(cat_from,cat_to,axises):
    for frm,frm_blocklist in categories[cat_from]:
        for to,to_blocklist in categories[cat_to]:
            if to in frm_blocklist or frm in to_blocklist:
                continue

            wfc.connect(frm,to,axises)

empty = add(.25, ["not_left","not_right","not_up","not_down"])

s_p = 1.0/2.0

#straight = add(1.0, {"x":"line","negx":"line","y":"no","negy":"no"})

straight_h = add(s_p, ["left","right","not_up","not_down"])
straight_v = add(s_p, ["not_left","not_right","up","down"])

e_p =1.0/4.0

edge_ur = add(e_p, ["right", "up", "not_left", "not_down"])
edge_dl = add(e_p, ["not_right", "not_up", "left", "down"])
edge_ul = add(e_p, ["not_right", "up", "left", "not_down"])
edge_dr = add(e_p, ["right", "not_up", "not_left", "down"])

end_p =.01/4.0

end_l = add(end_p, ["left","not_right","not_up","not_down"],block_self=True)
end_r = add(end_p, ["not_left","right","not_up","not_down"],block_self=True)
end_u = add(end_p, ["not_left","not_right","up","not_down"],block_self=True)
end_d = add(end_p, ["not_left","not_right","not_up","down"],block_self=True)

cross = add(2.5, ["left","right","up","down"], block_self=True, blocklist={edge_ur, edge_ul, edge_dl, edge_dr})


tiles = {}
for i in range(cross+1):
    tiles[i]=np.zeros((3,3),dtype=np.uint8)

tiles[straight_h][1]=1
tiles[straight_v][:,1]=1

tiles[cross][1]=1
tiles[cross][:,1]=1

tiles[edge_ur][1,1:]=1
tiles[edge_ur][:2,1]=1

tiles[edge_ul][1,:2]=1
tiles[edge_ul][:1,1]=1

tiles[edge_dr][1,1:]=1
tiles[edge_dr][1:,1]=1

tiles[edge_dl][1,:1]=1
tiles[edge_dl][1:,1]=1

arr =np.zeros((dim*3,dim*3),dtype=np.uint8)

def draw():
    values = wfc.values()[0]
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            val = values[y,x]
            arr[y*3:y*3+3,x*3:x*3+3] = tiles[val]
    return arr

mass_connect("right", "left",["x"])
mass_connect("down", "up",["y"])
mass_connect("not_right", "not_left",["x"])
mass_connect("not_down", "not_up",["y"])

wfc.setup_state()

writer = FfmpegWriter("out.avi", (dim*3,dim*3))

i =0
while True:
    value = wfc.find_lowest_entropy()
    if value is None:
        break
    index, tile = value
    wfc.collapse(index, tile)
    if i % 8 ==0:
        writer.write(draw())
    i += 1

#wfc.collapse_all()
assert wfc.all_collapsed()

save_image("out.png",wfc.values()[0])
