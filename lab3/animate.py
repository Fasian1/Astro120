import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyfits as pf
from glob import glob

fig = plt.figure()

globbed = glob('/home/data/Planet_Transit/HD209458/HD209458_data/*')
globbed.sort()
globbed = globbed[:300]
data = []
for filepath in globbed:
    load = pf.open(filepath)
    datatoadd = load[0].data
    data.append(datatoadd)

def f(i):
    global data
    return data[i]
path = "/home/data/Planet_Transit/HD209458/HD209458_data/0004.fts"
i=4
im = plt.imshow(f(i), cmap='gray_r', animated=True)

def updatefig(i):
    im.set_array(f(i+4))
    return im,

ani = animation.FuncAnimation(fig, updatefig,frames=100)
plt.show()



