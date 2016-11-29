import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyfits as pf
from glob import glob
import sys


fig = plt.figure()
ax = plt.gca()
time_text = ax.text(5, 5,'')

path = sys.argv[1]
globbed = glob(path)
globbed.sort()
#
# globbed = globbed[505:510]
data = []
i = 0

def fits_data(path):
    hdu_list = pf.open(path)  # loads the fits object
    image_data = hdu_list[0].data
    return image_data


for filepath in globbed:
    # load = pf.open(filepath)
    # datatoadd = load[0].data
    # load.close()
    datatoadd = fits_data(filepath)
    data.append(datatoadd)
    print(i)
    i += 1


def f(i):
    global data
    return data[i]


i = 0
im = plt.imshow(f(i), origin='upper', vmin=0, vmax=1000, cmap='gray_r', animated=True)


def updatefig(i):
    im.set_array(f(i))
    time_text.set_text(str(i))
    print(globbed[i])
    return im,


# animating it frame by frame. update fig updates the plot for however many frames.
# interval is the number of miliseconds in between each update funciton call.
ani = animation.FuncAnimation(fig, updatefig, frames=len(globbed), interval=1)
plt.show()
