import numpy as np
import sys
import matplotlib.pyplot as plt
import pyfits as pf
from glob import glob

arg = sys.argv[1]

def fits_load(path):
    hdu_list = pf.open(path) #loads the fits object
    img = hdu_list[0].data
    header = hdu_list[0].header
    cmap = plt.cm.get_cmap('jet')
    plt.imshow(img, origin='lower', cmap=cmap)
    bar = plt.colorbar()
    bar.set_label("Response Map Values", fontsize = 16)
    plt.show()

fits_load(arg)
