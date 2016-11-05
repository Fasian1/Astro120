import numpy as np
import sys
import matplotlib.pyplot as plt
import pyfits as pf
from glob import glob

arg = sys.argv[1]

def fits_load(path):
    globbed = glob(path)
    for file in globbed:
        hdu_list = pf.open(file) #loads the fits object
        img = hdu_list[0].data
        header = hdu_list[0].header
        cmap = plt.cm.get_cmap('jet')

        # Frames 1-3 dont work dont use them. our data is from frams 4 to 568.
        # Frames 4-325 are 5 second exp. Frames 326-568 are 4 second.
        if(int(file[-8:-5]) < 325):
            darks = "4SecDarks.fits"

        else:
            darks = "5SecDarks.fits"

        dark_list = pf.open(darks)
        dark_data = dark_list[0].data
        flat_list = pf.open("V_band.fits")
        flatfield = flat_list[0].data
        img = (img - dark_data)/flatfield

        plt.imshow(img, origin='lower', cmap="gray_r", vmin=0, vmax=1000)
        bar = plt.colorbar()
        bar.set_label("Response Map Values", fontsize=16)
        plt.show()

fits_load(arg)
