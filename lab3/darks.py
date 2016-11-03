# -*- coding: utf-8 -*-
import sys
import numpy as np
from glob import glob
import pyfits as pf
import matplotlib.pyplot as plt

arg = sys.argv[1]

plt.rcParams['figure.figsize'] = (20, 20)


def fits_load(path):
    hdu_list = pf.open(path)  # loads the fits object
    image_data = hdu_list[0].data
    header = hdu_list[0].header
    return header, image_data


# glob in the four and five second darks and all the data of the transit data
def darks(arg):

    secs = int(arg)
    data = glob('/home/data/Planet_Transit/HD209458/HD209458_data/*')
    data.sort()

    head, img = fits_load(data[4])
    # change the number to get different frames of data.
    # Frames 1-3 dont work dont use them. our data is from frams 4 to 568.
    # Frames 4-325 are 5 second exp. Frames 326-568 are 4 second.
    fitsList = []
    if secs == 4:
        # making the four second darks
        foursec = glob('/home/data/Planet_Transit/HD209458/HD209458_4s_darks/*')
        foursec.sort()
        for j4 in np.arange(20):
            bs4, a4 = fits_load(foursec[j4])
            fitsList.append(a4)
    else:
        # making the five second darks
        fivesec = glob('/home/data/Planet_Transit/HD209458/HD209458_5s_darks/*')
        fivesec.sort()
        for j5 in np.arange(20):
            bs5, a5 = fits_load(fivesec[j5])
            fitsList.append(a5)

    avg_dark = fitsList[0]
    for i in range(19):
        avg_dark = avg_dark + fitsList[i+1]
    avg_dark = avg_dark/20.0
    # without=img-avg_dark
    print(avg_dark)
    hdu = pf.PrimaryHDU(avg_dark)
    hdulist = pf.HDUList([hdu])
    if(secs == 4):
        hdulist.writeto("4SecDarks.fits")
    else:
        hdulist.writeto("5SecDarks.fits")

    return avg_dark


# so without is the imgage dark subtracted make sure you are subtracting the right dark
# plt.imshow(without, origin='lower', cmap='gray_r',vmin=0, vmax=1000) #ploting the dark sub data
# plt.show()

darks(arg)
