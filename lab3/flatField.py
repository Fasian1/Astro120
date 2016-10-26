import numpy as np
import sys
import matplotlib.pyplot as plt
import pyfits as pf
from glob import glob

arg = sys.argv[1]
def flatfield(arg):
    directory = ''

    if(arg == "r"):
        directory = "/home/data/Planet_Transit/flatfields/R_band/r_flat_*"
    elif(arg == "v"):
        directory = "/home/data/Planet_Transit/flatfields/V_band/Vflat_*"
    elif(arg == "rd"):
        directory = "/home/data/Planet_Transit/flatfields/R_band_darks/r_dark_*"
    elif(arg == "vd"):
        directory = "/home/data/Planet_Transit/flatfields/V_band_darks/Dark1s_*"
    else:
        print("no.")
        return
    globbedFits = glob(directory)
    pixelx = []
    loader1 = pf.open(globbedFits[0])
    data1 = loader1[0].data
    slopes = np.zeros((len(data1),len(data1[0])))

    datacube = np.ndarray(shape=(len(data1), len(data1[0]), 0))
    for fits in globbedFits:
        loader = pf.open(fits)
        data = loader[0].data
        header = loader[0].header
        mean = np.mean(data)
        pixelx.append(mean)
        datacube = np.dstack((datacube, data))
    for i in range(len(data1)):
        for j in range(len(data1[0])):
            slope = np.polyfit(pixelx, datacube[i][j], 1)
            slopes[i][j] = slope[0]
    hdu = pf.PrimaryHDU(slopes)
    hdulist = pf.HDUList([hdu])
    if(arg == "r"):
        hdulist.writeto("R_band.fits")
    elif(arg == "v"):
        hdulist.writeto("V_band.fits")
    elif(arg == "rd"):
        hdulist.writeto("Rdark_band.fits")
    elif(arg == "vd"):
        hdulist.writeto("Vdark_band.fits")
    else:
        hdulist.writeto("flatfield.fits")


flatfield(arg)
