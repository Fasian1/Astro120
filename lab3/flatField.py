#Define a function to handle loading of fits files given a path:
import numpy as np
import sys
import matplotlib.pyplot as plt
import pyfits as pf #(or from astropy.io import fits as pf)
from glob import glob
#%matplotlib inline
#plt.rcParams['figure.figsize'] = (20,20)
arg = sys.argv[1]
def flatfield(arg):
    directory = ''

    if(arg == "r"):
        directory = "/home/data/Planet_Transit/flatfields/R_band/*"
    elif(arg == "v"):
        directory = "/home/data/Planet_Transit/flatfields/V_band/*"
    else:
        print("no.")
        return
    globbedFits = glob(directory)
    flatfieldarray = np.array([])
    for fits in globbedFits:
        loader = pf.open(fits)
        data = loader[0].data
        header = loader[0].header
        mean = np.mean(data)
        pixelvals = []
        for i in range(data):
            for j in range(data[0]):
                pixelvals.append(data[i,j]
        meanval = sum(pixelvals)/(float)(len(pixelvals))
        slope = meanval/mean
        print(slope)


#def fits_load(path):
#    hdu_list = pf.open(path) #loads the fits object
#    image_data = hdu_list[0].data
#    header = hdu_list[0].header
#    return header, image_data
#Lets load a sample fits file
#head, img = fits_load('2011_09_13.fits')
#print 'PIXSCALE: ', head['PIXSCALE'] #example of querying the header for information
#print img #show we have a 2D array of values. 
# How to load a large number of files at once
#incand_files = [np.genfromtxt(f, skiprows=17, skip_footer=1, usecols=(0,1)) for f in fnames]
#print len(incand_files)

flatfield(arg)
