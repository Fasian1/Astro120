import numpy as np
import sys
import matplotlib.pyplot as plt
import pyfits as pf
from glob import glob

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
    copy = np.array([])
    lstRatios = []
    for fits in globbedFits:
        loader = pf.open(fits)
        data = loader[0].data
        header = loader[0].header
        mean = np.mean(data)
        pixelvals = []
        copy = np.add(copy, fits)
#        for i in range(len(data)):
#            for j in range(len(data[0])):
#                pixelvals.append(data[i,j])
#        meanval = sum(pixelvals)/(float)(len(pixelvals))
        ratio = data/mean
        lstRatios.append(ratio)
    

flatfield(arg)
