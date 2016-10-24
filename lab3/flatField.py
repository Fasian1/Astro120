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
    slopes = np.zeros((1336,2004))
    lstRatios = []
    pixelx = []
    pixely = []
    loader1 = pf.open(globbedFits[0])
    data1 = loader1[0].data
    for i in range(len(data1)):
        for j in range(len(data1[0])):
            for fits in globbedFits:
                loader = pf.open(fits)
                data = loader[0].data
                header = loader[0].header
                mean = np.mean(data)
                pixely.append(data[i][j])
                pixelx.append(mean)
            slope = np.polyfit(pixelx, pixely, 1)
            #slopes[i][j] = slope[0]
        print(slope)
    print(slopes)

flatfield(arg)
