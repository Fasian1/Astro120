# -*- coding: utf-8 -*-
# This code gives a list of time the data was collected at
import numpy as np
from glob import glob
import pyfits as pf
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20, 20)


def fits_load(path):
    hdu_list = pf.open(path)  # loads the fits object
    image_data = hdu_list[0].data
    header = hdu_list[0].header
    return header, image_data


data = glob('/home/data/Planet_Transit/HD209458/HD209458_data/*')
data.sort()

###########################################################################################

datatime = np.arange(569).tolist()

for bur in np.arange(569)[4:569]:
    head, doesnotmatteratall = fits_load(data[bur])
    datatime[bur] = head[17]

datatimesec = np.arange(569).tolist()
for j in np.arange(569)[4:569]:
    Hr = float(datatime[j][1])
    Mn = float(datatime[j][3:5])
    Is = float(datatime[j][6:])

    secaftermidnight = 3600 * Hr + 60 * Mn + Is
    datatimesec[j] = secaftermidnight

timefromstart = np.arange(569).tolist()
for s in np.arange(569)[4:569]:
    timefromstart[s] = datatimesec[s] - datatimesec[4]
