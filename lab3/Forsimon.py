# -*- coding: utf-8 -*-
import sys
import numpy as np
from glob import glob
import pyfits as pf
import matplotlib.pyplot as plt

arg = sys.arvg[1]
secs = int(arg)

plt.rcParams['figure.figsize'] = (20,20)

def fits_load(path):
    hdu_list = pf.open(path) #loads the fits object
    image_data = hdu_list[0].data
    header = hdu_list[0].header
    return header, image_data

#glob in the four and five second darks and all the data of the transit data

data=glob('/home/data/Planet_Transit/HD209458/HD209458_data/*')
data.sort()

head, img = fits_load(data[4]). #change the number to get different frames of data. Frames 1-3 dont work dont use them. our data is from frams 4 to 568. Frames 4-325 are 5 second exp. Frames 326-568 are 4 second.
avg_dark = img

if(secs == 4):
	#making the four second darks
	foursec=glob('/home/data/Planet_Transit/HD209458/HD209458_4s_darks/*')
	foursec.sort()
	joke4=[]
	for j4 in np.arange(20):
		bs4, a4=fits_load(foursec[j4])
		joke4.append(a4)
else:
	#making the five second darks
	fivesec=glob('/home/data/Planet_Transit/HD209458/HD209458_5s_darks/*')
	fivesec.sort()
	joke5=[]
	for j5 in np.arange(20):
		bs5, a5=fits_load(fivesec[j5])
		joke5.append(a5)

#ass4 is the four second dark and ass5 is the five second dark
avg_dark=np.mean(avg_dark)
#without=img-avg_dark
return avg_dark

#so without is the imgage dark subtracted make sure you are subtracting the right dark
# plt.imshow(without, origin='lower', cmap='gray_r',vmin=0, vmax=1000) #ploting the dark sub data
# plt.show()

