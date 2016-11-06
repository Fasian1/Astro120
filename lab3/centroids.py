# -*- coding: utf-8 -*-
#written by john
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


def centroids():
    data = glob('/home/sko/Cordata/*')
    data.sort()
    # datafix = np.arange(569).tolist()
    #
    # for j in np.arange(569)[0:565]:
    #     if(len(data[j]) == 16):
    #         apple = int(data[j][-6:-5])
    #     if(len(data[j]) == 17):
    #         apple = int(data[j][-7:-5])
    #     if(len(data[j]) == 18):
    #         apple = int(data[j][-8:-5])
    #     datafix[apple] = data[j]


    # brightpoints = [0, 1, 2, 3]
    # for g in np.arange(569)[4:569]:
    #     head, img = fits_load(datafix[g])
    #     y = 0
    #     maxPix = np.argmax(img)
    #     #converting pixel number to axes
    #     x = maxPix % 2004
    #     y = maxPix // 2004
    #     addme = np.array((x, y))
    #     brightpoints.append(addme)
    ######################################################
    # for path2 in data:
    #     hdu_list = pf.open(path2)
    #     img = hdu_list[0].data
    #     median = np.median(img)
    #     img = img - median
    #     for i in range(len(img)):
    #         for j in range(len(img[0])):
    #             if img[i][j] < 0:
    #                 img[i][j] = 0
    #     # hdu_list.writeto("Med"+path2[-9:])
    #     print(img)


    brightpoints = []
    for path in data:
        # hdu_list = pf.open(path)  # loads the fits object
        # image_data = hdu_list[0].data
        head, img = fits_load(path)
        maxPix = np.argmax(img)
        x = maxPix % 2004
        y = maxPix // 2004
        addme = np.array((x,y))
        brightpoints.append(addme)

    allcents = []

    for f in range(len(data)):
        pointshalf = []
        # hdu_list = pf.open(path)  # loads the fits object
        # img = hdu_list[0].data
        head, img = fits_load(data[f])

        yp = brightpoints[f][1]
        xp = brightpoints[f][0]

        for vary in np.arange(yp - 50, yp + 50):
            if vary >= 1336:
                vary = 1335
            for varx in np.arange(xp - 50, xp + 50):
                if varx >= 2004:
                    varx = 2003
                if img[vary][varx] >= img[yp][xp] / 2.:
                    alliwishforisdeath = np.array((varx, vary, img[vary][varx]))
                    pointshalf.append(alliwishforisdeath)
        xsum = []
        ysum = []
        isum = []

        for j in np.arange(len(pointshalf)):
            xsum.append(pointshalf[j][0] * pointshalf[j][2])
            ysum.append(pointshalf[j][1] * pointshalf[j][2])
            isum.append(pointshalf[j][2])
        CX = np.sum(xsum) / np.sum(isum)
        CY = np.sum(ysum) / np.sum(isum)
        cent = np.array((CY, CX))
        allcents.append(cent)
    hdu = pf.PrimaryHDU(allcents)
    hdulist = pf.HDUList([hdu])
    hdulist.writeto("allcents.fits")
    return allcents

    # meme = []
    # for alb in np.arange(569)[4:569]:
    #     jorish = str(allcents[alb][0])
    #     if len(jorish) < 6:
    #         meme.append(alb)
    # print(meme)


centroids()
# This code purpose is to give all the centroids of all frames
# it gives it in the from of a list of 1d arrays with 2 elements the x and y in that order
# elements 0 1 2 and 3 are left unfilled on purpose
# so allcents[4] would give you an array and that array would be [x,y] cordinates of centroids for frame 4
