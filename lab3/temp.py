import numpy as np
from glob import glob
import pyfits as pf
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift

plt.rcParams['figure.figsize'] = (20, 20)


def fits_load(path):
    hdu_list = pf.open(path)  # loads the fits object
    image_data = hdu_list[0].data
    header = hdu_list[0].header
    return header, image_data


data = glob('.Cordata/*')
data.sort()

datafix = np.arange(569).tolist()

for j in np.arange(569)[0:565]:
    if len(data[j]) == 29:
        apple = int(data[j][23:24])
    if len(data[j]) == 30:
        apple = int(data[j][23:25])
    if len(data[j]) == 31:
        apple = int(data[j][23:26])

    datafix[apple] = data[j]

brightpoints = [0, 1, 2, 3]
ayy = 3
for g in np.arange(569)[4:569]:
    head, img = fits_load(datafix[g])
    y = 0
    x = np.argmax(img)
    while x > 2004:
        x = x - 2004
        y = y + 1
    addme = np.array((x, y))
    brightpoints.append(addme)
    ayy = ayy + 1
    print ayy

allcents = [0, 1, 2, 3]

for f in np.arange(569)[4:569]:

    pointshalf = []

    head, img = fits_load(datafix[f])

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
    cent = np.array((CX, CY))
    allcents.append(cent)
    print len(allcents)

meme = []

for alb in np.arange(569)[4:569]:
    jorish = str(allcents[alb][0])
    if len(jorish) < 6:
        meme.append(alb)



        # This code purpose is to give all the centroids of all frames
# it gives it in the from of a list of 1d arrays with 2 elements the x and y in that order
# elements 0 1 2 and 3 are left unfilled on purpose
# so allcents[4] would give you an array and that array would be [x,y] cordinates of centroids for frame 4

for i in np.arange(569)[4:569]:
    head, img = fits_load(datafix[i])
    ay = allcents[4][1] - allcents[i][1]
    ax = allcents[4][0] - allcents[i][0]
    shifted_image = shift(img, [ay, ax])
    plt.imshow(shifted_image, origin='lower', cmap='gray_r', vmin=0, vmax=1000)
    name = str(i) + '.png'
    plt.savefig(name)