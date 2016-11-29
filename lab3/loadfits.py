import numpy as np
import sys
import matplotlib.pyplot as plt
import pyfits as pf
from glob import glob

arg = sys.argv[1]

def fits_load(path):
    hdu_list = pf.open(path)  # loads the fits object
    image_data = hdu_list[0].data
    return image_data


# def loadfits(path):
    # globbed = glob(path)
    # for file in globbed:
    #     # hdu_list = pf.open(file) #loads the fits object
    #     # img = hdu_list[0].data
    #     # header = hdu_list[0].header
    #     cmap = plt.cm.get_cmap('jet')
    #     img = fits_load(file)
    #
    #     flat_list = pf.open("V_band.fits")
    #     flatfield = flat_list[0].data
    #
    #     plt.imshow(flatfield, origin='lower', cmap='jet')
    #     bar = plt.colorbar()
    #     bar.ax.tick_params(labelsize=20)
    #
    #     bar.set_label("Response Map Values", fontsize=24)
    #     plt.show()

########################################################################################
def loadfits(path):
    globbed = glob(path)
    for file in globbed:
        # hdu_list = pf.open(file) #loads the fits object
        # img = hdu_list[0].data
        # header = hdu_list[0].header
        # cmap = plt.cm.get_cmap('jet')
        img = fits_load(file)

        # Frames 1-3 dont work dont use them. our data is from frams 4 to 568.
        # Frames 4-325 are 5 second exp. Frames 326-568 are 4 second.
        # if(int(file[-8:-5]) < 325):
        #     darks = "4SecDarks.fits"
        #
        # else:
        #     darks = "5SecDarks.fits"

        # dark_list = pf.open(darks)
        # dark_data = dark_list[0].data
        # flat_list = pf.open("V_band.fits")
        # flatfield = flat_list[0].data
        # img = (img - dark_data)/flatfield

        plt.imshow(img, origin='lower', cmap="gray_r", vmin=0, vmax=1000)
        # plt.imshow(img, cmap=cmap)
        bar = plt.colorbar()
        bar.set_label("Response Map Values", fontsize=16)
        plt.show()
#############################################################################################hijacking this for saving corrected HD189 files
# def loadfits(path):
#     globbed = glob(path)
#     newName = ".fits"
#     count = 0
#     for file in globbed:
#         # hdu_list = pf.open(file) #loads the fits object
#         # img = hdu_list[0].data
#         # header = hdu_list[0].header
#         # cmap = plt.cm.get_cmap('jet')
#         img = fits_load(file)
#
#         # Frames 1-3 dont work dont use them. our data is from frams 4 to 568.
#         # Frames 4-325 are 5 second exp. Frames 326-568 are 4 second.
#         darks = 'hd189darks.fits'
#         darkslist = fits_load(darks)
#         flat_list = fits_load("V_band.fits")
#         img = (img - darkslist)/flat_list
#
#         hdu = pf.PrimaryHDU(img)
#         hdulist = pf.HDUList([hdu])
#         newName = str(count)
#         for i in range(4-len(str(count))):
#             newName = "0"+newName
#         newName = newName + ".fits"
#         print(newName)
#         hdulist.writeto(newName)
#         count += 1
#         # plt.imshow(img, origin='upper', cmap="gray_r", vmin=0, vmax=1000)
#         # plt.show()



loadfits(arg)
