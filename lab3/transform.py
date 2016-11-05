import numpy as np
import scipy, sys
from glob import glob
import pyfits as pf
from scipy.ndimage.interpolation import shift


path = sys.argv[1]

def fits_data(path):
    hdu_list = pf.open(path)  # loads the fits object
    image_data = hdu_list[0].data
    return image_data


def transform(path):
    allcents = fits_data('./allcents.fits')
    globbed = glob(path)
    globbed.sort()
    globbed = globbed[1:]
    first = allcents[0]
    allcents = allcents[1:]
    # print(len(globbed))
    # print(len(allcents))
    # print(globbed)
    for i in range(len(globbed)):
        print(i)
        # data = fits_data(globbed[i])
        hdu_list = pf.open(globbed[i])
        data = hdu_list[0].data
        hdu_list.close()
        shiftcoor = first - allcents[i]
        # print(shift)
        # print(data)
        # fourier = np.fft.fft2(data)
        # nextFourier = scipy.ndimage.fourier.fourier_shift(fourier, shift)
        # shiftedFourier = np.fft.ifft2(nextFourier)
        # realOutput = np.real(shiftedFourier)
        # print("shifted:")
        # print(realOutput)

        shifted_image = shift(data, shiftcoor)
        hdu = pf.PrimaryHDU(shifted_image)
        hdulist = pf.HDUList([hdu])
        writeTo = globbed[i][-9:]
        hdulist.writeto(writeTo)

transform(path)