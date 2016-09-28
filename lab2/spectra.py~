import numpy as np
import sys
import matplotlib.pyplot as plt

filename = sys.argv[1]

def spectra(filename):
    read = np.genfromtxt(filename, skip_header=17, skip_footer=1, usecols=(0,1))
    data = read[:,1]
    dy = data[1:] - data[0:-1]
    absdy = abs(dy)

    mediam = np.median(data)
    meandy = np.mean(absdy)
    meandata = np.mean(data)
    std = np.sqrt(np.sum((absdy - meandy)**2.)/(np.float(absdy.size)-1.))

    #newdy = absdy[:]
    #for change in absdy:
    #    if change > meandy*2:
    #        newdy = np.delete(newdy, change)

    #newmeandy = np.mean(newdy)
    #std = np.sqrt(np.sum((newdy - newmeandy)**2.)/(np.float(newdy.size)-1.))

    

    cutoffdy = meandy + .3*std
    

    prev = 0
    first = 0
    potpeak = {}
    peaks = []
    newpeak = False
    finished = False
    for pix,inten in enumerate(data):
        if pix < 2:
            continue
        if abs(dy[pix-1]) > cutoffdy:
            if dy[pix-1] < 0:
                if len(potpeak) > 0 and newpeak and finished:
                    key = potpeak.keys()
                    value = potpeak.values()
                    peaks.append(key[value.index(max(value))])
                    potpeak = {}
                    finished = False
                    newpeak = True
                else:
                    if newpeak and dy[pix+1] > 0:
                        finished = True
            else:
                potpeak[pix] = inten
                prev = pix
                if not newpeak:
                    newpeak = True
        newpeak = True
        finished = True
    

    print(peaks)
    print(len(peaks))
    plt.plot(data)
    plt.plot(peaks, data[peaks])
    plt.show()
spectra(filename)





