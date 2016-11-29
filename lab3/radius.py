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


def transit():
    data = glob('/home/sko/translatedFits/*')
    data.sort()
    datalen = len(data)

    sc = np.array([516, 1154])  # np.array([  516.28712204,  1154.4116618 ])

    rs1 = np.array([364, 1237])
    rs2 = np.array([726, 1035])
    rs3 = np.array([392, 729])

    # sc = np.array([609, 545])
    # rs1 = np.array([1454, 220])

    # find B

    # Bset = [0,1,2,3]

    Bset2 = []
    sigmabset = []

    yp = sc[1]
    xp = sc[0]

    yp1 = rs1[1]
    yp2 = rs2[1]
    yp3 = rs3[1]
    xp1 = rs1[0]
    xp2 = rs2[0]
    xp3 = rs3[0]

    for f in np.arange(datalen):
        head, img = fits_load(data[f])
        pointsinan = []
        for vary in list(np.arange(yp - 60, yp - 50)) and list(np.arange(yp + 50, yp + 60)):
            for varx in list(np.arange(xp - 60, xp + 60)):
                pointsinan.append(img[vary][varx])
        for vary in list(np.arange(yp - 50, yp + 50)):
            for varx in list(np.arange(xp - 60, xp - 50)) and list(np.arange(xp + 50, xp + 60)):
                pointsinan.append(img[vary][varx])
        sigmab = np.std(pointsinan)
        sigmabset.append(sigmab)
        Bset2.append(np.mean(pointsinan))

    q = .37
    G = 1.7
    C = q / G
    # apature calc
    # r = 1
    # stnset = []
    # rset = []
    # while r <= 50:
    #     rset.append(r)
    #     setoffstar = []
    #     N = 0
    #     head, img = fits_load(data[0])
    #     for vary in np.arange(yp - r, yp + r):
    #         for varx in np.arange(xp - r, xp + r):
    #             f = img[vary][varx]
    #             fstar = f - Bset2[0]
    #             setoffstar.append(fstar)
    #             N = N + 1
    #     signal = sum(setoffstar)
    #     noisesq = C * signal + N * sigmabset[0] ** 2
    #     noise = noisesq ** .5
    #     stn = signal / noise
    #     stnset.append(stn)
    #     r = r + 1
    #
    # r1 = 1
    # stnset1 = []
    # rset1 = []
    # while r1 <= 50:
    #     rset1.append(r1)
    #     setoffstar1 = []
    #     N1 = 0
    #     head1, img1 = fits_load(data[0])
    #     for vary1 in np.arange(yp1 - r1, yp1 + r1):
    #         for varx1 in np.arange(xp1 - r1, xp1 + r1):
    #             f1 = img1[vary1][varx1]
    #             fstar1 = f1 - Bset2[0]
    #             setoffstar1.append(fstar1)
    #             N1 = N1 + 1
    #     signal1 = sum(setoffstar1)
    #     noisesq1 = C * signal1 + N1 * sigmabset[0] ** 2
    #     noise1 = noisesq1 ** .5
    #     stn1 = signal1 / noise1
    #     stnset1.append(stn1)
    #     r1 = r1 + 1

    # r = 1

    # stnset2=[]
    # rset=[]
    # while r<=50:
    #	rset.append(r)
    #	setoffstar = []
    #	N = 0
    #	head, img = fits_load(data[j])
    #	for vary in np.arange(yp2-r,yp2+r):
    #		for varx in np.arange(xp2-r,xp2+r):
    #			f = img[vary][varx]
    #			fstar=f-Bset2[j]
    #			setoffstar.append(fstar)
    #			N = N+1
    #	signal = sum(setoffstar)
    #	C = q/G
    #	noisesq = C*signal+N*sigmabset[j]**2
    #	noise = noisesq**.5
    #	stn = signal/noise
    #	stnset2.append(stn)
    #	r=r+1

    # r = 1

    # stnset3=[]
    # rset=[]
    # while r<=50:
    #	rset.append(r)
    #	setoffstar = []
    #	N = 0
    #	head, img = fits_load(data[j])
    #	for vary in np.arange(yp3-r,yp3+r):
    #		for varx in np.arange(xp3-r,xp3+r):
    #			f = img[vary][varx]
    #			fstar=f-Bset2[j]
    #			setoffstar.append(fstar)
    #			N = N+1
    #	signal = sum(setoffstar)
    #	C = q/G
    #	noisesq = C*signal+N*sigmabset[j]**2
    #	noise = noisesq**.5
    #	stn = signal/noise
    #	stnset3.append(stn)
    #	r=r+1

    # plt.figure()
    # plt.plot(rset, stnset, 'bo')
    # plt.plot(rset1, stnset1, 'go')
    # # plt.plot(rset,stnset2,'ro')
    # # plt.plot(rset,stnset3,'yo')
    #
    # plt.xlabel('Length of r in pixels')
    # plt.ylabel('Signal to Noise Ratio')
    # plt.show()

    r = 5
    # r = 6
    a = 4
    zach = []
    for j in np.arange(datalen):
        setoffstarss = []
        N = 0
        head, img = fits_load(data[j])
        for vary in np.arange(yp - r, yp + r):
            for varx in np.arange(xp - r, xp + r):
                f = img[vary][varx]
                fstar = f - Bset2[j]
                setoffstarss.append(fstar)
                N = N + 1
        signal = sum(setoffstarss)
        C = q / G
        noisesq = C * signal + N * sigmabset[j] ** 2
        noise = noisesq ** .5

        setoffstarr1 = []
        N = 0
        head, img = fits_load(data[j])
        for vary in np.arange(yp1 - r, yp1 + r):
            for varx in np.arange(xp1 - r, xp1 + r):
                f = img[vary][varx]
                fstar = f - Bset2[j]
                setoffstarr1.append(fstar)
                N = N + 1
        signal1 = sum(setoffstarr1)
        C = q / G
        noisesq1 = C * signal1 + N * sigmabset[j] ** 2
        noise1 = noisesq1 ** .5

        setoffstarr2 = []
        N = 0
        head, img = fits_load(data[j])
        for vary in np.arange(yp2 - r, yp2 + r):
            for varx in np.arange(xp2 - r, xp2 + r):
                f = img[vary][varx]
                fstar = f - Bset2[j]
                setoffstarr2.append(fstar)
                N = N + 1
        signal2 = sum(setoffstarr2)
        C = q / G
        noisesq2 = C * signal2 + N * sigmabset[j] ** 2
        noise2 = noisesq2 ** .5

        setoffstarr3 = []
        N = 0
        head, img = fits_load(data[j])
        for vary in np.arange(yp3 - r, yp3 + r):
            for varx in np.arange(xp3 - r, xp3 + r):
                f = img[vary][varx]
                fstar = f - Bset2[j]
                setoffstarr3.append(fstar)
                N = N + 1
        signal3 = sum(setoffstarr3)
        C = q / G
        noisesq3 = C * signal3 + N * sigmabset[j] ** 2
        noise3 = noisesq3 ** .5

        sa = np.array((signal, noise))
        sa1 = np.array((signal1, noise1))
        sa2 = np.array((signal2, noise2))
        sa3 = np.array((signal3, noise3))
        jmoney = np.array((sa, sa1, sa2, sa3))
        # jmoney = np.array((sa, sa1))
        zach.append(jmoney)
        # a = a + 1
        # print a

    data2 = glob('/home/data/Planet_Transit/HD209458/HD209458_data/*')
    # data2 = glob('/home/data/Planet_Transit/HD189733/SI/*')
    data2.sort()

    ###########################################################################################
    # get seconds for x axis later
    datatime = np.arange(len(data2)).tolist()

    for bur in np.arange(len(data2)):
        head, doesnotmatteratall = fits_load(data2[bur])
        datatime[bur] = head[17]

    datatimesec = np.arange(datalen).tolist()
    for j in np.arange(datalen):
        Hr = float(datatime[j][1])

        Mn = float(datatime[j][3:5])

        Is = float(datatime[j][6:])

        secaftermidnight = 3600 * Hr + 60 * Mn + Is

        datatimesec[j] = secaftermidnight

    timefromstart = np.arange(datalen).tolist()
    for s in np.arange(datalen):
        timefromstart[s] = datatimesec[s] - datatimesec[0]
    ########################################################################################


    # good = glob('/home/sko/translatedFits/*')

    # lgood = []
    #
    # for i in np.arange(len(good)):
    #
    #     if len(good[i]) == 37:
    #         add = int(good[i][-5:-4])
    #
    #     if len(good[i]) == 38:
    #         add = int(good[i][-6:-4])
    #
    #     if len(good[i]) == 39:
    #         add = int(good[i][-7:-4])
    #
    #     lgood.append(add)
    #     lgood.sort()

    fux1 = []
    errorfux1 = []

    for e in range(len(data)):
        pui = zach[e][0][0] / zach[e][1][0]
        fux1.append(pui)
        ko = pui * ((zach[e][0][1] / zach[e][0][0]) ** 2 + (zach[e][1][1] / zach[e][1][0]) ** 2) ** .5
        errorfux1.append(ko)

    fux2 = []
    errorfux2 = []
    for e in range(len(data)):
        pui = zach[e][0][0] / zach[e][2][0]
        fux2.append(pui)
        ko = pui * ((zach[e][0][1] / zach[e][0][0]) ** 2 + (zach[e][2][1] / zach[e][2][0]) ** 2) ** .5
        errorfux2.append(ko)

    fux3 = []
    errorfux3 = []
    for e in range(len(data)):
        pui = zach[e][0][0] / zach[e][3][0]
        fux3.append(pui)
        ko = pui * ((zach[e][0][1] / zach[e][0][0]) ** 2 + (zach[e][3][1] / zach[e][3][0]) ** 2) ** .5
        errorfux3.append(ko)


    avgfux1 = sum(fux1) / len(fux1)
    avgfux2 = sum(fux2) / len(fux2)
    avgfux3 = sum(fux3) / len(fux3)

    normalfux1 = fux1 / avgfux1
    normalfux2 = fux2 / avgfux2
    normalfux3 = fux3 / avgfux3

    normalerrorfux1 = errorfux1 / avgfux1
    normalerrorfux2 = errorfux2 / avgfux2
    normalerrorfux3 = errorfux3 / avgfux3

    # ted = []
    # for bill in np.arange(len(data)):
    #     ted.append(timefromstart[bill])

    # plt.errorbar(ted, normalfux1, yerr=normalerrorfux1, fmt='.')
    plt.errorbar(timefromstart, normalfux1, yerr=normalerrorfux1, fmt='.')
    plt.show()

    wfux = np.array([])
    werrorfux = np.array([])
    for e in np.arange(len(data)):
        n = [normalfux1[e], normalfux2[e], normalfux3[e]]
        SIGMA = [normalerrorfux1[e], normalerrorfux2[e], normalerrorfux3[e]]
        topset = []
        botset = []
        for f in [0, 1, 2]:
            top = n[f] / (SIGMA[f] ** 2)
            bot = 1. / (SIGMA[f] ** 2)
            topset.append(top)
            botset.append(bot)
        crap = (1. / (SIGMA[0] ** 2)) + (1. / (SIGMA[1] ** 2)) + (1. / (SIGMA[2] ** 2))
        acrap = 1 / (SIGMA[0] * crap)
        bcrap = 1 / (SIGMA[1] * crap)
        ccrap = 1 / (SIGMA[2] * crap)
        werrorfux = np.append(werrorfux, ((acrap ** 2) + (bcrap ** 2) + (ccrap ** 2)) ** .5)
        wfux = np.append(wfux, (sum(topset) / sum(botset)))
#####################################################################################################
        # f = 0
        # n = [normalfux1[e]]
        # SIGMA = [normalerrorfux1[e]]
        # topset = []
        # botset = []
        # top = n[f] / (SIGMA[f] ** 2)
        # bot = 1. / (SIGMA[f] ** 2)
        # topset.append(top)
        # botset.append(bot)
        # crap = 1. / (SIGMA[0] ** 2)
        # acrap = 1. / (SIGMA[0] * crap)
        # werrorfux = np.append(werrorfux, ((acrap ** 2) ** .5))
        # wfux = np.append(wfux, (sum(topset) / sum(botset)))

    plt.errorbar(timefromstart, wfux, yerr=werrorfux, fmt='.')
    plt.errorbar(timefromstart,normalfux1,yerr=normalerrorfux1,fmt='b.')
    plt.errorbar(timefromstart,normalfux2,yerr=normalerrorfux2,fmt='r.')
    plt.errorbar(timefromstart,normalfux3,yerr=normalerrorfux3,fmt='g.')
    # yerr=werrorfux
    # plt.errorbar(ted,normalfux1,yerr=normalerrorfux1,fmt='b.')
    # plt.errorbar(ted,normalfux2,yerr=normalerrorfux2,fmt='r.')
    # plt.errorbar(ted,normalfux3,yerr=normalerrorfux3,fmt='g.')
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.xlabel('Time from first exposure [$Seconds$]', size=25)
    plt.ylabel('Normalized Ratio of Science star over reference star', size=25)
    plt.show()

    # Produce bins for the histogram
    # step = timefromstart[-1]/100
    # bins = np.arange(0, timefromstart[-1], step)
    bins = np.arange(0, len(data), 6)
    # Use the np.where function to find the frequency of each planet size category
    step = timefromstart[-1]/(len(data)/6+1)
    frequency = np.arange(0, timefromstart[-1], step)
    bin_width = (timefromstart[-1] - timefromstart[0])/len(bins)
    # currbin = 0
    # copy = timefromstart[:]
    # count = 1
    # index = 0
    # while(len(copy) > 0):
    #     if(copy[index] < bins[count]):
    #         frequency[count] = np.append(frequency, copy[index])
    #         copy = np.delete(copy, copy[index])
    #         index += 1
    #     else:
    #         count += 1
    # print(frequency)
    # for i in range(len(timefromstart)):
    #     if timefromstart[i] < currbin:
    #         frequency = frequency.append()
    # for i in range(len(bins) - 1):
    #     temp = timefromstart[bins[i]:bins[i+1]]
    #     frequency = np.append(frequency, temp)

    ############################################################
    # frequency = np.append(frequency, timefromstart[bins[len(bins)]-1:len(data)-1])
    # Find the width of each bin
    # Use the bin width to find the center of each bin
    bin_centers = frequency[0:-1] + 0.5 * bin_width


    binSumList = []
    binSqList = []

    for k in range(len(bins)-1):
        binsum = sum(wfux[bins[k]: bins[k+1]])/6
        binsquared = (sum(werrorfux[bins[k]:bins[k+1]]**2)**0.5)/6
        binSumList.append(binsum)
        binSqList.append(binsquared)
    plt.errorbar(bin_centers, binSumList, yerr=binSqList, fmt='.')
    # plt.errorbar(pillx, pilly, yerr=pillye, fmt='.')
    # plt.errorbar(ted,wfux,yerr=werrorfux,fmt='.')
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.xlabel('Time from first exposure [$Seconds$]', size=25)
    plt.ylabel('Binned Average of Normalized Ratio of Science star over reference star', size=25)
    plt.show()


    # plt.errorbar(ted,fux1,yerr=errorfux1,fmt='.')
    # plt.xlabel('Seconds from first observation')
    # plt.ylabel('Signal of SS/Signal of RS1')

    # plt.show()


    # mom=3
    # fux2=[]
    # errorfux2=[]
    # for jo in lgood:
    #	pui=zach[jo][mom][0]
    #	fux2.append(pui)
    #	ko=zach[jo][mom][1]
    #	errorfux2.append(ko)

    # plt.errorbar(ted,fux2,yerr=errorfux2,fmt='.')
    # plt.xlabel('Seconds from first observation')
    # plt.ylabel('Signal of RS3 in ADU')
    # plt.show()


transit()
#
