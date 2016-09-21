import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.misc as sc
from scipy.interpolate import spline
from scipy.optimize import curve_fit

a = sys.argv

def main(a):
	if '1' in a:
		figure1()
	if '2' in a:
		figure2()
	if '3' in a:
		figure3()
	if '4' in a:
		figure4()
	if '5' in a:
		figure5()		
	if '6' in a:
		figure6()
	if '7' in a:
		figure7()				
	if '8' in a:
		figure8()
	if '9' in a:
		figure9()
	if '10' in a:
		figure10()
	if '11' in a:
		figure11(lambda x,y:(1./x)*np.exp(-y/x))
	if '12' in a:
		figure12()				
	if '14' in a:
		figure14(lambda x,y:(y**(x))*np.exp(-y)/(sc.factorial(x)))				


def figure1():
	x = np.loadtxt('pjzsIncrease10,000_160901_1743_40.csv', delimiter=',', dtype='int32')
	t = x[:,1]
	plt.plot(t)
#	plt.title('Figure 1: Time vs Event Number')	
	plt.xlabel('Event Number', fontsize=16)
	plt.ylabel('Clock Tick',fontsize=16)
	plt.show()

def figure2():
	x = np.loadtxt('pjzsIncrease10,000_160901_1743_40.csv', delimiter=',', dtype='int32')
	t = x[:,1]
	dt = t[1:] - t[0:-1]
	plt.plot(dt)
#	plt.title('Figure 2: Interval vs Event Number')
	plt.xlabel('Event Number', fontsize=16)
	plt.ylabel('Interval [Clock Tick]', fontsize=16)
	plt.show()

def figure3():
	x = np.loadtxt('pjzsIncrease10,000_160901_1743_40.csv', delimiter=',', dtype='int32')
	t = x[:,1]
	dt = t[1:] - t[0:-1]
	plt.plot(dt,',')
#	plt.title('Figure 3 Interval vs Event Number (Dots)')
	plt.xlabel('Event Number', fontsize=16)
	plt.ylabel('Interval [Clock Tick]', fontsize=16)
	plt.show()

def figure4():
	x = np.loadtxt('pjzsIncrease10,000_160901_1743_40.csv', delimiter=',', dtype=np.int32)
	t = x[:,1]
	dt = t[1:] - t[0:-1]	
	marr = np.array([])
	steps = np.array([])
	i = np.arange(dt.size)
	nstep = 1000
	for j in i[0::nstep]:              
		m = np.mean(dt[j:j+nstep])   
		marr = np.append(marr,m)
		steps = np.append(steps, j)

	plt.plot(steps,marr,'o')
#	plt.title('Figure 4 Mean Interval for 10 Chunks of 1000 Events')
	plt.xlabel('Start Index', fontsize=16)
	plt.ylabel('Mean Interval [Clock Ticks]', fontsize=16)
	plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
	plt.show()


def figure5():
	x = np.loadtxt('pjzsIncrease10,000_160901_1743_40.csv', delimiter=',', dtype=np.int32)
	t = x[:,1]
	dt = t[1:] - t[0:-1]	
	marr = np.array([])
	steps = np.array([])
	i = np.arange(dt.size)
	nstep = 100
	for j in i[0::nstep]:              
		m = np.mean(dt[0:j+nstep])   
		marr = np.append(marr,m)
		steps = np.append(steps, j)
	plt.plot(steps,marr,'o')
#	plt.title('Figure 5 Mean Interval with increasing fractions of ')
	plt.xlabel('Number of Intervals Averaged', fontsize=16)
	plt.ylabel('Mean Interval [Clock Ticks]', fontsize=16)
	plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
	plt.show()

def figure6():
	x = np.loadtxt('pjzsIncrease10,000_160901_1743_40.csv', delimiter=',', dtype=np.int32)
	t = x[:,1]
	dt = t[1:] - t[0:-1]	
	marr = np.array([])
	steps = np.array([])
	i = np.arange(dt.size)
	nstep = 100
	for j in i[0::nstep]:              
		m = np.mean(dt[j:j+nstep])   
		marr = np.append(marr,m)
		steps = np.append(steps, j)

	plt.plot(steps,marr,'o')
#	plt.title('Figure 6 Mean Interval for Chunks of 100 Events')
	plt.xlabel('Start Index', fontsize=16)
	plt.ylabel('Mean Interval [Clock Ticks]', fontsize=16)
	plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
	plt.show()

def figure7():
	x = np.loadtxt('pjzsIncrease10,000_160901_1743_40.csv', delimiter=',', dtype='int32')
	t = x[:,1]
	dt = t[1:] - t[0:-1]	
	steps = np.array([])
	i = np.arange(dt.size)
	stdarr = np.array([])
	for k in i[1:400:5]:
		steps = np.append(steps, k)
		marr = np.array([])

		for j in i[0::k]:
			m = np.mean(dt[j:j+k])   
			marr = np.append(marr,m)
		mu = np.sum(marr)/np.float(marr.size)
		std = np.sqrt(np.sum((marr - mu)**2.)/(np.float(marr.size)-1.))
		stdarr = np.append(stdarr, std)

	steps = steps[1:]
	stdarr = stdarr[1:steps.size+1]
	plt.plot(steps, stdarr,'o')
#	plt.title('Figure 6 Mean Interval for Chunks of 100 Events')
	plt.xlabel('Start Index', fontsize=16)
	plt.ylabel('Mean Interval [Clock Ticks]', fontsize=16)
	plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
	plt.show()

def figure8():
	x = np.loadtxt('pjzsIncrease10,000_160901_1743_40.csv', delimiter=',', dtype='int32')
	t = x[:,1]
	dt = t[1:] - t[0:-1]	
	steps = np.array([])
	i = np.arange(dt.size)
	stdarr = np.array([])
	for k in i[1:400:5]:
		steps = np.append(steps, 1./np.sqrt(k))
		marr = np.array([])

		for j in i[0::k]:
			m = np.mean(dt[j:j+k])   
			marr = np.append(marr,m)
		mu = np.sum(marr)/np.float(marr.size)
		std = np.sqrt(np.sum((marr - mu)**2.)/(np.float(marr.size)-1.))
		stdarr = np.append(stdarr, std)

	steps = steps[1:]
	stdarr = stdarr[1:steps.size+1]
	linex = np.array([])
	liney = np.array([])
	linex = np.append(linex, steps[0])
	linex = np.append(linex, steps[steps.size-1])
	liney = np.append(liney, stdarr[0])
	liney = np.append(liney, stdarr[steps.size-1])

	plt.plot(steps, stdarr,'o')
	plt.plot(linex,liney)
#	plt.title('Figure 6 Mean Interval for Chunks of 100 Events')
	plt.xlabel('Start Index', fontsize=16)
	plt.ylabel('Mean Interval [Clock Ticks]', fontsize=16)
	plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
	plt.show()


#/////////////////////////////////////////Part 2 begins here//////////////////////////////////////////

def figure9():
	#code from figure 3
	x = np.loadtxt('pjzsIncrease10,000_160901_1743_40.csv', delimiter=',', dtype='int32')
	t = x[:,1]
	dt = t[1:] - t[0:-1]
	
	#New code, STAAAAAAAAAAAAAAAAART
	N = 250 #use 250 cause 500 was way too many to look pretty with. Looks clower to example figure.
	# define the lower and upper bin edges and bin width 
	bw = (dt.max()-dt.min())/(N-1.)
	binl = dt.min() + bw * np.arange(N)

	# define the array to hold the occurrence count
	bincount = np.array([])# loop through the bins
	for bin in binl:    
		count = np.where((dt >= bin) & (dt < bin+bw))[0].size    
		bincount = np.append(bincount,count)
	
	#compute bin centers for plotting

	binc = binl + 0.5*bw
	plt.figure()
	plt.plot(binc,bincount,drawstyle='steps-mid', lw = 1.5)
	x1,x2,y1,y2 = plt.axis()
	plt.axis((x1,5e7,y1,2000))
	##plt.title('Figure 9 Frequency vs Interval')
	plt.xlabel('Interval [Ticks]', fontsize=16)
	plt.ylabel('Frequency', fontsize=16)
	plt.show()

def figure10():
	#code from figure 3
	x = np.loadtxt('pjzsIncrease10,000_160901_1743_40.csv', delimiter=',', dtype='int32')
	t = x[:,1]
	dt = t[1:] - t[0:-1]
	
	#New code, STAAAAAAAAAAAAAAAAART
	N = 250
	# define the lower and upper bin edges and bin width 
	bw = (4000-dt.min())/(N-1.)
	binl = dt.min() + bw * np.arange(N)
	
	# define the array to hold the occurrence count
	bincount = np.array([])# loop through the bins
	index = 0;
	for bin in binl:    
		count = np.where((dt >= bin) & (dt < bin+bw))[0].size
		bincount = np.append(bincount,count)
	
	#compute bin centers for plotting

	binc = binl + 0.5*bw
	plt.figure()
	plt.plot(binc,bincount,drawstyle='steps-mid', lw = 1.3)
	x1,x2,y1,y2 = plt.axis()
	plt.axis((x1,4000,y1,y2))
	##plt.title('Figure 9 Frequency vs Interval')
	plt.xlabel('Interval [Ticks]', fontsize=16)
	plt.ylabel('Frequency', fontsize=16)
	plt.show()

def figure11(maths):
	#code from figure 3 + 5
	x = np.loadtxt('pjzsIncrease10,000_160901_1743_40.csv', delimiter=',', dtype='int32')
	t = x[:,1]
	dt = t[1:] - t[0:-1]
	m = np.mean(dt[0:dt.size-1])  

	marr = np.array([])
	i = np.arange(dt.size)
	for j in i:              
		m = np.mean(dt[0:j])   
		marr = np.append(marr,m)

	#New code, STAAAAAAAAAAAAAAAAART
	N = 50
	# define the lower and upper bin edges and bin width 
	bw = (dt.max()-4000)/(N-1.)
	binl = 4000 + bw * np.arange(N)
	
	# define the array to hold the occurrence count
	bincount = np.array([])# loop through the bins
	index = 0;
	for bin in binl:    
		count = np.where((dt >= bin) & (dt < bin+bw))[0].size
		bincount = np.append(bincount,count)
	
	#compute bin centers for plotting

	binc = binl + 0.5*bw
	y = maths(m,binc)*10000*bw
	fig = plt.figure()
	ax = fig.add_subplot(211)
	ax.plot(binc,bincount,drawstyle='steps-mid', lw = 1.3)
	ax.plot(binc, y)
	ax.set_yscale('log')
	plt.xlabel('Interval [Ticks]', fontsize=16)
	plt.ylabel('Frequency', fontsize=16)
	plt.xlim(0,1e8)

	ax2 = fig.add_subplot(212)
	ax2.plot(binc,bincount,drawstyle='steps-mid', lw = 1.3)
	ax2.plot(binc, y)
	plt.xlabel('Interval [Ticks]', fontsize=16)
	plt.ylabel('Frequency', fontsize=16)
	plt.subplots_adjust(hspace=0.2)

	plt.show()

def figure12():
	a = np.loadtxt('pjzsOff10,000_160901_1741_40.csv', delimiter=',', dtype='int32')	
	b = np.loadtxt('pjzs1LED10,000_160901_1732_40.csv', delimiter=',', dtype='int32')	
	c = np.loadtxt('pjzsPMT_160913_1353_40.csv', delimiter=',', dtype='int32')
	d = np.loadtxt('pjzsLEDIN_160913_1355_40.csv', delimiter=',', dtype='int32')
	e = np.loadtxt('pjzsIncrease10,000_160901_1743_40.csv', delimiter=',', dtype='int32')
	f = np.loadtxt('pjzs3Max10,000_160901_1738_40.csv', delimiter=',', dtype='int32')

	lst = [a,b,c,d,e,f]
	marr = np.array([])	
	stdarr = np.array([])

	for item in lst:
		t = item[:,1]
		dt = t[1:] - t[0:-1]
		count = 0
		for interval in dt:
			if interval < 10000:
				dt = np.delete(dt, count)
			else: count += 1	
		m = np.mean(dt[0:dt.size-1]) 
		marr = np.append(marr,m)
		std = np.sqrt(np.sum((dt - m)**2.)/(np.float(dt.size)-1.))
		stdarr = np.append(stdarr, std)
	x = np.arange(dt.max())
	y = x
	plt.plot(marr, stdarr, 'o')
	plt.plot(x,y)
	plt.xlabel('Start Index', fontsize=16)
	plt.ylabel('Mean Interval [Clock Ticks]', fontsize=16)
	plt.show()

def figure14(pois):
	x = np.loadtxt('pjzsIncrease10,000_160901_1743_40.csv', delimiter=',', dtype='int32')
	t = x[:,1]
	dt = t[1:] - t[0:-1]
	#veto afterpulses!
	wa = np.where(dt>4000)[0]
	dt = dt[wa]
	
	#reconstruct time series without jumps
	t1 = np.cumsum(dt)
	t1 = t1[:2499]
	fig = plt.figure()
	ax = fig.add_subplot(311)
	plt.plot(np.arange(2499),t1)
	plt.xlim(0,3000)
	plt.xlabel('Event Number', fontsize=16)
	plt.ylabel('Time [ticks]', fontsize=16)


	N = 1500
	# define the lower and upper bin edges and bin width 
	bw = (t1.max()-t1.min())/(N-1.)
	binl = t1.min() + bw * np.arange(N)
	
	# define the array to hold the occurrence count
	bincount = np.array([])# loop through the bins
	index = 0;
	for bin in binl:    
		count = np.where((t1 >= bin) & (t1 < bin+bw))[0].size
		bincount = np.append(bincount,count)
	
	#compute bin centers for plotting

	binc = binl + 0.5*bw
	ax2 = fig.add_subplot(312)
	ax2.plot(binc, bincount, drawstyle='steps-mid', lw = 1.3)
	plt.xlabel('Time [ticks]', fontsize=16)
	plt.ylabel('Counts per Bin', fontsize=16)

	xvals = np.arange(bincount.max()+1)
	xvals2 = np.linspace(xvals.min(), xvals.max(), 100)

	freq = np.arange(bincount.max()+1)
	for bins in bincount:
		freq[int(bins)] += 1

	m = np.mean(bincount)
	y = pois(np.arange(bincount.max()+1), m)
	y = y*bincount.size

	ax3 = fig.add_subplot(313)
	ax3.plot(range(freq.size),freq, drawstyle='steps-mid', lw = 1.3)

	#for very wrong poisson.
	#xvals = np.arange((bincount.max()+1)*10.)
	#m = np.mean(bincount)*10.
	#y = pois(xvals, m)
	#y = y*bincount.size*10
	#ax3.plot(xvals/10., y, label=r'$\bar{x}$')

	#for our poisson using integers
	ax3.plot(xvals, y)

	#For interpolated poisson
	#power_smooth = spline(xvals, y, xvals2)
	#ax3.plot(xvals2, power_smooth, label='$\bar{x}$')

	#for real poisson
	#parameters, cov_matrix = curve_fit(pois, range(freq.size), freq)
	#ax3.plot(xvals2,pois(xvals2, *parameters)*bincount.size)

	plt.xlabel('Counts per Bin', fontsize=16)
	plt.ylabel('Frequency', fontsize=16)
	plt.ylim(0,600)
	plt.xlim(-1,9)
	plt.subplots_adjust(hspace=0.2)
	plt.show()

main(a)
