import numpy as np
import sys
import matplotlib.pyplot as plt

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
		figure11(lambda x:(1./x)*np.exp(-1.42e7/x))
		# figure11(lambda x:-(1./-1.42e7)*np.exp(x/-1.42e7))				
		# figure11()
	if '12' in a:
		figure12()
	if '13' in a:
		figure13()				
	if '14' in a:
		figure14()					


def figure1():
	x = np.loadtxt('pjzsIncrease10,000_160901_1743_40.csv', delimiter=',', dtype='int32')
	t = x[:,1]
	plt.plot(t)
	plt.title('Figure 1: Time vs Event Number')	
	plt.xlabel('Event Number')
	plt.ylabel('Clock Tick')
	plt.show()

def figure2():
	x = np.loadtxt('pjzsIncrease10,000_160901_1743_40.csv', delimiter=',', dtype='int32')
	t = x[:,1]
	dt = t[1:] - t[0:-1]
	plt.plot(dt)
	plt.title('Figure 2: Interval vs Event Number')
	plt.xlabel('Event Number')
	plt.ylabel('Interval [Clock Tick]')
	plt.show()

def figure3():
	x = np.loadtxt('pjzsIncrease10,000_160901_1743_40.csv', delimiter=',', dtype='int32')
	t = x[:,1]
	dt = t[1:] - t[0:-1]
	plt.plot(dt,',')
	plt.title('Figure 3 Interval vs Event Number (Dots)')
	plt.xlabel('Event Number')
	plt.ylabel('Interval [Clock Tick]')
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
	plt.title('Figure 4 Mean Interval for 10 Chunks of 1000 Events')
	plt.xlabel('Start Index')
	plt.ylabel('Mean Interval [Clock Ticks]')
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
	plt.title('Figure 5 Mean Interval with increasing fractions of ')
	plt.xlabel('Number of Intervals Averaged')
	plt.ylabel('Mean Interval [Clock Ticks]')
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
	plt.title('Figure 6 Mean Interval for Chunks of 100 Events')
	plt.xlabel('Start Index')
	plt.ylabel('Mean Interval [Clock Ticks]')
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
	plt.title('Figure 6 Mean Interval for Chunks of 100 Events')
	plt.xlabel('Start Index')
	plt.ylabel('Mean Interval [Clock Ticks]')
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
	plt.title('Figure 6 Mean Interval for Chunks of 100 Events')
	plt.xlabel('Start Index')
	plt.ylabel('Mean Interval [Clock Ticks]')
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
	plt.title('Figure 9 Frequency vs Interval')
	plt.xlabel('Interval [Ticks]')
	plt.ylabel('Frequency')
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
	plt.title('Figure 9 Frequency vs Interval')
	plt.xlabel('Interval [Ticks]')
	plt.ylabel('Frequency')
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
	y = maths(marr)	
	plt.figure()

	# plt.plot(binc,bincount,drawstyle='steps-mid', lw = 1.3)
	plt.plot(binc, y)
	x1,x2,y1,y2 = plt.axis()
	plt.title('Figure 9 Frequency vs Interval')
	plt.xlabel('Interval [Ticks]')
	plt.ylabel('Frequency')
	plt.show()

main(a)
