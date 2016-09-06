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

main(a)
