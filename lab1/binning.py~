import numpy as np
import matplotlib.pyplot as plt

file = 'pjzs1LED10,000_160901_1732_40.csv'

# Get the data
x = np.loadtxt(file, delimiter=',',dtype='int32')
t = x[:,1]
#t = t*1.0
# Compute the intervals
dt =t[1:] - t[0:-1]

#veto afterpulses
wa = np.where(dt > 3000)[0]
dt = dt[wa]

#Reconstruct the time series
t1 = np.cumsum(dt)
t1 = t1[:2499]

plt.plot(np.arange(2499),t1)
plt.show()
