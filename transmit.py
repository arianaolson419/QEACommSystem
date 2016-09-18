'''
Transmitter for communications system
Transmits an audio signal
By Ariana Olson September 2016
'''
import scikits.audiolab as audiolab
import numpy as np
import matplotlib.pyplot as plt


#create the time vector
Fs = 10000	#samples/second
timeDuration = 1	#second
t = np.linspace(0, timeDuration, Fs)
note = 440	# an A note
y = np.sin(t*note*2*np.pi)
h = np.array([0.1, -0.08, 0.06, -0.05, 0.04])
x = np.array([1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1])

xc = np.convolve(x, h)


if __name__ == '__main__':
	plt.stem(xc)	#stem plot of convolved signal
	audiolab.play(y, fs=Fs)	#play a 440Hz note
	plt.show()