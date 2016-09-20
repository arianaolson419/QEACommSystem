'''
Transmitter for communications system
Transmits an audio signal
By Ariana Olson September 2016
'''
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt


#create the time vector
Fs = 10000	#samples/second
timeDuration = 0.5	#second
numberOfSamples = Fs * 0.5
t = np.linspace(0, timeDuration, Fs*timeDuration)
note = 440	# an A note
y = np.sin(t*note*2*np.pi)
h = np.array([0.1, -0.08, 0.06, -0.05, 0.04])
x = np.array([1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1])

xc = np.convolve(x, h)
sd.default.samplerate = Fs
sd.default.channels = 1


if __name__ == '__main__':
	sd.play(y)	#play a 440Hz note
	plt.plot(y)
	plt.show()