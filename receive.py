import sounddevice as sd
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

duration = 2
Fs = 44100
sd.default.samplerate = Fs
sd.default.channels = 1


t = np.linspace(0, duration, Fs*duration)
noise = np.cos(t * 2 * pi * 440)


myrecording = sd.playrec(noise)
sd.wait()
myrecording = np.reshape(myrecording, len(myrecording))
n = np.linspace(-99, 100)
wc = 400 * 2 * pi / Fs
h = wc/pi*np.sinc(wc*n/np.pi)

recFilt = np.convolve(myrecording, h, 'same')

M = np.abs(np.fft.fftshift(np.fft.fft(myrecording)))
R = np.abs(np.fft.fftshift(np.fft.fft(recFilt)))
fs = np.linspace(-np.pi, np.pi, len(M))

if __name__ == "__main__":
	plt.subplot(311)
	plt.plot(fs, M)

	plt.subplot(312)
	plt.plot(t, myrecording)
	
	plt.subplot(313)
	plt.plot(fs, R)

	plt.show()

