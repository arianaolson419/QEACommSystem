'''
Acoustic Modem Project QEAF2016

by Ariana Olson

Implements Binary Phase-Shift Keying using audible frequencies to create an acoustic modem. 
One computer is used as both the transmitter and receiver, as this program is meant to be more
of a proof of concept than a functional communications system.

When the program is run, the user is prompted to type in a short message. The message is then translated into binary bits,
modulated into an audible signal, and demodulated and translated back to text. 

WARNING: it is best to keep messages to one short word in length as an audble, and not 
very pleasant, signal is produced, which can be extremely annoying to yourself and your neighbors.
'''
import sounddevice as sd
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

def getMessage():
	'''
	Prompts the user to type in a short message that will be transmitted via an audio signal

	returns the message as a string
	'''
	message = raw_input("Type a message: ")
	return message

def StringToBits(message):
	'''
	Converts a string to its binary representation

	message: a string

	returns a numpy array of binary bits (8 bits per character)
	'''
	asciiArray = np.array([ord(c) for c in message], dtype=np.uint8)
	binArray = np.unpackbits(asciiArray)

	return binArray

def Modulate(binArray, Fs, fc, bitTime = .25):
	'''
	Modulates a binary array into an array that can be played
	as an audible signal. Modulation done using Binary Phase-
	Shift Keying.

	binArray: a numpy array of binary bits
	Fs: the sampling frequency measured in samples/second
	fc: the frequency of the carrier wave, measured in Hz
	bitTime: how long each bit is played for, measured in seconds

	returns a numpy array that represents an audible signal
	'''

	numberOfBits = len(binArray)	# number of zeros and ones in array
	duration = bitTime * numberOfBits	# time it takes to play the audio signal. measured in seconds
	
	#initialize arrays
	signalArray = np.ones(int(duration * Fs))	# initialze the signal array
	t = np.linspace(0, duration, Fs * duration)	# the time array
	carrier = np.cos(2 * pi * fc * t)	# the carrier wave to be phase shifted

	# encode the message
	count = 0
	bitLength = int(Fs * bitTime)	#number of points representing each bit 
	# translate the bits into 1s and -1s
	for i in binArray:
		if i == 1:
			amp = 1
		else:
			amp = -1
		firstIndex = count * bitLength
		secondIndex = (count + 1) * bitLength
		signalArray[firstIndex:secondIndex] *= amp
		count += 1
	# phase of carrier is shifted by pi at each sign change
	# cos(x + pi) = -cos(x)
	audioSignal = signalArray * carrier	
	return audioSignal

def PlayRecord(signalArray, Fs):
	'''
	Simultaneously plays an array representing an audio signal 
	and records the resulting sound

	signalArray: a numpy array representing an audio signal
	Fs: the sampling frequency measured in samples per second

	returns a numpy array of the recorded signal
	'''

	sd.default.samplerate = Fs
	sd.default.channels = 1

	return sd.playrec(signalArray, blocking=True)


def Demodulate(signalArray, Fs):
	# todo: implement demodulation
	pass

def BitsToString(binArray):
	'''
	Translates an array of binary bits to a string.

	binArray: a numpy array of 1s and 0s. The number of
	elements in the array must be a multiple of 8 in order to
	represent complete 8 bit integers

	returns a string
	'''
	asciiArray = np.packbits(binArray)
	asciiList = asciiArray.tolist()
	return ''.join(chr(i) for i in asciiList)

def LowPass(signalArray, wc):
	'''
	Implements a low pass filter

	signal array: a numpy array representing an audio signal
	wc: the desired corner frequency

	returns a numpy array representing the filtered audio signal 
	of the same size as the original signal
	'''
	n = np.arange(-41,42)
	h = wc / pi * np.sinc(wc * n / pi)

	filteredSignal = np.convolve(signalArray, h, 'same')
	return filteredSignal

def HighPass(signalArray, wc):
	'''
	Implements a high pass filter

	signal array: a numpy array representing an audio signal
	wc: the desired corner frequency

	returns a numpy array representing the filtered audio signal 
	of the same size as the original signal
	'''
	filteredSignal = signalArray - LowPass(signalArray, wc)
	return filteredSignal

if __name__ == "__main__":
	Fs = 44100
	message = StringToBits("A")
	audioArray = Modulate(message, Fs, 440)
	# receivedArray = PlayRecord(audioArray, Fs)
	t = np.linspace(0, len(audioArray), len(audioArray))
	c = np.cos(2 * pi * t * 440)
	l = LowPass(HighPass(audioArray, 200), 200)
	S = np.fft.fft(l)
	S = np.fft.fftshift(S)
	plt.plot(abs(S))
	plt.show()
	# Y = np.fft.fft(receivedArray)
	# Y = np.fft.fftshift(Y, )
	# fs = np.linspace(-pi, pi, len(receivedArray))
	# plt.plot(fs, np.abs(Y))
	# plt.show()

