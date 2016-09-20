import sounddevice as sd
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

def getMessage():
	prompt = raw_input("Type a message: ")
	return prompt

def StringToBits(message):
	asciiArray = np.array([ord(c) for c in message], dtype=np.uint8)
	binArray = np.unpackbits(asciiArray)
	return binArray

def Modulate(binArray, Fs, fc, bitTime = .25):
	numberOfBits = len(binArray)	# number of zeros and ones in array
	duration = bitTime * numberOfBits	# time it takes to play the audio signal
	signalArray = np.ones(int(duration * Fs))	# initialze the signal array with ones
	t = np.linspace(0, duration, Fs * duration)
	carrier = np.cos(2 * pi * fc * t)

	count = 0
	bitLength = int(Fs * bitTime)	#number of data points representing each bit 
	# create the square wave
	for i in binArray:
		if i == 1:
			amp = 1
		else:
			amp = -1
		firstIndex = count * bitLength
		secondIndex = (count + 1) * bitLength
		signalArray[firstIndex:secondIndex] *= amp
		count += 1
	audioSignal = signalArray * carrier
	return audioSignal

def PlayRecord(signalArray, Fs):
	sd.default.samplerate = Fs
	sd.default.channels = 1
	return sd.playrec(signalArray, blocking=True)


def Demodulate(signalArray, Fs):
	# todo: implement demodulation
	pass

def BitsToString(binArray):
	asciiArray = np.packbits(binArray)
	asciiList = asciiArray.tolist()
	return ''.join(chr(i) for i in asciiList)

def Transmit(Fs):
	message = "Ariana"
	binArray = StringToBits(message)
	audioSignal = Modulate(binArray, Fs, 440)
	sd.play(audioSignal, Fs)


if __name__ == "__main__":
	Fs = 44100
	sd.default.samplerate = Fs
	sd.default.channels = 1
	message = StringToBits("A")
	audioArray = Modulate(message, Fs, 440)
	receivedArray = PlayRecord(audioArray, Fs)
	t = np.linspace(0, len(receivedArray), len(receivedArray))
	Y = np.fft.fft(receivedArray)
	Y = np.fft.fftshift(Y, )
	fs = np.linspace(-pi, pi, len(receivedArray))
	plt.plot(fs, np.abs(Y))
	plt.show()

