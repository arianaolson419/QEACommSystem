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

def StringToBits(message):
	'''
	Converts a string to its binary representation

	message: a string

	returns a numpy array of binary bits (8 bits per character)
	'''
	asciiArray = np.array([ord(c) for c in message], dtype=np.uint8)
	binArray = np.unpackbits(asciiArray)
	return binArray

def BitsToSignal(binArray, Fs, bitTime):
	'''
	encodes an array of binary bits into a signal made up of 1s and -1s.

	binArray: a numpy array of binary bits
	Fs: the sampling frequency measured in samples/second
	bitTime: how long each bit is played for, measured in seconds

	returns a numpy array of 1s and -1s
	'''
	numberOfBits = len(binArray)	# number of zeros and ones in array
	duration = bitTime * numberOfBits	# time it takes to play the audio signal. measured in seconds

	#initialize arrays
	signalArray = np.ones(int(duration * Fs))	# initialze the signal array

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
	return signalArray
	
def Modulate(signalArray, Fs, fc):
	'''
	Modulates a signal into an array that can be played
	as an audible signal. Modulation done using Binary Phase-
	Shift Keying.

	signalArray: a numpy array of 1s and -1s
	Fs: the sampling frequency measured in samples/second
	fc: the frequency of the carrier wave, measured in Hz
	
	returns a numpy array that represents an audible signal
	'''
	
	#create the carrier wave
	t = np.linspace(0, len(signalArray) / Fs, len(signalArray))	# the time array
	carrier = np.cos(2 * pi * fc * t)	# the carrier wave to be phase shifted

	audioSignal = signalArray * carrier	  # modulate the signal
	return audioSignal

def PlayRecord(signalArray, Fs, record=True):
	'''
	Simultaneously plays an array representing an audio signal 
	and records the resulting sound

	signalArray: a numpy array representing an audio signal
	Fs: the sampling frequency measured in samples per second
	record: a boolean indicating if sound should be recorded

	returns a numpy array of the recorded signal if record is True
	otherwise function is void.
	'''

	arrayShape = np.shape(signalArray)	#size of the np array
	
	sd.default.samplerate = Fs
	sd.default.channels = 1

	if record == True:
		# plays and records souns simultaneously
		audioArray = sd.playrec(signalArray, blocking=True)
		return np.reshape(audioArray, arrayShape)	#reshapes to decrease memory usage in calculations
	else:
		# only plays sound
		audioArray = sd.play(signalArray, blocking=True)




def Demodulate(signalArray, Fs, fc):
	'''
	Demodulates an audio signal that is assumed to have been modulated using BPSK back into an 
	approximation of the original signal before BPSK and was performed

	signalArray: a numpy array representing a BPSK modulated audio signal
	Fs: the sampling frequency
	fc the frequency of the carrier wave

	returns a numpy array representing a reconstruction of the original pre-modulated signal
	'''

	signalArray = HighPass(LowPass(signalArray, fc * 1.1, Fs), fc * 0.9, Fs)	# bandpass to isolate fc
	
	t = np.linspace(0, len(signalArray)/Fs, len(signalArray))	
	c = np.cos(2 * pi * fc * t)
	
	shiftedArray = signalArray * c 	# multiply by a cosine of the carrier frequency to shift the signal
	middle = LowPass(shiftedArray, fc/2, Fs)	# remove the outer frequencies that are artifacts of the shift
	
	return middle * 2	# amplify the signal

def SignalToBits(signalArray, bitTime, Fs):
	'''
	Converts a BPSK demodulated signal into an array of binary bits

	signalArray: a numpy array representing a BPSK demodulated signal
	bitTime: how long each bit is played for, measured in seconds
	Fs: the sampling frequency, measured in samples/second

	returns a numpy array of binary bits
	'''
	binArray = [0]

	threshold = np.max(signalArray) / 3
	step = int(bitTime * Fs)
	index = 0
	sign = 0

	for i in range(len(signalArray)):
		if np.abs(signalArray[i]) > threshold:
			# detects start of the message
			index = i + int(step * 1.25)	# index of first bit
			# determines if first bit is positive or negative
			if signalArray[i] > 0:
				sign = 1
			elif signalArray[i] < 0:
				sign = -1 
			break
	while index < len(signalArray):
		# checks if signal is positive or negative
		if signalArray[index] > 0:
			currentSign = 1
		elif signalArray[index] < 0:
			currentSign = -1

		if currentSign == sign:
			# next bit is a 0 if the sign is the same as the initial bit
			binArray.append(0)
		else:
			#next bit is a 1 if the sign is different than the initial bit
			binArray.append(1)

		index += step


	return binArray

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

def LowPass(signalArray, wc, Fs):
	'''
	Implements a low pass filter

	signal array: a numpy array representing an audio signal
	wc: the desired corner frequency in Hz
	Fs: the sampling frequency, measured in samples/second

	returns a numpy array representing the filtered audio signal 
	of the same size as the original signal
	'''

	W = wc  * 2 * pi / Fs    # converts corner frequency to radians/sample
	n = np.arange(-41,42)
	h = W / pi * np.sinc(W * n / pi)	# impulse response of the low pass filter
	filteredSignal = np.convolve(signalArray, h, 'same')
	return filteredSignal

def HighPass(signalArray, wc, Fs):
	'''
	Implements a high pass filter

	signal array: a numpy array representing an audio signal
	wc: the desired corner frequency
	Fs: the sampling frequency, measured in samples/second

	returns a numpy array representing the filtered audio signal 
	of the same size as the original signal
	'''
	filteredSignal = signalArray - LowPass(signalArray, wc, Fs)
	return filteredSignal

def TimeDomainPlot(signalArray, Fs):
	'''
	creates a a plot of a signal in the time domain

	signalArray: a numpy array representing a time domain signal
	Fs: the sampling frequency, measured in samples/second
	'''

	t = np.linspace(0, len(signalArray)/Fs, len(signalArray))	#create the x axis, measured in seconds
	plt.plot(t, signalArray)

def FrequencyDomainPlot(signalArray):
	'''
	creates an FFT plot of a signal

	signalArray: a numpy array representing a time domain signal
	'''
	S = np.fft.fftshift(np.fft.fft(signalArray))	# take the fft of the signal
	fs = np.linspace(-pi, pi, len(S))	# create the x axis, measured in radians/sample
	plt.plot(fs, abs(S))

def Modem(messageOut, Fs = 44100, fc = 800, bitTime = 0.25):
	'''
	An acoustic modem that translates a string to an audio signal and then back to a string again. 
	Acts as both the transmitter and the receiver.

	Generates plots of the message along its journey.

	messageOut: a string to be transmitted.
	Fs: the sampling frequency, measured in samples/second
	fc: the frequency of the audio signal, measured in Hz
	bitTime: how long each bit is played for, measured in seconds

	displays the original and received messages, but is a void function
	'''

	# generate a signal to "warm up" the speakers and minimize DC offset of the message 
	helloSignal = BitsToSignal(np.zeros(4), Fs, bitTime)
	helloSound = Modulate(helloSignal, Fs, fc)
	
	# Translate the message into an audio signal
	binArray = StringToBits(messageOut)
	encodedSignal = BitsToSignal(binArray, Fs, bitTime)
	transmittedSignal = Modulate(encodedSignal, Fs, fc)

	# Play the warm up tone
	PlayRecord(helloSound, Fs, record=False)

	# Play and record the audio signal
	receivedSignal = PlayRecord(transmittedSignal, Fs)

	# Translate back to a string
	demod = Demodulate(receivedSignal, Fs, fc)
	binArray = SignalToBits(demod, bitTime, Fs)
	messageIn = BitsToString(binArray)

	# Display messages
	print "transmitted: ", messageOut
	print "received: ", messageIn
	
	# Make the plots
	plt.figure(1)

	plt.subplot(411)
	TimeDomainPlot(encodedSignal, Fs)
	plt.title("Pre-Modulated Signal")

	plt.subplot(412)
	FrequencyDomainPlot(encodedSignal)
	plt.title("FFT of Pre-Modulated Signal")
	
	plt.subplot(413)
	TimeDomainPlot(transmittedSignal, Fs)
	plt.title("Modulated Signal")

	plt.subplot(414)
	FrequencyDomainPlot(transmittedSignal)
	plt.title("FFT of Modulated Signal")

	plt.figure(2)

	plt.subplot(411)
	TimeDomainPlot(receivedSignal, Fs)
	plt.title("Received Signal")

	plt.subplot(412)
	FrequencyDomainPlot(receivedSignal)
	plt.title("FFT of Received Signal")

	plt.subplot(413)
	TimeDomainPlot(demod, Fs)
	plt.title("Demodulated Signal")

	plt.subplot(414)
	FrequencyDomainPlot(demod)
	plt.title("FFT of Demodulated Signal")

	plt.show()

if __name__ == "__main__":
	Modem("a")
	





