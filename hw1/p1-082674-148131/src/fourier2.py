import numpy as np
import cv2
from matplotlib import pyplot as plt


def equalize_rest_0(m,minval):
	"""
	fill every element greater than minval as 0
	"""
	matrix = m[:,:,:]
	for i in range(matrix.shape[0]): 	
		for j in range(matrix.shape[1]):
			matrix[ matrix > minval] = 0
	#for i in matrix: 	
	#	print i
	return matrix

def find_nth_biggest(a, n):
    #sort the elements in the array a
	result = sorted(a,reverse=True)
	return result[n-1]

def find_nth_smallest(a, n):
    #sort the elements in the array a
	result = sorted(a)
	return result[n-1]

def spectrum_lowest_values(dft,percentage=None):
	"""
	selects the n% of the lowest values in phase spectrum
	(bigger than 0)
	"""
	#no_zeros_array = np.trim_zeros(spectrum.flatten())
	array = dft.flatten()
	minval = np.min(dft[np.nonzero(dft)])
	print minval
#	if percentage is None:
		#get the lowest value which is different of 0 
#		minval = np.min(dft[np.nonzero(dft)])
#		print minval
#		print "none"				
#	else:
		#get what is the n-th lowest value according to the percentage
#	 	nth = int((array.shape[0]*percentage)/100)
		#find n-th lowest value
#		minval = find_nth_smallest(array, nth)
#		print minval
#		print "percetage ",percentage		
	
	#makes every other point become equal to 0
	result = equalize_rest_0(dft,minval)	
	#dft[dft > minval] = 0
	
	return result

def idft_output(dft_shift,percentage=None):
	f_spec = spectrum_lowest_values(dft_shift,percentage)

	f_ishift = np.fft.ifftshift(f_spec)
	img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE)
	magnitude_spectrum = (cv2.magnitude(img_back[:,:,0],img_back[:,:,1]))
	phase_spectrum = (cv2.phase(img_back[:, :, 0], img_back[:, :, 1], True))

	return phase_spectrum, magnitude_spectrum


img = cv2.imread('../input/input-p1-3-1-0.jpg',0)
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
#magnitude_spectrum = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])
phase_spectrum = 40*np.log(cv2.phase(dft_shift[:, :, 0], dft_shift[:, :, 1], True))
#phase_spectrum = cv2.phase(dft_shift[:, :, 0], dft_shift[:, :, 1], True)

#obtains magnitude and phase spectrum
cv2.imwrite('../output/output-p1-3-1-0.png', magnitude_spectrum)
cv2.imwrite('../output/output-p1-3-1-1.png', phase_spectrum)

#takes only 1 point, the lowest value
spectrums = idft_output(dft_shift)
cv2.imwrite('../output/output-p1-3-1-2.png', spectrums[0])
cv2.imwrite('../output/output-p1-3-1-7.png', spectrums[1])

#takes 25% of the lowest values
spectrums = idft_output(dft_shift,25)
cv2.imwrite('../output/output-p1-3-1-3.png', spectrums[0])
cv2.imwrite('../output/output-p1-3-1-8.png', spectrums[1])

#takes 50% of the lowest values
spectrums = idft_output(dft_shift,50)
cv2.imwrite('../output/output-p1-3-1-4.png', spectrums[0])
cv2.imwrite('../output/output-p1-3-1-9.png', spectrums[1])

#takes 75% of the lowest values
spectrums = idft_output(dft_shift,75)
cv2.imwrite('../output/output-p1-3-1-5.png', spectrums[0])
cv2.imwrite('../output/output-p1-3-1-10.png', spectrums[1])

#takes 100% of the lowest values
spectrums = idft_output(dft_shift,100)
cv2.imwrite('../output/output-p1-3-1-6.png', spectrums[0])
cv2.imwrite('../output/output-p1-3-1-11.png', spectrums[1])
