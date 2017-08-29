import numpy as np
import cv2
from matplotlib import pyplot as plt

def find_nth_smallest_proper_way(a, n):
    return np.partition(a, n-1)[n-1]

def spectrum_lowest_values(spectrum,percentage=None):
	"""
	selects the n% of the lowest values in phase spectrum
	(bigger than 0)
	"""
	#no_zeros_array = np.trim_zeros(spectrum.flatten())
	array = spectrum.flatten()
	
	if percentage is None:

		#get the lowest value which is different of 0 
		minval = np.min(spectrum[np.nonzero(spectrum)])
		
	else:
		#get what is the n-th lowest value according to the percentage
	 	nth = int((array.shape[0]*percentage)/100)
		#find n-th lowest value
		minval = find_nth_smallest_proper_way(array, nth)		
	
	#makes every other point become equal to 0
	spectrum[spectrum > minval] = 0
	
	return spectrum

img = cv2.imread('../input/input-p1-3-1-0.jpg',0)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

phase_spectrum = 40*np.log(cv2.phase(dft_shift[:, :, 0], dft_shift[:, :, 1], True))


#obtains magnitude and phase spectrum
cv2.imwrite('../output/output-p1-3-1-0.png', magnitude_spectrum)
cv2.imwrite('../output/output-p1-3-1-1.png', phase_spectrum)

#takes only 1 point, the lowest value
f_spec = spectrum_lowest_values(phase_spectrum)

f_ishift = np.fft.ifftshift(f_spec)
img_back = cv2.idft(f_ishift)

cv2.imwrite('../output/output-p1-3-1-2.png', img_back)

#takes 25% of the lowest values
f_spec = spectrum_lowest_values(phase_spectrum,25)

f_ishift = np.fft.ifftshift(f_spec)
img_back = cv2.idft(f_ishift)

cv2.imwrite('../output/output-p1-3-1-3.png', img_back)

#takes 50% of the lowest values
f_spec = spectrum_lowest_values(phase_spectrum,50)

f_ishift = np.fft.ifftshift(f_spec)
img_back = cv2.idft(f_ishift)

cv2.imwrite('../output/output-p1-3-1-4.png', img_back)

#takes 75% of the lowest values
f_spec = spectrum_lowest_values(phase_spectrum,75)

f_ishift = np.fft.ifftshift(f_spec)
img_back = cv2.idft(f_ishift)

cv2.imwrite('../output/output-p1-3-1-5.png', img_back)

#takes 100% of the lowest values
f_spec = spectrum_lowest_values(phase_spectrum,100)

f_ishift = np.fft.ifftshift(f_spec)
img_back = cv2.idft(f_ishift)

cv2.imwrite('../output/output-p1-3-1-6.png', img_back)
