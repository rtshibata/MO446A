import numpy as np
import cv2
from matplotlib import pyplot as plt

def blend(imgleft, imgright, mask):

	imgleft = np.multiply(imgleft, mask)
	imgright = np.multiply(imgright, 1 - mask)

	dft_l = cv2.dft(np.float32(imgleft),flags = cv2.DFT_COMPLEX_OUTPUT)
	dft_shift_l = np.fft.fftshift(dft_l)
	dft_r = cv2.dft(np.float32(imgright),flags = cv2.DFT_COMPLEX_OUTPUT)
	dft_shift_r = np.fft.fftshift(dft_r)

#	img1 = img1 * (mask/255)
#	img2 = img2 * (1 - mask/255)

#	dft_1 = cv2.dft(np.float32(img1),flags = cv2.DFT_COMPLEX_OUTPUT)
#	dft_2 = cv2.dft(np.float32(img2),flags = cv2.DFT_COMPLEX_OUTPUT)
#	dft_shift_1 = np.fft.fftshift(dft_1)
#	dft_shift_2 = np.fft.fftshift(dft_2)
	
	magnitude_spectrum_1 = 20*np.log(cv2.magnitude(dft_shift_l[:,:,0],dft_shift_l[:,:,1]))
	#magnitude_spectrum = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])
	phase_spectrum_1 = 40*np.log(cv2.phase(dft_shift_l[:, :, 0], dft_shift_l[:, :, 1], True))
	magnitude_spectrum_2 = 20*np.log(cv2.magnitude(dft_shift_r[:,:,0],dft_shift_r[:,:,1]))
	#magnitude_spectrum = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])
	phase_spectrum_2 = 40*np.log(cv2.phase(dft_shift_r[:, :, 0], dft_shift_r[:, :, 1], True))

#	img1_freq = magnitude_spectrum_1 * np.exp(1j*phase_spectrum_1)
#	img2_freq = magnitude_spectrum_2 * np.exp(1j*phase_spectrum_2)
#	img_blend = img1_freq + img2_freq

	mag_ = magnitude_spectrum_1 + magnitude_spectrum_2
	pha_ = phase_spectrum_1 + phase_spectrum_2

	#100% of pixels are utilized for idft
	#result,_ = idft_output_max(img_blend,100)
	f_ishift = np.fft.ifftshift(mag_)
	img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE)
	f_ishift2 = np.fft.ifftshift(pha_)
	img_back2 = cv2.idft(f_ishift2, flags=cv2.DFT_SCALE)
#	magnitude_spectrum = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
#	phase_spectrum = cv2.phase(img_back2[:, :, 0], img_back2[:, :, 1], True)
	return phase_spectrum, magnitude_spectrum


def equalize_rest_0(m,val):
	"""
	fill every element greater than val as 0
	"""
	matrix = np.where(m > val, 0, m)
	return matrix

def equalize_rest_0_bigger(m,val):
	"""
	fill every element smaller than val as 0
	"""
	matrix = np.where(m < val, 0, m)
	return matrix

def find_nth_biggest(a, n):
    #sort the elements in the array a
	result = sorted(a,reverse=True)
	return result[n-1]

def find_nth_smallest(a, n):
    #sort the elements in the array a
	result = sorted(a)
	return result[n-1]

def spectrum_biggest_values(dft,percentage=None):
	"""
	selects the n% of the biggest values in phase spectrum
	(bigger than 0)
	"""
	#no_zeros_array = np.trim_zeros(spectrum.flatten())
	array = dft.flatten()
	
	if percentage is None:
		#get the biggest value which is different of 0 
		maxval = np.max(dft[np.nonzero(dft)])
		print maxval				
	else:
		#get what is the n-th biggest value according to the percentage
	 	nth = int((array.shape[0]*percentage)/100)
		#find n-th biggest value
		maxval = find_nth_biggest(array, nth)
		print maxval		
	
	#makes every other point become equal to 0
	result = equalize_rest_0_bigger(dft,maxval)	
	#dft[dft > minval] = 0
	
	return result

def spectrum_lowest_values(dft,percentage=None):
	"""
	selects the n% of the lowest values in phase spectrum
	(bigger than 0)
	"""
	#no_zeros_array = np.trim_zeros(spectrum.flatten())
	array = dft.flatten()
	
	if percentage is None:
		#get the lowest value which is different of 0 
		minval = np.min(dft[np.nonzero(dft)])
		print minval				
	else:
		#get what is the n-th lowest value according to the percentage
	 	nth = int((array.shape[0]*percentage)/100)
		#find n-th lowest value
		minval = find_nth_smallest(array, nth)
		print minval		
	
	#makes every other point become equal to 0
	result = equalize_rest_0(dft,minval)	
	#dft[dft > minval] = 0
	
	return result

def idft_output_min(dft_shift,percentage=None):
	f_spec = spectrum_lowest_values(dft_shift,percentage)

	f_ishift = np.fft.ifftshift(f_spec)
	img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE)
	magnitude_spectrum = (cv2.magnitude(img_back[:,:,0],img_back[:,:,1]))
	phase_spectrum = (cv2.phase(img_back[:, :, 0], img_back[:, :, 1], True))

	return phase_spectrum, magnitude_spectrum

def idft_output_max(dft_shift,percentage=None):
	f_spec = spectrum_biggest_values(dft_shift,percentage)

	f_ishift = np.fft.ifftshift(f_spec)
	img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE)
	magnitude_spectrum = (cv2.magnitude(img_back[:,:,0],img_back[:,:,1]))
	phase_spectrum = (cv2.phase(img_back[:, :, 0], img_back[:, :, 1], True))

	return phase_spectrum, magnitude_spectrum


