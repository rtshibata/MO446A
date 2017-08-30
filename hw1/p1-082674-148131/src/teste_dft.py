import numpy as np
import cv2
from matplotlib import pyplot as plt
import fourier 
import sys
import os

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
spectrums = fourier.idft_output_min(dft_shift)
cv2.imwrite('../output/output-p1-3-1-2.png', spectrums[0])
cv2.imwrite('../output/output-p1-3-1-7.png', spectrums[1])

#takes 25% of the lowest values
spectrums = fourier.idft_output_min(dft_shift,25)
cv2.imwrite('../output/output-p1-3-1-3.png', spectrums[0])
cv2.imwrite('../output/output-p1-3-1-8.png', spectrums[1])

#takes 50% of the lowest values
spectrums = fourier.idft_output_min(dft_shift,50)
cv2.imwrite('../output/output-p1-3-1-4.png', spectrums[0])
cv2.imwrite('../output/output-p1-3-1-9.png', spectrums[1])

#takes 75% of the lowest values
spectrums = fourier.idft_output_min(dft_shift,75)
cv2.imwrite('../output/output-p1-3-1-5.png', spectrums[0])
cv2.imwrite('../output/output-p1-3-1-10.png', spectrums[1])

#takes 100% of the lowest values
spectrums = fourier.idft_output_min(dft_shift,100)
cv2.imwrite('../output/output-p1-3-1-6.png', spectrums[0])
cv2.imwrite('../output/output-p1-3-1-11.png', spectrums[1])

#takes only 1 point, the highest value
spectrums = fourier.idft_output_max(dft_shift)
cv2.imwrite('../output/output-p1-3-1-12.png', spectrums[0])
cv2.imwrite('../output/output-p1-3-1-17.png', spectrums[1])

#takes 25% of the highest values
spectrums = fourier.idft_output_max(dft_shift,25)
cv2.imwrite('../output/output-p1-3-1-13.png', spectrums[0])
cv2.imwrite('../output/output-p1-3-1-18.png', spectrums[1])

#takes 50% of the highst values
spectrums = fourier.idft_output_max(dft_shift,50)
cv2.imwrite('../output/output-p1-3-1-14.png', spectrums[0])
cv2.imwrite('../output/output-p1-3-1-19.png', spectrums[1])

#takes 75% of the highest values
spectrums = fourier.idft_output_max(dft_shift,75)
cv2.imwrite('../output/output-p1-3-1-15.png', spectrums[0])
cv2.imwrite('../output/output-p1-3-1-20.png', spectrums[1])

#takes 100% of the highest values
spectrums = fourier.idft_output_max(dft_shift,100)
cv2.imwrite('../output/output-p1-3-1-16.png', spectrums[0])
cv2.imwrite('../output/output-p1-3-1-21.png', spectrums[1])

#blending images:
#dir_path = os.path.dirname(os.path.realpath(__file__))
# Messi
imgleft = cv2.imread(#dir_path+
'../input/input-p1-3-1-0.jpg', 0)
if imgleft is None:
	print("Image 1 not found.")
	sys.exit()
# Ronaldo
imgright = cv2.imread(#dir_path+
'../input/input-p1-3-2-1.jpg', 0)
if imgright is None:
	print("Image 2 not found.")
	sys.exit()
if imgleft.shape != imgright.shape:
	print("Images is not the same sizes")
	sys.exit()

height, length = imgleft.shape[0], imgleft.shape[1]
blend_mask = np.ones((height, length//2))
blend_mask = np.concatenate((blend_mask, np.zeros((height, length-length//2))), axis=1)


fourier_result = fourier.blend(imgleft, imgright, blend_mask)
cv2.imwrite('../output/output-p1-3-2-0.png', fourier_result[0])
cv2.imwrite('../output/output-p1-3-2-1.png', fourier_result[1])
