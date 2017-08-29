import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../input/input-p1-3-1-0.jpg',0)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

phase_spectrum = 40*np.log(cv2.phase(dft_shift[:, :, 0], dft_shift[:, :, 1], True))

cv2.imwrite('../output/output-p1-3-1-0.png', magnitude_spectrum)
cv2.imwrite('../output/output-p1-3-1-1.png', phase_spectrum)
