import convolution as conv
import numpy as np
import cv2
import time
import sys


img = cv2.imread('../input/input-p1-2-1-0.png', 1)
if img is None:
	print("Image not found.")
	sys.exit()
#Kernel 3x3 edge detection
msk = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
start_time = time.time()
img_conv = conv.convolution (img, msk)
print("A 3x3 mask took %s seconds in our convolution function" % round(time.time()-start_time, 4))
cv2.imwrite('../output/output-p1-2-1-0.png', img_conv)
start_time = time.time()
img_conv_opencv = cv2.filter2D(img, -1, msk)
print("A 3x3 mask took %s seconds in OpenCV filter" % round(time.time()-start_time, 4))
#cv2.imwrite('../output/opencv0.png', img_conv_opencv)


#Kernel 7x7 box blur
msk = np.full((7,7), 1/(7*7))
start_time = time.time()
img_conv = conv.convolution (img, msk)
print("A 7x7 mask took %s seconds in our convolution function" % round(time.time()-start_time, 4))
cv2.imwrite('../output/output-p1-2-1-1.png', img_conv)
start_time = time.time()
img_conv_opencv = cv2.filter2D(img, -1, msk)
print("A 7x7 mask took %s seconds in OpenCV filter" % round(time.time()-start_time, 4))
#cv2.imwrite('../output/opencv1.png', img_conv_opencv)


#Kernel 15x15 box blur
msk = np.full((15,15), 1/(15*15))
start_time = time.time()
img_conv = conv.convolution (img, msk)
print("A 15x15 mask took %s seconds in our convolution function" % round(time.time()-start_time, 4))
cv2.imwrite('../output/output-p1-2-1-2.png', img_conv)
start_time = time.time()
img_conv_opencv = cv2.filter2D(img, -1, msk)
print("A 15x15 mask took %s seconds in OpenCV filter" % round(time.time()-start_time, 4))
#cv2.imwrite('../output/opencv2.png', img_conv_opencv)

#Kernel 50x50 box blur
msk = np.full((50,50), 1/(50*50))
start_time = time.time()
img_conv = conv.convolution (img, msk)
print("A 50x50 mask took %s seconds in our convolution function" % round(time.time()-start_time, 4))
cv2.imwrite('../output/output-p1-2-1-3.png', img_conv)
start_time = time.time()
img_conv_opencv = cv2.filter2D(img, -1, msk)
print("A 50x50 mask took %s seconds in OpenCV filter" % round(time.time()-start_time, 4))
#cv2.imwrite('../output/opencv3.png', img_conv_opencv)


#Testes com kernels de formas diferentes, nao é necesario para o programa final
#msk = np.array([[-0.5, 0, 0.5]])
#img_conv = conv.convolution (img, msk)
#cv2.imwrite('../output/output-p1-2-1-teste-0.png', img_conv)

#msk = np.array([[0.6]])
#img_conv = conv.convolution (img, msk)
#cv2.imwrite('../output/output-p1-2-1-teste-01.png', img_conv)

#msk = np.array([[-1, -1, -1], [-1, 5, -1]])
#img_conv = conv.convolution (img, msk)
#cv2.imwrite('../output/output-p1-2-1--teste-02.png', img_conv)
