import numpy as np
import cv2
import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

def convolution (img_input, mask):
	heigh, length, rgb = img_input.shape

	kernel = np.fliplr(np.flipud(mask)) # flip the kernel
	k_shape = kernel.shape
	k_heigh, k_length = k_shape[0], k_shape[1]
	h, l = k_heigh//2, k_length//2

	# odd variable used so even kernels can work.
	odd = k_heigh%2 # 0 if even, 1 if odd

	if(heigh < k_heigh or length < k_length):
		return -1

	img_output = np.empty(img_input.shape)
	for i in range(heigh):
		for j in range(length):
			# get the sub-array
			sub_img = img[max(0, i-h):min(heigh, i+h+odd), max(0, j-l):min(length, j+l+odd)]
			sub_h, sub_l, sub_rgb = sub_img.shape
			if sub_h != k_heigh or sub_l != k_length: 
				# if the sizes are different, it means it's near a border.
				# the next code pad the sub-array like it's doubly periodic.
				# eg. it makes the value[heigh][0] = value[heigh-1][0],
				# and value[heigh+1][0] = value[heigh-2][0], and so on.
				if i-h < 0:
					aux = np.flipud(sub_img[0:h-i,:])
					sub_img = np.concatenate((aux, sub_img), axis=0)
				if i+h+odd > heigh:
					aux = np.flipud(sub_img[heigh-(i+h+odd):heigh,:])
					sub_img = np.concatenate((sub_img,aux), axis=0)
				if j-l < 0:
					aux = np.fliplr(sub_img[:,0:l-j])
					sub_img = np.concatenate((aux, sub_img), axis=1)
				if j+l+odd > length:
					aux = np.fliplr(sub_img[:,length-(j+l+odd):length])
					sub_img = np.concatenate((sub_img,aux), axis=1)
			if(sub_img.shape[0] != k_heigh or sub_img.shape[1] != k_length):
				# checks to see if the img was padded correctly
				print("Shape Error.")
				return -1
			for n in range(rgb):
				m = np.multiply(sub_img[:,:,n], kernel)
				s = np.sum(m)
				img_output[i,j,n] = s
	return img_output

img = cv2.imread('../input/input-p1-2-1-0.png', 1)

#Kernel 3x3
msk = np.full((3,3), 1/(3*3))
img_conv = convolution (img, msk)
cv2.imwrite('../output/output-p1-2-1-0.png', img_conv)
#Kernel 7x7
msk = np.full((7,7), 1/(7*7))
img_conv = convolution (img, msk)
cv2.imwrite('../output/output-p1-2-1-1.png', img_conv)
#Kernel 15x15
msk = np.full((15,15), 1/(15*15))
img_conv = convolution (img, msk)
cv2.imwrite('../output/output-p1-2-1-2.png', img_conv)

#Kernel 50x50
msk = np.full((50,50), 1/(50*50))
img_conv = convolution (img, msk)
cv2.imwrite('../output/output-p1-2-1-3.png', img_conv)





