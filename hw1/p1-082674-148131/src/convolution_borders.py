import numpy as np
import cv2
import sys

def zero (sub_img, i, j, l, h, hodd, lodd, height, length):
	if i-h < 0:
		aux = np.zeros((h-i, sub_img.shape[1], 3))
		sub_img = np.concatenate((aux, sub_img), axis=0)
	if i+h+hodd > height:
		aux = np.zeros((i+h+hodd-height,sub_img.shape[1], 3))
		sub_img = np.concatenate((sub_img,aux), axis=0)
	if j-l < 0:
		aux = np.zeros((sub_img.shape[0],l-j, 3))
		sub_img = np.concatenate((aux, sub_img), axis=1)
	if j+l+lodd > length:
		aux = np.zeros((sub_img.shape[0],j+l+lodd-length, 3))
		sub_img = np.concatenate((sub_img,aux), axis=1)
	return sub_img

def mirror (sub_img, i, j, l, h, hodd, lodd, height, length):
	if i-h < 0:
		aux = np.flipud(sub_img[0:h-i,:])
		sub_img = np.concatenate((aux, sub_img), axis=0)
	if i+h+hodd > height:
		aux = np.flipud(sub_img[height-(i+h+hodd):height,:])
		sub_img = np.concatenate((sub_img,aux), axis=0)
	if j-l < 0:
		aux = np.fliplr(sub_img[:,0:l-j])
		sub_img = np.concatenate((aux, sub_img), axis=1)
	if j+l+lodd > length:
		aux = np.fliplr(sub_img[:,length-(j+l+lodd):length])
		sub_img = np.concatenate((sub_img,aux), axis=1)
	return sub_img

def clamp(sub_img, i, j, l, h, hodd, lodd, height, length):
	if i-h < 0:
		aux = sub_img[0:1,:]
		aux = np.repeat(aux, h-i, axis=0)
		sub_img = np.concatenate((aux, sub_img), axis=0)
	if i+h+hodd > height:
		aux = sub_img[-1:,:]
		aux = np.repeat(aux, i+h+hodd-height, axis=0)
		sub_img = np.concatenate((sub_img,aux), axis=0)
	if j-l < 0:
		aux = sub_img[:,0:1]
		aux = np.repeat(aux, l-j, axis=1)
		sub_img = np.concatenate((aux, sub_img), axis=1)
	if j+l+lodd > length:
		aux = sub_img[:,-1:]
		aux = np.repeat(aux, j+l+lodd-length, axis=1)
		sub_img = np.concatenate((sub_img,aux), axis=1)
	return sub_img

def convolution (img_input, mask, border = 1):
	height, length, rgb = img_input.shape

	kernel = np.fliplr(np.flipud(mask)) # flip the kernel
	k_shape = kernel.shape
	k_height, k_length = k_shape[0], k_shape[1]
	h, l = k_height//2, k_length//2

	# odd variable used so even kernels can work.
	hodd = k_height%2
	lodd = k_length%2 

	if(height < k_height or length < k_length):
		return -1

	img_output = np.empty(img_input.shape)
	for i in range(height):
		for j in range(length):
			# get the sub-array
			sub_img = img_input[max(0, i-h):min(height, i+h+hodd), max(0, j-l):min(length, j+l+lodd)]
			sub_h, sub_l, sub_rgb = sub_img.shape
			if sub_h != k_height or sub_l != k_length: 
				#pads the border
				if border == 0:
					sub_img = zero(sub_img, i, j, l, h, hodd, lodd, height, length)
				elif border == 1:
					sub_img = mirror(sub_img, i, j, l, h, hodd, lodd, height, length)
				elif border == 2:
					sub_img = clamp(sub_img, i, j, l, h, hodd, lodd, height, length)
			if(sub_img.shape[0] != k_height or sub_img.shape[1] != k_length):
				# checks to see if the img was padded correctly
				print("Shape Error.", i, j, sub_img.shape, k_shape)
				return -1
			for n in range(rgb):
				m = np.multiply(sub_img[:,:,n], kernel)
				s = np.sum(m)
				img_output[i,j,n] = s
	return img_output


img = cv2.imread('../input/input-p1-2-1-1.jpg', 1)
if img is None:
	print("Image not found.")
	sys.exit()
#Kernel 15x15 box blur detection
msk = np.full((15, 15), 1/(15*15))
img_conv = convolution (img, msk, 0)
cv2.imwrite('../output/output-p1-2-1-5.png', img_conv)

img_conv = convolution (img, msk, 1)
cv2.imwrite('../output/output-p1-2-1-6.png', img_conv)

img_conv = convolution (img, msk, 2)
cv2.imwrite('../output/output-p1-2-1-7.png', img_conv)

















