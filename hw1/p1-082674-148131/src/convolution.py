import numpy as np

def convolution (img_input, mask):
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
				# if the sizes are different, it means it's near a border.
				# the next code pad the sub-array by mirroing
				# eg. it makes the value[height][0] = value[height-1][0],
				# and value[height+1][0] = value[height-2][0], and so on.
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
			if(sub_img.shape[0] != k_height or sub_img.shape[1] != k_length):
				# checks to see if the img was padded correctly
				print("Shape Error.")
				return -1
			for n in range(rgb):
				m = np.multiply(sub_img[:,:,n], kernel)
				s = np.sum(m)
				img_output[i,j,n] = s
	return img_output
