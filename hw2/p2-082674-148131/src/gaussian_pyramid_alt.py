import cv2
import numpy as np
import math

class GPyramid:
	def height(self):
		return len(self.images)

	@staticmethod
	def down(image, expected_shape=None, o=math.sqrt(2)/2):
		# Desce um nivel na piramide
		# Duplica linhas e colunas
		image_down = np.repeat(image, 2, axis=0)
		image_down = np.repeat(image_down, 2, axis=1)

		#Interpolation with a 3x3 box blur kernel.
		image_down = cv2.GaussianBlur(image_down, (5, 5), o)

		# Fix the shape in case the lower image was odd
		if expected_shape is not None:	
			if image_down.shape[0] > expected_shape[0]:
				image_down = np.delete(image_down, image_down.shape[0]-1, axis=0)
				if image_down.shape[0] != expected_shape[0]:
					print("Wrong shape in the axis=0.")
					return None
			if image_down.shape[1] > expected_shape[1]:
				image_down = np.delete(image_down, image_down.shape[1]-1, axis=1)
				if image_down.shape[1] != expected_shape[1]:
					print("Wrong shape in the axis=1.")
					return None

		return image_down
	@staticmethod
	def up(image, o=math.sqrt(2)/2):
		# Blur
		image_up = cv2.GaussianBlur(image, (5, 5), o)
		# Remove linhas e colunas pares
		image_up = image_up[::2,::2]
		return image_up

	@staticmethod
	def blur(image, o=math.sqrt(2)/2):
		image_blured = cv2.GaussianBlur(image, (5, 5), o)
		return image_blured

	def access(self, level):
		if(level < 0):
			print("Level value must be positive.\n")
			return None
		if (len(self.images) > level):
			return self.images[level][0]
		else:
			for i in range(len(self.images)-1, level+1):
				self.images.append(self.up(self.images[i][0]))
			return self.images[level]

	def __init__(self, image, levels, n_images=3, o=math.sqrt(2)/2, k=2):
		self.images = []
		img = [self.down(image, o=o)]
		for j in range(n_images-1):
			img.append(self.blur(img[j], k))
		self.images.append(img)
		for i in range(1, levels):
			img = [self.up(self.images[i-1][0], o)]
			for j in range(n_images-1):
				img.append(self.blur(img[j], k))
			self.images.append(img)
