import convolution as conv
import numpy as np

class GPyramid:
	def heigh(self):
		return len(self.images)

	@staticmethod
	def down(image, expected_shape=None):
		# Desce um nivel na piramide
		# Duplica linhas e colunas
		image_down = np.repeat(image, 2, axis=0)
		image_down = np.repeat(image_down, 2, axis=1)

		# Interpolar
		# Pode utilizar uma fucao de biblioteca nesse caso(prof. confimou no moodle)

		# Fix the shape in case the lower image was odd
		if expected_shape is not None:
			if image_down.shape[0] > expected_shape[0]:
				image_down = np.delete(image_down, image_down.shape[0]-1, axis=0)
				if image_down.shape[0] != expected_shape[0]:
					print("Wrong shape in the axis=0. Expected shape:", expected_shape[0], "Shape given:", image_down.shape[0])
					return None
			if image_down.shape[1] > expected_shape[1]:
				image_down = np.delete(image_down, image_down.shape[1]-1, axis=1)
				if image_down.shape[1] != expected_shape[1]:
					print("Wrong shape in the axis=1. Expected shape:", expected_shape[1], "Shape given:", image_down.shape[1])
					return None

		return image_down
	@staticmethod
	def up(image):
		# Gaussian mask, eu escolhi essa mascara só para testar a piramide, 
		# mas sera necessario testar varias masks 
		# ou encontrar uma mask que conseguimos explicar pq é a melhor.
		msk = np.array([[1, 4, 7, 4, 1],
						[4,16,26,16, 4],
						[7,26,41,26, 7],
						[4,16,26,16, 4],
						[1, 4, 7, 4, 1]])*1/273
		# Blur
		image_up = conv.convolution(image, msk)
		# Remove linhas e colunas pares
		image_up = image_up[::2,::2]
		return image_up

	def access(self, level):
		if(level < 0):
			print("Level value must be positive.\n")
			return None
		if (len(self.images) > level):
			return self.images[level]
		else:
			for i in range(len(self.images)-1, level-1):
				self.images.append(self.up(self.images[i]))
			return self.images[level]

	def __init__(self, image, levels):
		self.images = []
		self.images.append(image)
		for i in range(levels-1):
			self.images.append(self.up(self.images[i]))

