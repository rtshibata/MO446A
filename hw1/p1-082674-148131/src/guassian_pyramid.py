import convolution as conv
import cv2
import numpy as np

class GPyramid:
	images = []

	def down(self, image):
		# Desce um nivel na piramide
		# Duplica linhas e colunas
		# Interpola usando a função convolution 'conv.convolution(img, msk)'
		# return a imagem nova	
		return

	def up(self, image):
		# Sobe um nivel na piramide
		# Blur usando uma gaussian mask e a função convolution 'conv.convolution(img, msk)'
		# Remove linhas e colunas pares
		# return a imagem nova
		return image

	def access(self, level):
		if (images.shape > level):
			return images[level]
		else:
			# Encontra o level utilizando up/down
			return

	def __init__(self, image, levels):
		self.images.append(image)
		for i in range(levels):
			self.images.append(self.up(self.images[i]))


img = cv2.imread('../input/input-p1-2-1-0.png', 1)
G = GPyramid(img, 3)
print(G.images[0].shape, G.images[1].shape, G.images[2].shape)
		
