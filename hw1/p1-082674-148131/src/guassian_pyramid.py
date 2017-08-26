import convolution as conv
import cv2
import numpy as np
import sys

class GPyramid:
	images = []

	def down(self, image):
		# Desce um nivel na piramide
		# Duplica linhas e colunas
		image_down = np.repeat(image, 2, axis=0)
		image_down = np.repeat(image_down, 2, axis=1)

		# Interpolar
		# Pode utilizar uma fucao de biblioteca nesse caso(prof. confimou no moodle)

		return image_down

	def up(self, image):
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
			for i in range(len(self.images)-1, level):
				self.images.append(self.up(self.images[i]))
			return self.images[level]

	def __init__(self, image, levels):
		self.images.append(image)
		for i in range(levels):
			self.images.append(self.up(self.images[i]))

img = cv2.imread('../input/input-p1-2-2-0.jpg', 1)
if img is None:
	print("Image not found.")
	sys.exit()
G = GPyramid(img, 2)
#Down-scaling(Subindo na piramide)
cv2.imwrite('../output/output-p1-2-2-0.png', G.access(1))
cv2.imwrite('../output/output-p1-2-2-1.png', G.access(2))
#Upscaling(DEscendo na piramide)
cv2.imwrite('../output/output-p1-2-2-3.png', G.down(G.images[1]))
cv2.imwrite('../output/output-p1-2-2-4.png', G.down(G.images[2]))

