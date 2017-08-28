import gaussian_pyramid as gp
import convolution as conv
import numpy as np

class LPyramid:

	def height(self):
		return len(self.images)
	
	# Returns the sum of the laplacian image i with the gaussian image i+1 upsampled.
	def down(self, image, level):
		return self.images[level] + gp.GPyramid.down(image, self.images[level].shape)

	# Appends the gaussian image i+1 to the pyramid
	# generates the laplacian image i
	# returns the gaussian image.
	def up(self):
		level = len(self.images)
		gaussian_this = self.images[level-1]
		gaussian_up = gp.GPyramid.up(gaussian_this)
		self.images.append(gaussian_up)
		self.images[level-1] = gaussian_this - gp.GPyramid.down(gaussian_up, gaussian_this.shape)
		return self.images[level]

	# Returns the gaussian image at the requested level.
	def access(self, level):
		pyramid_len = len(self.images)
		if(level < 0):
			print("Level value must be positive.\n")
			return None
		if (level == pyramid_len-1):
			return self.images[level]
		elif (level > pyramid_len):
			for i in range(pyramid_len, level-1):
				output = self.up()
			return output
		else:
			output = self.images[pyramid_len-1]
			for i in reversed(range(level, pyramid_len-1)):
				output = self.down(output, i)
			return output

	# Generates a pyramid of the image with the inputed level.
	def __init__(self, image=None, levels=None):
		self.images = []
		if(image is not None):
			self.images.append(image)
			if (levels is not None):
				for i in range(levels-1):
					self.up()

