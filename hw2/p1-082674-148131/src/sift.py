import cv2
import numpy as np
import gaussian_pyramid_alt as gp
import math

class Sift:	

	# Find the magnitude and orientation
	@staticmethod
	def magnitude_orientation(image):
		h, l = image.shape[0], image.shape[1]
		magnitude = np.zeros(image.shape, dtype='int64')
		orientation = np.zeros(image.shape, dtype='int64')
		for i in range(h-1):
			for j in range(l-1):
				magnitude[i,j] = math.sqrt(math.pow(image[i,j]-image[i+1,j], 2) + math.pow(image[i,j]-image[i,j+1], 2))
				orientation[i,j] = math.atan2(image[i,j]-image[i+1,j], image[i,j+1]-image[i,j])
		return magnitude, orientation	

	# INCOMPLETE
	# Creates histograms of the key points' area to find a orientation for each key point
	def threshold (self, mag_threshold, ori_threshold):	
		for l in range(1 ,len(self.dog)-1):
			for o in range(len(self.dog[0])):
				magnitude, orientation = self.magnitude_orientation(self.dog[l][o])
				new_kp = []
				for kp in self.key_points[l-1][o]:
					if magnitude[kp] >= mag_threshold:
						new_kp.append(kp)
				self.key_points[l-1][o] = new_kp
		self.img_key_points()

	# Look for edges on the key points
	# Remove points that are not edges based on the threshold
	def find_edges(self, threshold, k = 0.2):
		msk1 = np.array([[-1,-2,1],	[0, 0, 0], [1, 2, 1]])
		msk2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
		
		for l in range(1 ,len(self.dog)-1):
			for o in range(len(self.dog[0])):
				Ix = cv2.filter2D(self.dog[l][o], -1, msk1)
				Iy = cv2.filter2D(self.dog[l][o], -1, msk2)
				new_kp = []
				for kp in self.key_points[l-1][o]:
					i, j = kp
					sub_Ix = Ix[i-1:i+1, j-1:j+1]
					sub_Iy = Iy[i-1:i+1, j-1:j+1]
					M = np.array([	[ np.sum(np.multiply(sub_Ix, sub_Ix)), np.sum(np.multiply(sub_Ix, sub_Iy))],
									[ np.sum(np.multiply(sub_Ix, sub_Iy)), np.sum(np.multiply(sub_Iy, sub_Iy))] ])
					R = np.linalg.det(M) - k*math.pow(np.trace(M), 2)
					if R >= threshold:
						new_kp.append(kp)
				self.key_points[l-1][o] = new_kp
		self.img_key_points()

	# Creates an images of the key points
	def img_key_points(self):
		self.array_points = []
		for l in range(1, len(self.dog)-1):
			ap_list = []
			for o in range(len(self.dog[0])):
				ap = np.zeros(self.dog[l][o].shape)
				for indx  in self.key_points[l-1][o]:
					ap[indx] = self.dog[l][o][indx]
				ap_list.append(ap)
			self.array_points.append(ap_list)

	# Verify if the pixel in the possition (i,j) is the maximun or minimum of it's neighbours
	@staticmethod
	def isminmax (i, j, images):
		pixel = images[0][i, j]
		notmax = False
		notmin = False
		for k in range(3):
			if k == 0: sub_img = images[k][i-1:i+2, j-1:j+2]
			elif k == 1: sub_img = images[k][i//2-1:i//2+2, j//2-1:j//2+2]
			else: sub_img = images[k][i*2-1:i*2+2, j*2-1:j*2+2]
			for m in range(3):
				for n in range(3):
					if(sub_img[m,n] > pixel):
						notmax = True
						if(notmin):
							return False
					elif(sub_img[m,n] < pixel):
						notmin = True
						if(notmax):
							return False	
		return True

	# Find keypoints for 'image'.
	@staticmethod
	def find_keypoints(image, image_up, image_down):
		height, length = image.shape[0], image.shape[1]
		keypoints = []
		for i in range(2, height-2):
			for j in range(2, length-2):
				if(Sift.isminmax(i, j, [image, image_up, image_down])):
					keypoints.append((i,j))
		return keypoints

	# Find all keypoints on all Difference of Gaussian images.
	def get_all_keypoints(self):
		self.key_points = []
		for l in range(1, len(self.dog)-1):
			key_points_oct = []
			for o in range(len(self.dog[0])):
				key_points_oct.append(self.find_keypoints(self.dog[l][o], self.dog[l+1][o], self.dog[l-1][o]))
			self.key_points.append(key_points_oct)
	
	# Find al Difference of Gaussian images for the pyramid
	@staticmethod
	def diference_of_gaussian(images, o, k):
		dog = []
		for i in range(len(images)):
			level_dog = []
			for j in range(len(images[i])-1):
				level_dog.append(images[i][j] - images[i][j+1])
			dog.append(level_dog)
		return dog
		
	def __init__(self, image, levels, oct_levels, o, k):
		if levels < 4:
			print("At least 4 levels required, value given was:", blur_levels)
			return -1
		self.GP = gp.GPyramid(image, levels, oct_levels,  o, k)
		self.dog = self.diference_of_gaussian(self.GP.images, o, k)
		self.get_all_keypoints()
		self.img_key_points()

