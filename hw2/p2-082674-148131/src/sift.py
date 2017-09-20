import cv2
import numpy as np
import gaussian_pyramid_alt as gp
import math

class Sift:
	# Get numpy array of descriptors
	def get_descriptors(self):
		return self.desc_list

	# Get list of KeyPoints
	def get_keypoints(self):
		return self.all_keypoints

	# Find descriptors for each keypoint
	def find_descriptors(self, threshold=0.2, o=math.sqrt(2)):
		gauss_vector = cv2.getGaussianKernel(16, o)		
		gauss_array = np.dot(gauss_vector, np.transpose(gauss_vector))
		self.desc_list = np.empty((0, 128))
		for l in range(1 ,len(self.dog)-1):
			for o in range(len(self.dog[0])):
				desc = np.empty((len(self.key_points_struct[l-1][o]), 128))
				desc_i=0
				for kp in self.key_points_struct[l-1][o]:
					i, j = int(kp.pt[0]), int(kp.pt[1])
					image16 = self.dog[l][o][i-8:i+8, j-8:j+8]
					for w1 in range(0, 16, 4):
						for w2 in range(0, 16, 4):
							image4 = image16[w1:w1+4, w2:w2+4]
							weight = gauss_array[w1:w1+4, w2:w2+4]
							histogram = self.keypoint_histogram(image4, 8, weight, kp.angle)
							desc_j = w2*2 + 8*w1
							for h in range(len(histogram)):
								desc[desc_i, desc_j+h] = histogram[h]
					desc_i+=1
				desc_sum = np.sum(desc)
				if desc_sum != 0:
					desc = desc/desc_sum
					desc = np.where(desc>threshold, threshold, desc)
				self.desc_list = np.append(self.desc_list, desc, 0)
		
	# Creates histogram
	@staticmethod
	def keypoint_histogram(image, size=36, weight=None, kp_angle=0):
		h, l = image.shape[0], image.shape[1]
		if weight is None:
			weight = np.ones(image.shape)
		magnitude = np.zeros(image.shape)
		orientation = np.zeros(image.shape)
		histogram = np.zeros(size)
		div = 360/size
		for i in range(h-1):
			for j in range(l-1):
				x = float(image[i,j])-float(image[i+1,j])
				y = float(image[i,j])-float(image[i,j+1])
				x2 = float(image[i,j])-float(image[i+1,j])
				y2 = float(image[i,j+1])-float(image[i,j])
				magnitude[i,j] = weight[i,j]*math.sqrt(math.pow(x, 2) + math.pow(y, 2))
				orientation[i,j] = (math.degrees(math.atan2(x2, y2)) - kp_angle)%360
				histogram[int(orientation[i,j]//div)] += int(magnitude[i,j])
		return histogram	

	# Get the orientation of each key point, 
	# if some key point has more then one possible orientaion
	# then creates new key points for these orientations
	def key_orientation(self):
		self.key_points_struct = []
		self.all_keypoints = []
		for l in range(1 ,len(self.dog)-1):
			kps_line=[]
			for o in range(len(self.dog[0])):
				new_kp_list = []
				c = int(math.pow(2,l-1))
				for kp in self.key_points[l-1][o]:
					i, j = kp
					sub_image = self.dog[l][o][i-1*c:i+1*c+1, j-1*c:j+1*c+1]
					histogram = self.keypoint_histogram(sub_image)
					indx = np.argmax(histogram)
					threshold = histogram[indx]*0.8
					new_kp = cv2.KeyPoint(i*c, j*c, _size=2*c+1, _angle=indx*10, _response=1, _octave=l-1)
					new_kp_list.append(new_kp)
					self.all_keypoints.append(new_kp)
					other_keys = np.where(histogram>=threshold)
					for ok in other_keys[0]:
						if ok != indx:
							new_kp = cv2.KeyPoint(i*c, j*c, _size=2*c+1, _angle=indx*ok*10, _response=1, _octave=l-1)
							new_kp_list.append(new_kp)
							self.all_keypoints.append(new_kp)
				kps_line.append(new_kp_list) 
			self.key_points_struct.append(kps_line)

	# Find the magnitude
	@staticmethod
	def magnitude(image):
		h, l = image.shape[0], image.shape[1]
		magnitude = np.zeros(image.shape)
		for i in range(h-1):
			for j in range(l-1):
				x = float(image[i,j])-float(image[i+1,j])
				y = float(image[i,j])-float(image[i,j+1])
				magnitude[i,j] = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
		return magnitude

	# Removes points with magnitudes lower than the threshold
	def threshold (self, mag_threshold=100):	
		for l in range(1 ,len(self.dog)-1):
			for o in range(len(self.dog[0])):
				magnitude = self.magnitude(self.dog[l][o])
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
		
		for l in range(1 ,len(self.GP.images)-1):
			for o in range(len(self.GP.images[0])-1):
				Ix = cv2.filter2D(self.GP.images[l][o], -1, msk1)
				Iy = cv2.filter2D(self.GP.images[l][o], -1, msk2)
				new_kp = []
				for kp in self.key_points[l-1][o]:
					i, j = kp
					sub_Ix = Ix[i-1:i+2, j-1:j+2]
					sub_Iy = Iy[i-1:i+2, j-1:j+2]
					M = np.array([	[ float(np.sum(np.multiply(sub_Ix, sub_Ix))), float(np.sum(np.multiply(sub_Ix, sub_Iy)))],
									[ float(np.sum(np.multiply(sub_Ix, sub_Iy))), float(np.sum(np.multiply(sub_Iy, sub_Iy)))] ])
					R = M[0,0]*M[1,1] - M[0,1]*M[1,0] - k*math.pow(np.trace(M), 2)
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
					ap[indx] = 255
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
		
	def __init__(self, image, levels, img_per_pyramid, o, k):
		if levels < 3:
			print("At least 3 levels required, value given was:", levels)
			return -1
		self.GP = gp.GPyramid(image, levels, img_per_pyramid,  o, k)
		self.dog = self.diference_of_gaussian(self.GP.images, o, k)
		self.get_all_keypoints()
		self.img_key_points()

