import cv2
import numpy as np

# Verify if the pixel in the possition (i,j) is the maximun of it's neighbours
def ismax (i, j, img, wz):
	pixel = img[i, j]
	sub_img = img[i-wz:i+wz+1, j-wz:j+wz+1]
	for line in sub_img:
		for val in line:
			if(val > pixel):
				return False	
	return True


def get_keypoints(img, thresh, blocksize=20, ksize=3, k=0.05, wz=10):
	if len(img.shape) > 2:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		gray = img
	corner = cv2.cornerHarris(gray, blocksize, ksize, k)
	corner_norm = cv2.normalize(corner, 0, 255)
	corner_norm_scaled = cv2.convertScaleAbs(corner_norm)
	points = []
	for i in range(corner_norm.shape[1]):
		for j  in range(corner_norm.shape[0]):
			if corner_norm[j,i] > thresh:
				if ismax(j,i, corner_norm, wz):
					points.append((j,i))
	return points
