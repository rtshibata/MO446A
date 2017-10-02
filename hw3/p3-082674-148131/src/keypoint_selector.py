import cv2
import numpy as np

def get_keypoints(img, threshold):
	thresh = threshold
	corner = cv2.cornerHarris(img, 5, 3, 0.01)
	corner_norm = cv2.normalize(corner, 0, 255)
	corner_norm_scaled = cv2.convertScaleAbs(corner_norm)
	points = []
	for i in range(corner_norm.shape[0]):
		for j  in range(corner_norm.shape[1]):
			if corner_norm[i, j] > thresh:
				points.append((i, j))
	return points
