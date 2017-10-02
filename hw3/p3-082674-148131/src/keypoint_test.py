import cv2
import numpy as np
import sys
import os

NUM_IMAGES = 20
NUN_START = 20

dir_path = os.path.dirname(os.path.realpath(__file__))
output = dir_path+'/../output/output-p3-2-0-'
counter = 0
thresh = 10
for i in range(NUN_START, NUN_START+NUM_IMAGES):
	img = cv2.imread(dir_path+'/../input/dino00'+str(i)+'.png', 0)
	if img is None:
		print("Image "+str(i)+" not found.")
		sys.exit()
	corner = cv2.cornerHarris(img, 5, 3, 0.01)
	corner_norm = cv2.normalize(corner, 0, 255)
	corner_norm_scaled = cv2.convertScaleAbs(corner_norm)
	for i in range(corner_norm.shape[1]):
		for j  in range(corner_norm.shape[0]):
			if corner_norm[j,i] > thresh:
				cv2.circle(corner_norm_scaled, (i, j), 5,  255)
	cv2.imwrite(output+str(counter)+'.png', corner_norm_scaled)
	counter += 1
