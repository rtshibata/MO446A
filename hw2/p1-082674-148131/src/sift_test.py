import sift
import cv2
import os
import math

dir_path = os.path.dirname(os.path.realpath(__file__))
img = cv2.imread(dir_path+'/../input/input-p1-3-1-0.jpg', 0)
sift = sift.Sift(img, 4, 5, math.sqrt(2), 2)
if img is None:
	print("Image not found.")
	sys.exit()

output = dir_path+'/../output/output-p2-3-1-'
counter = 0
for l in range(len(sift.GP.images)):
	for o in range(len(sift.GP.images[0])):
		cv2.imwrite(output+str(counter)+'.png', sift.GP.images[l][o])
		counter = counter+1
for l in range(len(sift.dog)):
	for o in range(len(sift.dog[0])):
		cv2.imwrite(output+str(counter)+'.png', sift.dog[l][o])
		counter = counter+1
for l in range(len(sift.array_points)):
	for o in range(len(sift.array_points[0])):
		cv2.imwrite(output+str(counter)+'.png', sift.array_points[l][o]+sift.GP.images[l+1][o])
		counter = counter+1

sift.find_edges(threshold = 500, k = 0.2)
for l in range(len(sift.array_points)):
	for o in range(len(sift.array_points[0])):
		cv2.imwrite(output+str(counter)+'.png', sift.array_points[l][o]+sift.GP.images[l+1][o])
		counter = counter+1
