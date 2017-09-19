import sift as s
import cv2
import numpy as np
import os
import math
import matching as m

dir_path = os.path.dirname(os.path.realpath(__file__))
img = cv2.imread(dir_path+'/../input/input-p2-3-1-0.jpg', 0)
#img = cv2.imread(dir_path+'/../input/input-p2-3-2-1.jpg', 0)
sift1 = s.Sift(img, 3, 2, math.sqrt(2), 2)
if img is None:
	print("Image not found.")
	sys.exit()

output = dir_path+'/../output/output-p2-3-1-'
counter = 0
for l in range(len(sift1.GP.images)):
	for o in range(len(sift1.GP.images[0])):
		cv2.imwrite(output+str(counter)+'.png', sift1.GP.images[l][o])
		counter = counter+1
for l in range(len(sift1.dog)):
	for o in range(len(sift1.dog[0])):
		cv2.imwrite(output+str(counter)+'.png', sift1.dog[l][o])
		counter = counter+1
for l in range(len(sift1.array_points)):
	for o in range(len(sift1.array_points[0])):
		cv2.imwrite(output+str(counter)+'.png', sift1.array_points[l][o]+sift1.GP.images[l+1][o])
		counter = counter+1
sift1.threshold(200)
for l in range(len(sift1.array_points)):
	for o in range(len(sift1.array_points[0])):
		cv2.imwrite(output+str(counter)+'.png', sift1.array_points[l][o]+sift1.GP.images[l+1][o])
		counter = counter+1
sift1.find_edges(threshold = 50, k = 0.2)
for l in range(len(sift1.array_points)):
	for o in range(len(sift1.array_points[0])):
		cv2.imwrite(output+str(counter)+'.png', sift1.array_points[l][o]+sift1.GP.images[l+1][o])
		counter = counter+1
sift1.key_orientation()
for l in range(1, len(sift1.dog)-1):
	for o in range(len(sift1.dog[0])):
		img_out = np.empty(sift1.GP.images[l][o].shape)
		cv2.drawKeypoints(sift1.GP.images[l][o], sift1.key_points_struct[l-1][o], img_out, 3)
		cv2.imwrite(output+str(counter)+'.png', img_out)
		counter = counter+1
sift1.get_descriptors()

img = cv2.imread(dir_path+'/../input/input-p1-3-1-01.jpg', 0)
#img = cv2.imread(dir_path+'/../input/input-p1-3-2-1.jpg', 0)
sift2 = s.Sift(img, 3, 2, math.sqrt(2), 2)
if img is None:
	print("Image not found.")
	sys.exit()
for l in range(len(sift2.GP.images)):
	for o in range(len(sift2.GP.images[0])):
		cv2.imwrite(output+str(counter)+'.png', sift2.GP.images[l][o])
		counter = counter+1
for l in range(len(sift2.dog)):
	for o in range(len(sift2.dog[0])):
		cv2.imwrite(output+str(counter)+'.png', sift2.dog[l][o])
		counter = counter+1
for l in range(len(sift2.array_points)):
	for o in range(len(sift2.array_points[0])):
		cv2.imwrite(output+str(counter)+'.png', sift2.array_points[l][o]+sift2.GP.images[l+1][o])
		counter = counter+1
sift2.threshold(200)
for l in range(len(sift2.array_points)):
	for o in range(len(sift2.array_points[0])):
		cv2.imwrite(output+str(counter)+'.png', sift2.array_points[l][o]+sift2.GP.images[l+1][o])
		counter = counter+1
sift2.find_edges(threshold = 50, k = 0.2)
for l in range(len(sift2.array_points)):
	for o in range(len(sift2.array_points[0])):
		cv2.imwrite(output+str(counter)+'.png', sift2.array_points[l][o]+sift2.GP.images[l+1][o])
		counter = counter+1
sift2.key_orientation()
for l in range(1, len(sift2.dog)-1):
	for o in range(len(sift2.dog[0])):
		img_out = np.empty(sift2.GP.images[l][o].shape)
		cv2.drawKeypoints(sift2.GP.images[l][o], sift2.key_points_struct[l-1][o], img_out, 3)
		cv2.imwrite(output+str(counter)+'.png', img_out)
		counter = counter+1
sift2.get_descriptors()
print("numero de pontos de interesse de cada img\n  ")
print(len(sift1.desc_list[0][0]))
print(len(sift2.desc_list[0][0]))

#match hypothesis
n_pixels,_ = sift1.dog[0][0].shape
t_semelhanca = n_pixels/2
list_points = m.find_matches(sift1,sift2,t_semelhanca)

#RANSAC

