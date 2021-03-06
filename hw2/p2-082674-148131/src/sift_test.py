import sift as s
import cv2
import numpy as np
import os
import sys
import math
import matching as m
import random
import transform
import ransac as ran

dir_path = os.path.dirname(os.path.realpath(__file__))
img_orig1 = cv2.imread(dir_path+'/../input/input-p2-3-1-0.png', 1)
if img_orig1 is None:
	print("Image 1 not found.")
	sys.exit()
img1 = cv2.cvtColor(img_orig1, cv2.COLOR_BGR2GRAY)
if img1 is None:
	print("Converting image 1 error")
	sys.exit()
sift1 = s.Sift(img1, 9, 2, math.sqrt(2), 2)

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
sift1.find_edges(threshold = -16, k = 0.2)
for l in range(len(sift1.array_points)):
	for o in range(len(sift1.array_points[0])):
		cv2.imwrite(output+str(counter)+'.png', sift1.array_points[l][o]+sift1.GP.images[l+1][o])
		counter = counter+1
sift1.threshold(50)
for l in range(len(sift1.array_points)):
	for o in range(len(sift1.array_points[0])):
		cv2.imwrite(output+str(counter)+'.png', sift1.array_points[l][o]+sift1.GP.images[l+1][o])
		counter = counter+1
sift1.key_orientation()
img_out = img_orig1
cv2.drawKeypoints(img1, sift1.get_keypoints(), img_out, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite(output+str(counter)+'.png', img_out)
counter = counter+1
sift1.find_descriptors()

img_orig2 = cv2.imread(dir_path+'/../input/input-p2-3-1-1.png', 1)
if img_orig2 is None:
	print("Image 2 not found.")
	sys.exit()
img2 = cv2.cvtColor(img_orig2, cv2.COLOR_BGR2GRAY)
if img2 is None:
	print("Converting image 2 error")
	sys.exit()
sift2 = s.Sift(img2, 9, 2, math.sqrt(2), 2)

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
sift2.find_edges(threshold =-16, k = 0.2)
for l in range(len(sift2.array_points)):
	for o in range(len(sift2.array_points[0])):
		cv2.imwrite(output+str(counter)+'.png', sift2.array_points[l][o]+sift2.GP.images[l+1][o])
		counter = counter+1
sift2.threshold(50)
for l in range(len(sift2.array_points)):
	for o in range(len(sift2.array_points[0])):
		cv2.imwrite(output+str(counter)+'.png', sift2.array_points[l][o]+sift2.GP.images[l+1][o])
		counter = counter+1 	
sift2.key_orientation()
img_out = img_orig2
cv2.drawKeypoints(img2, sift2.get_keypoints(), img_out, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite(output+str(counter)+'.png', img_out)
counter = counter+1

sift2.find_descriptors()

print("numero de pontos de interesse de cada img\n  ")
print(len(sift1.get_descriptors()))
print(len(sift2.get_descriptors()))

#match hypothesis
n_pixels,_ = sift1.dog[0][0].shape
t_semelhanca = n_pixels//2
list_index_desc = m.find_matches(sift1.get_descriptors(),sift2.get_descriptors(),t_semelhanca)
print(len(list_index_desc))

#RANSAC
n_ransac = len(list_index_desc)//3 #3 sem criterio
threshold = 1000 #numero sem criterio
n_good_model =  len(list_index_desc)//2 #mais de 50% ok

A,error,new_list_index_desc = ran.ransac_affine(sift1,sift2,n_ransac, threshold, n_good_model,list_index_desc)
#A,error,new_list_index_desc = ran.ransac_project(sift1,sift2,n_ransac, threshold, n_good_model,list_index_desc)

#transform using all the matches, generalization of solving XA = Y
final_A = transform.AffineTransf.final_transform(new_list_index_desc)
#final_A = transform.ProjectTransf.final_transform(new_list_index_desc)

