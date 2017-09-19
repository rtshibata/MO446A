import sift as s
import cv2
import numpy as np
import os
import math
import matching as m
import random

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
t_semelhanca = n_pixels//2
list_index_desc = m.find_matches(sift1,sift2,t_semelhanca)
#print(list_index_desc)

#RANSAC
n_ransac = len(list_index_desc)//3 #3 sem criterio
threshold = 10 #numero sem criterio
good_model =  len(list_index_desc)//2 #mais de 50% ok
maybe_model = []
consensus_set = []
list3p= []
best_error = math.inf

for count in range(n_ransac)
	list3p = list_index_desc[3*count:3*(count+1)]
	consensus_set = list3p
	not_inliers = list_index_desc[:3*count] + list_index_desc[3*(count+1):]
	i1, j1 = list3p[0]
	ponto1 = sift1.key_points.struct[i1].pt
	ponto1_ = sift2.key_points.struct[j1].pt
	i2, j2 = list3p[1]
	ponto2 = sift1.key_points.struct[i2].pt
	ponto2_ = sift2.key_points.struct[j2].pt
	i3, j3 = list3p[2]
	ponto3 = sift1.key_points.struct[i3].pt
	ponto3_ = sift2.key_points.struct[j3].pt
	#selects possible inliers	
	list3p = [ponto1,ponto2,ponto3,ponto1_,ponto2_,ponto3_]
	#affine transformation
	a = AffineTransf(list3p)
	#gets possible model
	maybe_model = a.getA()
	## qdo tiver menos de 3 descritores em not_inliers, cancelar o teste de erro
	## !!!!!!!!!
	###!!!!!!!!
	for index_desc in not_inliers:
		if a.error(list_index_desc) < t:
			consensus_set.append(index_desc)
	
	if len(consensus_set) > good_model:
		this_model = maybe_model
		sample1= randrange(0,len(consensus_set))
		sample2= randrange(0,len(consensus_set))
		sample3= randrange(0,len(consensus_set))
		list_sample = [sample1,sample2,sample3]
		this_error = a.error(list_sample)
		if this_error < best_error:
			best_model = this_model
			best_consensus_set = consensus_set
			best_error = this_error		
	#https://www.cse.buffalo.edu/~jryde/lectures/cse410/MobileRobotMapping_2.html
