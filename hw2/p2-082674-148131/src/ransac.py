import sift as s
import cv2
import numpy as np
import os
import math
import matching as m
import random
import transform

#RANSAC
def ransac_affine(sift1,sift2,n_ransac, threshold, n_good_model,list_index_desc):
	if n_ransac >= len(list_index_desc):
		print("Numero N deve ser menor que o numero dos pares de pontos matched dos descritores\n")
		return -1
	best_consensus_set = []
	best_model = np.zeros((6,1))
	maybe_model = []
	consensus_set = []
	list3p= []
	best_error = float('inf')

	keypoints1 = sift1.get_keypoints()
	keypoints2 = sift2.get_keypoints()
		
	for count in range(n_ransac):
		if 3*count >= len(list_index_desc):
			break
		elif len(list_index_desc[3*count:3*(count+1)]) < 3:
			break

		list3p = list_index_desc[3*count:3*(count+1)]
		consensus_set = list3p
		not_inliers = list_index_desc[:3*count] + list_index_desc[3*(count+1):]
		i1, j1 = list3p[0]
		ponto1 = keypoints1[i1].pt
		ponto1_ = keypoints2[j1].pt
		i2, j2 = list3p[1]
		ponto2 = keypoints1[i2].pt
		ponto2_ = keypoints2[j2].pt
		i3, j3 = list3p[2]
		ponto3 = keypoints1[i3].pt
		ponto3_ = keypoints2[j3].pt
		
		#selects possible inliers	
		list3p = [ponto1,ponto2,ponto3,ponto1_,ponto2_,ponto3_]
		#affine transformation
		a = transform.AffineTransf(list3p)
		#gets possible model
		maybe_model = a.getA()
		#pega de 3 em 3
		for index in range(len(not_inliers)//3):
			i1,j1 = not_inliers[index]
			ponto1 = keypoints1[i1].pt
			ponto1_ = keypoints2[j1].pt
			i2,j2 = not_inliers[index+1]
			ponto2 = keypoints1[i2].pt
			ponto2_ = keypoints2[j2].pt
			i3,j3 = not_inliers[index+2]
			ponto3 = keypoints1[i3].pt
			ponto3_ = keypoints2[j3].pt
			list_notinliers = [ponto1,ponto2,ponto3,ponto1_,ponto2_,ponto3_]
			if a.error(list_notinliers) < threshold:
				consensus_set.append(not_inliers[index])
				consensus_set.append(not_inliers[index+1])
				consensus_set.append(not_inliers[index+2])
	
		if len(consensus_set) > n_good_model:
			this_model = maybe_model
			sample1= random.randrange(0,len(consensus_set))
			sample2= random.randrange(0,len(consensus_set))
			sample3= random.randrange(0,len(consensus_set))
			i1,j1 = consensus_set[sample1]
			ponto1 = keypoints1[i1].pt
			ponto1_ = keypoints2[j1].pt
			i2,j2 = consensus_set[sample2]
			ponto2 = keypoints1[i2].pt
			ponto2_ = keypoints2[j2].pt
			i3,j3 = consensus_set[sample3]
			ponto3 = keypoints1[i3].pt
			ponto3_ = keypoints2[j3].pt
			list_sample = [ponto1,ponto2,ponto3,ponto1_,ponto2_,ponto3_]
			this_error = a.error(list_sample)
			if this_error < best_error:
				best_model = this_model
				best_consensus_set = consensus_set
				best_error = this_error
	
	return best_model, best_error, best_consensus_set

def ransac_project(sift1,sift2,n_ransac, threshold, n_good_model,list_index_desc):		
	if n_ransac >= len(list_index_desc):
		print("Numero N deve ser menor que o numero dos pares de pontos matched dos descritores\n")
		return -1

	best_consensus_set = []
	best_model = np.zeros((8,1))
	maybe_model = []
	consensus_set = []
	list4p= []
	best_error = float('inf')

	for count in range(n_ransac):
		if 4*count >= len(list_index_desc):
			break
		elif len(list_index_desc[4*count:4*(count+1)]) < 3:
			break

		list4p = list_index_desc[4*count:4*(count+1)]
		consensus_set = list4p
		not_inliers = list_index_desc[:4*count] + list_index_desc[4*(count+1):]
		i1, j1 = list4p[0]
		ponto1 = keypoints1[i1].pt
		ponto1_ = keypoints2[j1].pt
		i2, j2 = list4p[1]
		ponto2 = keypoints1[i2].pt
		ponto2_ = keypoints2[j2].pt
		i3, j3 = list4p[2]
		ponto3 = keypoints1[i3].pt
		ponto3_ = keypoints2[j3].pt
		i4, j4 = list4p[3]
		ponto4 = keypoints1[i4].pt
		ponto4_ = keypoints2[j4].pt
		#selects possible inliers	
		list4p = [ponto1,ponto2,ponto3,ponto4,ponto1_,ponto2_,ponto3_,ponto4_]
		#affine transformation
		a = transform.AffineTransf(list4p)
		#gets possible model
		maybe_model = a.getA()
		#pega de 4 em 4
		for index in range(len(not_inliers)//4):
			i1,j1 = not_inliers[index]
			ponto1 = keypoints1[i1].pt
			ponto1_ = keypoints2[j1].pt
			i2,j2 = not_inliers[index+1]
			ponto2 = keypoints1[i2].pt
			ponto2_ = keypoints2[j2].pt
			i3,j3 = not_inliers[index+2]
			ponto3 = keypoints1[i3].pt
			ponto3_ = keypoints2[j3].pt
			i4,j4 = not_inliers[index+3]
			ponto4 = keypoints1[i4].pt
			ponto4_ = keypoints2[j4].pt
			list_notinliers = [ponto1,ponto2,ponto3,ponto4,ponto1_,ponto2_,ponto3_,ponto4_]
			if a.error(list_notinliers) < threshold:
				consensus_set.append(not_inliers[index])
				consensus_set.append(not_inliers[index+1])
				consensus_set.append(not_inliers[index+2])
				consensus_set.append(not_inliers[index+3])
	
		if len(consensus_set) > n_good_model:
			this_model = maybe_model
			sample1= random.randrange(0,len(consensus_set))
			sample2= random.randrange(0,len(consensus_set))
			sample3= random.randrange(0,len(consensus_set))
			sample4= random.randrange(0,len(consensus_set))
			i1,j1 = consensus_set[sample1]
			ponto1 = keypoints1[i1].pt
			ponto1_ = keypoints2[j1].pt
			i2,j2 = consensus_set[sample2]
			ponto2 = keypoints1[i2].pt
			ponto2_ = keypoints2[j2].pt
			i3,j3 = consensus_set[sample3]
			ponto3 = keypoints1[i3].pt
			ponto3_ = keypoints2[j3].pt
			i4,j4 = not_inliers[index+3]
			ponto4 = keypoints1[i4].pt
			ponto4_ = keypoints2[j4].pt
			list_sample = [ponto1,ponto2,ponto3,ponto4,ponto1_,ponto2_,ponto3_,ponto4_]
			this_error = a.error(list_sample)
			if this_error < best_error:
				best_model = this_model
				best_consensus_set = consensus_set
				best_error = this_error
	
	return best_model, best_error, best_consensus_set

