import sift as s
import cv2
import numpy as np
import os
import math
import matching as m
import random

#RANSAC
#https://www.cse.buffalo.edu/~jryde/lectures/cse410/MobileRobotMapping_2.html
def ransac(n_ransac, threshold, n_good_model):
	maybe_model = []
	consensus_set = []
	list3p= []
	best_error = math.inf

	for count in range(n_ransac):
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
		#pega de 3 em 3
		for index in range(len(not_inliers)//3):
			i1,j1 = not_inliers[index]
			ponto1 = sift1.key_points.struct[i1].pt
			ponto1_ = sift2.key_points.struct[j1].pt
			i2,j2 = not_inliers[index+1]
			ponto2 = sift1.key_points.struct[i2].pt
			ponto2_ = sift2.key_points.struct[j2].pt
			i3,j3 = not_inliers[index+2]
			ponto3 = sift1.key_points.struct[i3].pt
			ponto3_ = sift2.key_points.struct[j3].pt
			list_notinliers = [ponto1,ponto2,ponto3,ponto1_,ponto2_,ponto3_]
			if a.error(list_notinliers) < t:
				consensus_set.append(not_inliers[index])
				consensus_set.append(not_inliers[index+1])
				consensus_set.append(not_inliers[index+2])
	
		if len(consensus_set) > n_good_model:
			this_model = maybe_model
			sample1= randrange(0,len(consensus_set))
			sample2= randrange(0,len(consensus_set))
			sample3= randrange(0,len(consensus_set))
			i1,j1 = consensus_set[sample1]
			ponto1 = sift1.key_points.struct[i1].pt
			ponto1_ = sift2.key_points.struct[j1].pt
			i2,j2 = consensus_set[sample2]
			ponto2 = sift1.key_points.struct[i2].pt
			ponto2_ = sift2.key_points.struct[j2].pt
			i3,j3 = consensus_set[sample3]
			ponto3 = sift1.key_points.struct[i3].pt
			ponto3_ = sift2.key_points.struct[j3].pt
			list_sample = [ponto1,ponto2,ponto3,ponto1_,ponto2_,ponto3_]
			this_error = a.error(list_sample)
			if this_error < best_error:
				best_model = this_model
				best_consensus_set = consensus_set
				best_error = this_error
	
	return best_model, best_error, best_consensus_set		

