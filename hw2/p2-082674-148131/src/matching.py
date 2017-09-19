import sift as s
import cv2
import numpy as np
import os
import math
from scipy.spatial import distance

#usa distancia euclidiana para comparar os descritores
def cmp_desc(desc1, desc2):
	return distance.euclidian(desc1,desc2)
	
#tem q depois atualizar para excluir/atualizar a lista dos descritores a cd iteracao
def find_matches(sift1, sift2,t):
	dist_min = math.inf
	i_min = math.inf
	pts_matches = []
	list1 = sift1.desc_list[0][0]
	list2 = sift2.desc_list[0][0]
	for i_desc1 in len(sift1.desc_list[0][0]):
		for i_desc2 in len(sift2.desc_list[0][0]):
			d = cmp_desc(sift1.desc_list[0][0][i_desc1],sift1.desc_list[0][0][i_desc2]) 
			if d < dist_min and d < t:
				dist_min = d
				i_min = i_desc2
		if i_min != math.inf:
			pair = (i_desc1, i_min)
			pts_matches.append(pair)
	return pts_matches
			
				
