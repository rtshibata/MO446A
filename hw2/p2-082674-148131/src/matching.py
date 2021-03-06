import cv2
import numpy as np
import os
import math
from scipy.spatial import distance


# usa distancia euclidiana para comparar os descritores
def cmp_desc(desc1, desc2):
	return distance.euclidean(desc1,desc2)
	
# tem q depois atualizar para excluir/atualizar a lista dos descritores a cd iteracao
def find_matches(descriptor1, descriptor2,t):
	pts_matches = []
	chosen_list = [False for i in range(descriptor2.shape[0])]
	for i_desc1 in range(descriptor1.shape[0]):
		dist_min = float('inf')
		i_min = float('inf')
#		dist_min = math.inf
#		i_min = math.inf
		for i_desc2 in range(descriptor2.shape[0]):
			if chosen_list[i_desc2] == False:
				d = cmp_desc(descriptor1[i_desc1,:], descriptor2[i_desc2,:]) 
				if d < dist_min and d < t:
					dist_min = d
					i_min = i_desc2
#		if i_min != math.inf:
		if i_min != float('inf'):
			chosen_list[i_min] = True
			pair = (i_desc1, i_min)
			pts_matches.append(pair)
	return pts_matches
			
				
