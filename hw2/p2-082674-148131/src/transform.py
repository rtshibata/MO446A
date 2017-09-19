import cv2
import numpy as np
import math

class AffineTransf:	
	def __init__(self, list_3pt_matches):
		if len(list_3pt_matches) != 3:
			print("3 point matches required")
			return -1
		for pt_match in list_3pt_matches:
			if not isinstance(pt_match, tuple):
				print("point matches must be tuples")
				return -1
		
		
		x1=		
		X = np.matrix([ ]
			
			
				)
				
		self.A = (np.linalg(transpose(X)*X))*(transpose(X)*Y) 
