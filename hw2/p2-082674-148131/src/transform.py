import cv2
import numpy as np
import math

class AffineTransf:	
	def __init__(self, list_3pt_matches):
		if len(list_3pt_matches) != 6:
			print("3 points of each image required")
			return -1
		for pt_match in list_3pt_matches:
			if not isinstance(pt_match, tuple):
				print("point matches must be tuples")
				return -1
		
		x1 , y1 = list_3pt_matches[0]
		x1_, y1_ = list_3pt_matches[3]	
		x2 , y2 = list_3pt_matches[1]
		x2_, y2_ = list_3pt_matches[4]	
		x3 , y3 = list_3pt_matches[2]
		x3_, y3_ = list_3pt_matches[5]	
		X = np.matrix([x1, y1, 1, 0, 0, 0],
					  [0, 0, 0, x1, y1, 1],
  					  [x2, y2, 1, 0, 0, 0],
  					  [0, 0, 0, x2, y2, 1],
  					  [x3, y3, 1, 0, 0, 0],
  					  [0, 0, 0, x3, y3, 1])
		
		Y = np.matrix([x1_,y1_,x2_,y2_,x3_,y3_])
		Xt = X.transpose()
		product1 = np.dot(Xt,X)
		inverse1 = np.linalg(product1)
		product2 = np.dot(Xt,Y)
		A = np.dot(inverse1,product2)
		self.A = A

		
		def getA(self):
			return self.A
		
		#calcula a diferenca entre XA e Y esperado: XA - Y = erro
		def error(self,list_i_desc):
			''' matriz X é definido pelos 3 pontos associados 
			aos indices do list_i_desc
			
			em list_i_desc ha indices de 3 pontos associados (x1,y1)
			(x2,y2),(x3,y3) alem dos (x1_,y1_),(x2_,y2_),(x3_,y3_)
			
			'''
