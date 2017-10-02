import cv2
import numpy as np

# Funcao que implementa o algoritmo tracking the Kanade-Lucas-Tomasi
# Entrada: 
# img: imagem t=0
# img_next: image t=1
# points: lista dos keypoints
# size: tamanho(sizeXsize) da janela de vizinhos
# Saida:
# new_points: key points que nao foram removidos 
# dist: lista de tupla (u,v) igual a o optical flow.
def klt (img, img_next, points, size = 15):
	h, l = img.shape[0], img.shape[1]

	# Encontra as derivadas utilizando Sobel
	msk1 = np.array([[-1,-2,1],	[0, 0, 0], [1, 2, 1]])
	msk2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	Ix = cv2.filter2D(img, -1, msk1)
	Iy = cv2.filter2D(img, -1, msk2)
	It = img - img_next
	
	dist = []
	new_points = []
	for x,y in points:
		sub_Ix = Ix[x-size:x+size+1, y-size:y+size+1]
		sub_Iy = Iy[x-size:x+size+1, y-size:y+size+1]
		sub_It = It[x-size:x+size+1, y-size:y+size+1]

		sum_IxIx = float(np.sum(np.multiply(sub_Ix, sub_Ix)))
		sum_IxIy = float(np.sum(np.multiply(sub_Ix, sub_Iy)))
		sum_IyIy = float(np.sum(np.multiply(sub_Iy, sub_Iy)))
		M = np.array([	[sum_IxIx, sum_IxIy],
						[sum_IxIy, sum_IyIy] ])

		sum_IxIt = float(np.sum(np.multiply(sub_Ix, sub_It)))
		sum_IyIt = float(np.sum(np.multiply(sub_Iy, sub_It)))
		b = np.array([	[sum_IxIt],
						[sum_IyIt] ])
		result = np.dot(np.linalg.inv(M), b)
		u, v = -result[0], -result[1]
		# Verifica se o local previsto esta fora da imagem
		# caso nao esteja adiciona a distancia a lista
		if x+u>0 and x+u<h and y+v>0 and y+v<l:
			dist.append((u,v))
			new_points.append((x,y))
	return new_points, dist
