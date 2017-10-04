import cv2
import numpy as np

# Funcao que implementa o algoritmo tracking the Kanade-Lucas-Tomasi
# Entrada: 
# img: imagem t=0
# img_next: image t=1
# points: lista dos keypoints
# size: tamanho(sizeXsize) da janela de vizinhos
# img_thresh: a distancia a borda que keypoints sao permitidos chegarem
#				caso estejam a uma distancia menor que img_thresh são removidos
# Saida:
# new_points: os novos pontos
# rmv_points: array mostrando quais pontos foram removidos e quais nao.
def klt (img, img_next, points, window_size = 15, img_thresh = 20):
	h, l = img.shape[0], img.shape[1]
	xt = h - img_thresh
	yt = l - img_thresh


	# Encontra as derivadas utilizando Sobel
	msk1 = np.array([	[-1,-2,-1],	
						[ 0, 0, 0], 
						[ 1, 2, 1]])
	msk2 = np.array([	[-1, 0, 1], 
						[-2, 0, 2], 
						[-1, 0, 1]])
	Ix = cv2.filter2D(img, -1, msk1)
	Iy = cv2.filter2D(img, -1, msk2)
	It = img_next - img
	
	new_points = None
	rmv_points = None
	for pi in points:
		x, y = pi.ravel()
		x, y = int(x), int(y)
		result = np.array([[x, y]])
		# Assumindo que img[x,y] == img_next[x+u, y+v]
		size = window_size//2
		sub_Ix = Ix[x-size:x+size+1, y-size:y+size+1]
		sub_Iy = Iy[x-size:x+size+1, y-size:y+size+1]
		sub_It = It[x-size:x+size+1, y-size:y+size+1]

		sum_IxIx = float(np.sum(np.multiply(sub_Ix, sub_Ix)))
		sum_IxIy = float(np.sum(np.multiply(sub_Ix, sub_Iy)))
		sum_IyIy = float(np.sum(np.multiply(sub_Iy, sub_Iy)))
		M = np.array([	[sum_IxIx, sum_IxIy],
						[sum_IxIy, sum_IyIy] ])

		# Verifica se é possivel inverter a matriz M, 
		# caso nao seja, faz com que d e v sejam 0.
		if (sum_IxIx != 0 or sum_IxIy != 0 or sum_IyIy != 0) and np.linalg.det(M) != 0:
			sum_IxIt = float(np.sum(np.multiply(sub_Ix, sub_It)))
			sum_IyIt = float(np.sum(np.multiply(sub_Iy, sub_It)))
			b = np.array([	[-sum_IxIt],
							[-sum_IyIt] ])
			d = np.dot(np.linalg.inv(M), b)
			result = result+d.transpose()

		# Verifica se o local previsto esta fora da imagem
		# caso nao esteja adiciona a distancia a lista
		if result[0][0]>img_thresh and result[0][0]<xt and result[0][1]>img_thresh and result[0][1]<yt:
			if new_points is None:
				new_points = result
			else:
				new_points = np.append(new_points, result, axis=0)
			if rmv_points is None: rmv_points = np.array([True])
			else:	rmv_points = np.append(rmv_points, np.array([True]))
		else:
			if rmv_points is None: rmv_points = np.array([False])
			else:	rmv_points = np.append(rmv_points, np.array([False]))
	return new_points, rmv_points
