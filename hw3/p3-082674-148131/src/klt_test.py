import cv2
import numpy as np
import sys
import os
import klt
import keypoint_selector as kps

dir_path = os.path.dirname(os.path.realpath(__file__))
output = dir_path+'/../output/output-p3-3-0-'
cap = cv2.VideoCapture(dir_path+'/../input/input-p3-2-0-0.mp4')
if cap is None:
	print("Video not found.")
	sys.exit()	
i=0
frame, gray = None, None
points = []
dists = []
while(cap.isOpened()):
	if i > 0:
		frame_prev, gray_prev = frame, gray	
	ret, frame = cap.read()
	if ret is True:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if i > 0:
			p = kps.get_keypoints(gray_prev, 15)
			#print(len(points))
			new_points, dist = klt.klt(gray_prev, gray, p)
			#Imprime saida para cada 20 imagens vista (só para nao ter muitos imagens salvas)
			if (i-1)%20 == 0:
				img_out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
				for z in range(len(new_points)):
					x,y = new_points[z]
					u,v = new_points[z][0]+int(dist[z][0]), new_points[z][1]+int(dist[z][1])
					# Ponto inicial eh verde, final azul
					# Na maioria dos casos o inicio e fim sao o mesmo ponto
					img_out[x,y] = (0,255,0)
					img_out[u,v] = (255,0,0)
				cv2.imwrite(output+str(i-1)+'.png', img_out)
			points.append(new_points)
			dists.append(dist)
		i+=1
	else:
		break
cap.release()
cv2.destroyAllWindows()
