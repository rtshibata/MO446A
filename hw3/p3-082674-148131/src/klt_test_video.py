import cv2
import numpy as np
import sys
import os
import klt
import keypoint_selector as kps

def tuplesum (tl1, tl2):
	new_tl = []
	for i in range(len(tl1)):
		new_tl.append((tl1[i][0]+tl2[i][0], tl1[i][1]+tl2[i][1]))
	return new_tl

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

dir_path = os.path.dirname(os.path.realpath(__file__))
output = dir_path+'/../output/output-p3-3-0-'
cap = cv2.VideoCapture(dir_path+'/../input/input-p3-2-0-0.mp4')
if cap is None:
	print("Video not found.")
	sys.exit()	
i=0
frame, gray = None, None
ret = True
points = []
dists = []
while(ret):
	if i > 0:
		frame_prev = frame
	ret, frame = cap.read()
	if ret is True:	
		if i == 0:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			p = cv2.goodFeaturesToTrack(gray, mask = None, **feature_params)
			points.append(p)	
			print("# of keypoints found: ", len(p))
			# Create a mask image for drawing purposes
			mask = np.zeros_like(frame)
			col = None
			for kn in range(len(p)):
				if col is None:	col = np.array([np.random.randint(0,255,3)])
				else:
					col = np.append(col, np.array([np.random.randint(0,255,3)]), axis=0)
			# Save the image with keypoints
			img_out = frame
			for indx, line in enumerate(p):
				a,b = line.ravel()
				img_out = cv2.circle(img_out,(a,b),5,col[indx].tolist(),-1)
			cv2.imwrite(output+str(i)+'.png', img_out)
		else:
			new_points, rmv_points = klt.klt(frame_prev, frame, points[i-1])
			if new_points is not None:
				col = col[rmv_points]
				points.append(new_points)
				img_out = frame
				for indx,(new,old) in enumerate(zip(new_points, points[i-1][rmv_points])):
					a,b = new.ravel()
					c,d = old.ravel()
					a,b,c,d = int(a), int(b), int(c), int(d),
					mask = cv2.line(mask, (a,b),(c,d), col[indx].tolist(), 2)
					img_out = cv2.circle(img_out,(a,b),5,col[indx].tolist(),-1)
				img_out = cv2.add(img_out,mask)
				cv2.imwrite(output+str(i)+'.png', img_out)
			else:
				ret = False
		i+=1
cap.release()
