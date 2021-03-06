#
# Testing the OpenCV implemetation of KLT
#


import numpy as np
import cv2
import sys
import os
import sfm

dir_path = os.path.dirname(os.path.realpath(__file__))
output = dir_path+'/../output/output-p3-3-0-'
cap = cv2.VideoCapture(dir_path+'/../input/input-p3-2-0-0.mp4')
if cap is None:
	print("Video not found.")
	sys.exit()	

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))


i=0
frame, gray = None, None
ret = True
points = []
dists = []
while(ret):
	if i == 0:
		# Take first frame and find corners in it
		ret, old_frame = cap.read()
		old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
		p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
		# Create a mask image for drawing purposes
		mask = np.zeros_like(old_frame)
		points.append(p0)
	else:
		ret,frame = cap.read()
		if ret is True:	
			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			# calculate optical flow
			p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
			points.append(p1)
			# Select good points
			good_new = p1[st==1]
			good_old = p0[st==1]
			# draw the tracks
			#for i,(new,old) in enumerate(zip(good_new,good_old)):
				#a,b = new.ravel()
				#c,d = old.ravel()
				#mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
				#frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
			#img = cv2.add(frame,mask)
			#cv2.imshow('frame',img)
			# Now update the previous frame and previous points
			old_gray = frame_gray.copy()
			p0 = good_new.reshape(-1,1,2)
	i+=1
cv2.destroyAllWindows()
cap.release()

for i in range(len(points)):
	points[i] = points[i][st==1]

sfm.SfM(points)

