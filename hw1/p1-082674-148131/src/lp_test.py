import laplacian_pyramid as lp
import cv2
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
img = cv2.imread(dir_path+'/../input/input-p1-2-3-0.jpeg', 1)
if img is None:
	print("Image not found.")
	sys.exit()
#Generates a Laplacian Pyramid with level 3
L = lp.LPyramid(img, 3)

#The laplacian images for the level 0 and 1
cv2.imwrite(dir_path+'/../output/output-p1-2-3-0.png', L.images[0])
cv2.imwrite(dir_path+'/../output/output-p1-2-3-1.png', L.images[1])
#The gaussian image at the top level 2
cv2.imwrite(dir_path+'/../output/output-p1-2-3-2.png', L.images[2])
#The Gaussian images found for levels 1 and 0
cv2.imwrite(dir_path+'/../output/output-p1-2-3-3.png', L.access(1))
cv2.imwrite(dir_path+'/../output/output-p1-2-3-4.png', L.access(0))
