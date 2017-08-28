import gaussian_pyramid as gp
import cv2
import numpy as np
import sys

img = cv2.imread('../input/input-p1-2-2-0.jpg', 1)
if img is None:
	print("Image not found.")
	sys.exit()
G = gp.GPyramid(img, 4)
#Down-scaling(Subindo na piramide)
cv2.imwrite('../output/output-p1-2-2-0.png', G.access(2))
cv2.imwrite('../output/output-p1-2-2-1.png', G.access(3))
cv2.imwrite('../output/output-p1-2-2-2.png', G.access(4))
#Upscaling(DEscendo na piramide)
cv2.imwrite('../output/output-p1-2-2-3.png', G.down(G.access(3)))

