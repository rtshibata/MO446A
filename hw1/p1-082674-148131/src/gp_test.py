import gaussian_pyramid as gp
import cv2
import numpy as np
import sys

img = cv2.imread('../input/input-p1-2-1-0.png', 1)
if img is None:
	print("Image not found.")
	sys.exit()
G = gp.GPyramid(img, 2)
#Down-scaling(Subindo na piramide)
cv2.imwrite('../output/output-p1-2-2-0.png', G.access(1))
cv2.imwrite('../output/output-p1-2-2-1.png', G.access(2))
#Upscaling(DEscendo na piramide)
cv2.imwrite('../output/output-p1-2-2-3.png', G.down(G.access(2)))
