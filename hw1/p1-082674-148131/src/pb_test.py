import pyramid_blending as pb
import gaussian_pyramid as gp
import laplacian_pyramid as lp
import numpy as np
import cv2
import sys

# Apple
imgleft = cv2.imread('../input/input-p1-2-4-0.jpg', 1)
if imgleft is None:
	print("Image 1 not found.")
	sys.exit()
# Orange
imgright = cv2.imread('../input/input-p1-2-4-1.jpg', 1)
if imgright is None:
	print("Image 2 not found.")
	sys.exit()
if imgleft.shape != imgright.shape:
	print("Images is not the same sizes")
	sys.exit()
heigh, length = imgleft.shape[0], imgleft.shape[1]
blend_mask = np.ones((heigh, length//2, 1))
blend_mask = np.concatenate((blend_mask, np.zeros((heigh, length-length//2, 1))), axis=1)
gp_mask = gp.GPyramid(blend_mask, 5)
lp_left = lp.LPyramid(imgleft, 5)
lp_right = lp.LPyramid(imgright, 5)

#The top level of each pyramid
cv2.imwrite('../output/output-p1-2-4-0.png', lp_left.images[0])
cv2.imwrite('../output/output-p1-2-4-1.png', lp_right.images[0])
cv2.imwrite('../output/output-p1-2-4-2.png', lp_left.access(4))
cv2.imwrite('../output/output-p1-2-4-3.png', lp_right.access(4))
lp_result = pb.blend(lp_left, lp_right, gp_mask)
cv2.imwrite('../output/output-p1-2-4-4.png', lp_result.access(4))
cv2.imwrite('../output/output-p1-2-4-5.png', lp_result.access(2))
cv2.imwrite('../output/output-p1-2-4-6.png', lp_result.access(0))
