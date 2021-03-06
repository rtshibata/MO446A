import pyramid_blending as pb
import gaussian_pyramid as gp
import laplacian_pyramid as lp
import numpy as np
import cv2
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
# Apple
imgleft = cv2.imread(dir_path+'/../input/input-p1-2-4-0.jpg', 1)
if imgleft is None:
	print("Image 1 not found.")
	sys.exit()
# Orange
imgright = cv2.imread(dir_path+'/../input/input-p1-2-4-1.jpg', 1)
if imgright is None:
	print("Image 2 not found.")
	sys.exit()
if imgleft.shape != imgright.shape:
	print("Images is not the same sizes")
	sys.exit()
height, length = imgleft.shape[0], imgleft.shape[1]
blend_mask = np.ones((height, length//2, 1))
blend_mask = np.concatenate((blend_mask, np.zeros((height, length-length//2, 1))), axis=1)
gp_mask = gp.GPyramid(blend_mask, 5)
lp_left = lp.LPyramid(imgleft, 5)
lp_right = lp.LPyramid(imgright, 5)

lp_result = pb.blend(lp_left, lp_right, gp_mask)
#The Gaussian images of the result
cv2.imwrite(dir_path+'/../output/output-p1-2-4-0.png', lp_result.access(4))
cv2.imwrite(dir_path+'/../output/output-p1-2-4-1.png', lp_result.access(2))
cv2.imwrite(dir_path+'/../output/output-p1-2-4-2.png', lp_result.access(0))


# Guardians
imgleft = cv2.imread(dir_path+'/../input/input-p1-2-4-2.png', 1)
if imgleft is None:
	print("Image 3 not found.")
	sys.exit()
# Star Wars
imgright = cv2.imread(dir_path+'/../input/input-p1-2-4-3.png', 1)
if imgright is None:
	print("Image 4 not found.")
	sys.exit()
if imgleft.shape != imgright.shape:
	print("Images is not the same sizes")
	sys.exit()
height, length = imgleft.shape[0], imgleft.shape[1]
blend_mask = np.ones((height, length//2, 1))
blend_mask = np.concatenate((blend_mask, np.zeros((height, length-length//2, 1))), axis=1)
gp_mask = gp.GPyramid(blend_mask, 5)
lp_left = lp.LPyramid(imgleft, 5)
lp_right = lp.LPyramid(imgright, 5)

lp_result = pb.blend(lp_left, lp_right, gp_mask)
cv2.imwrite(dir_path+'/../output/output-p1-2-4-3.png', lp_result.access(0))

# Pre-made mask
blend_mask = cv2.imread(dir_path+'/../input/input-p1-2-4-4.jpg', 1)
if imgright is None:
	print("Image 4 not found.")
	sys.exit()
blend_mask = blend_mask/255
gp_mask = gp.GPyramid(blend_mask, 5)

lp_result = pb.blend(lp_left, lp_right, gp_mask)
cv2.imwrite(dir_path+'/../output/output-p1-2-4-4.png', lp_result.access(0))

# Sun set
imgleft = cv2.imread(dir_path+'/../input/input-p1-2-4-5.png', 1)
if imgleft is None:
	print("Image 3 not found.")
	sys.exit()
# Space
imgright = cv2.imread(dir_path+'/../input/input-p1-2-4-6.jpg', 1)
if imgright is None:
	print("Image 4 not found.")
	sys.exit()
if imgleft.shape != imgright.shape:
	print("Images is not the same sizes")
	sys.exit()
height, length = imgleft.shape[0], imgleft.shape[1]
blend_mask = np.ones((height//2, length, 1))
blend_mask = np.concatenate((blend_mask, np.zeros((height-height//2, length, 1))), axis=0)
gp_mask = gp.GPyramid(blend_mask, 5)
lp_left = lp.LPyramid(imgleft, 5)
lp_right = lp.LPyramid(imgright, 5)
lp_result = pb.blend(lp_left, lp_right, gp_mask)
cv2.imwrite(dir_path+'/../output/output-p1-2-4-5.png', lp_result.access(0))
