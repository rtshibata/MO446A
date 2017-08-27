import gaussian_pyramid as gp
import laplacian_pyramid as lp
import numpy as np
import cv2


def blend(left, right, mask):
	if (left.heigh() != right.heigh() or left.heigh() != mask.heigh()):
		print("Pyramids with different levels.")
		return None
	lp_result = lp.LPyramid()
	for level in range (left.heigh()):
		result_left = np.multiply(left.images[level], mask.images[level])
		result_right = np.multiply(right.images[level], np.ones(mask.images[level].shape)-mask.images[level])
		lp_result.images.append(result_left + result_right)
	return lp_result


