from collections import Counter
from imgproc.utils import (check_img_arg, check_color, hex_to_rgb,
						   img_to_2d_num_hash, img_to_luminance, string_hash)
import numpy as np
from scipy.stats import entropy


#TODO: case where background color is not the mode pixel
def background_color(img):
	check_img_arg(img)
	h, w = img.shape[:2]
	mode = Counter(string_hash(img)).most_common(1)[0][0]
	return hex_to_rgb(mode)


def glob_entropy(img):

	def to_color_probas(img):
		check_img_arg(img)
		n = np.prod(img.shape[:2])
		imh = img_to_2d_num_hash(img)
		counts = Counter(np.ravel(imh))
		color_probas = [float(count) // n for _, count in counts.items()]
	
	color_probas = to_color_probas(img)
	return entropy(color_probas)


def num_colors(img):
	return len(np.unique(np.ravel(img_to_2d_num_hash(img))))


def glob_luminance(img):
	return np.mean(img_to_luminance(img))


def avg_dist_to_color(img, col):
	check_color(col)
	h, w = img.shape[:2]
	return np.mean([np.linalg.norm(img[i, j] - mode) for i in range(h) for j in range(w)])


def avg_dist_to_mode(img):
	mode = background_color(img)
	return avg_dist_to_color(img, mode)
