from collections import Counter
from imgproc.utils import is_iterable, is_rgb_image, hex_to_rgb, rgb_to_hex
import numpy as np


def has_values_in_range(arr, vmin=0, vmax=256):
	m, mx = np.min(arr), np.max(arr)
	if m >= vmin and mx < vmax:
		return True
	return False


def is_pixel(obj):
	if is_iterable(obj):
		if len(obj) == 3 and has_values_in_range(obj):
			return True
	return False


def string_hash(obj):
	if is_rgb_image(obj):
		return [rgb_to_hex(obj[i, j]) for i in range(obj.shape[0]) for j in range(obj.shape[1])]
	elif is_pixel(obj):
		return rgb_to_hex(obj)
	else:
		raise TypeError('obj argument should be an image or a pixel value.')


def num_hash(obj):
	if is_rgb_image(obj):
		return obj[:, :, 0] + 256 * obj[:, :, 1] + (256 ** 2) * obj[:, :, 2]
	elif is_pixel(obj):
		return obj[0] + 256 * obj[1] + (256 ** 2) * obj[2]
	else:
		raise TypeError('obj argument should be an image or a pixel value.')


#TODO: case where background color is not the mode pixel
def background_color(im):
	h, w = im.shape[:2]
	mc_pix = Counter(string_hash(im)).most_common(1)[0][0]
	return hex_to_rgb(mc_pix)