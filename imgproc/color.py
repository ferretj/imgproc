from imgproc.random import sample_from_array
from imgproc.utils import check_img_arg, hex_to_rgb, hilo, pwd
import json
import numpy as np
import os

DEF_BLACK_FILE = 'colors_black.json'
DEF_COLOR_FILE = 'colors.json'
DEF_WHITE_FILE = 'colors_white.json'


def complementary_color(rgb):
	hl = hilo(*rgb)
	return np.array([hl - c for c in rgb])


def complement(img):
	check_img_arg(img)
	h, w = img.shape[:2]
	img_mod = np.array([complementary_color(img[i, j]) for i in range(h) for j in range(w)])
	return img_mod.reshape(img.shape)


def fetch_color(name, color_file=DEF_COLOR_FILE):
	color_path = os.path.join(pwd(), color_file)
	with open(color_path, 'r') as f:
		data = json.load(f)
	for color_dict in data:
		if color_dict['name'].lower() == name.lower():
			return np.array(hex_to_rgb(color_dict['color']))
	raise ValueError('{} was not found among the available colors.'.format(name))


def fetch_random_color(color_file=DEF_COLOR_FILE):
	color_path = os.path.join(pwd(), color_file)
	with open(color_path, 'r') as f:
		data = json.load(f)  
	color_dict = sample_from_array(data)
	return np.array(hex_to_rgb(color_dict['color']))
