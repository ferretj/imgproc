from copy import deepcopy
from imgproc.utils import make_canvas
import numpy as np


def add_white_frame(img, frame_height, frame_width):
	fh, fw = frame_height // 2, frame_width // 2
	size = img.shape[0] + frame_height, img.shape[1] + frame_width
	framed_img = make_canvas(size)
	framed_img[fh: size[0] - fh, fw: size[1] - fw] = img
	return framed_img


def width_border(img, frame_width):
	fw = frame_width // 2
	size = img.shape[:2]
	framed_img = deepcopy(img)
	framed_img[:, 0: fw] = np.array([255, 255, 255])
	framed_img[:, size[1] - fw:] = np.array([255, 255, 255])
	return framed_img


def width_reduce(img, frame_width):
	fw = frame_width // 2
	size = img.shape[:2]
	framed_img = deepcopy(img[:, fw: size[1] - fw])
	return framed_img
