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


# assumption : images share same size
def build_grid(imgs, n_rows, n_cols):
	assert isinstance(imgs, list)
	assert n_rows * n_cols >= len(imgs)
	h, w = imgs[0].shape[:2]
	grid_size = (h * n_rows, w * n_cols)
	grid = make_canvas(grid_size)
	for i in range(h):
		for j in range(w):
			curr = i * n_cols + j
			if curr >= len(imgs):
				break
			grid[i * h: (i + 1) * h, j * w: (j + 1) * w] = imgs[curr]
	return grid
