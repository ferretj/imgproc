from imgproc.gradients import linear_interp
from imgproc.utils import make_canvas
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

DEF_CELLSIZE = 6.
DEF_FIGSIZE = 10.
DEF_GRAD_DIMS = (1000, 2000)
DEF_TWOSIZE = 9.


def infer_size(dims, def_size, redux=1., constrain_max=False):
	h, w = dims
	aspect_ratio = float(w) / h
	if aspect_ratio >= 1.:
		if constrain_max:
			redux = min(redux, 1. / aspect_ratio)
		size = (redux * def_size * aspect_ratio, redux * def_size)
	else:
		if constrain_max:
			redux = min(redux, aspect_ratio)
		size = (redux * def_size, redux * def_size * (1. / aspect_ratio))
	return size


def infer_size_from_img(img, def_size, redux=1., constrain_max=False):
	return infer_size(img.shape[:2], def_size, redux=redux, constrain_max=constrain_max)


def show(img, size=None, redux=1.):
	if img.ndim == 2:
		print('WARNING: colormap applied (since displaying a two-dim image)')
	if size is None:
		size = infer_size_from_img(img, DEF_FIGSIZE, redux)
	plt.figure(figsize=size)
	plt.imshow(img)
	plt.tick_params(left=None, bottom=None, labelleft=None, labelbottom=None)
	plt.show()


def _make_grid(imgs, size, n_rows, n_cols):
	fig = plt.figure(figsize=size)
	grid = ImageGrid(fig, 111,
	                 nrows_ncols=(n_rows, n_cols),
	                 axes_pad=0.1,
	                 label_mode="1",
	                 share_all=True)
	for i in range(len(imgs)):
		grid[i].imshow(imgs[i])
	plt.tick_params(left=None, bottom=None, labelleft=None, labelbottom=None)
	plt.show()


# assumption : first two images of array have same size
#TODO: could generalize to more images (not only view two)
def view_two(imgs, size=None, redux=1.):
	n = len(imgs)
	if n == 1:
		if warn:
			print('WARNING: applying show instead since only one image.')
		show(imgs[0], size=size, redux=redux)
	else:
		if size is None:
			cell_size = infer_size_from_img(imgs[0], DEF_TWOSIZE, redux, constrain_max=True)
			size = (cell_size[0] * 2, cell_size[1])
		_make_grid(imgs[:2], size, 1, 2)


# assumption : all imgs have the same size
# assumption : at least 3 images and preferably a total that is a multiple of 3
def grid(imgs, size=None, redux=1., warn=True):
	n = len(imgs)
	if n == 1:
		if warn:
			print('WARNING: applying show instead since only one image.')
		show(imgs[0], size=size, redux=redux)
	elif n == 2:
		if warn:
			print('WARNING: applying view_two instead since only two images.')
		view_two(imgs, size=size, redux=redux)
	else:
		k = n // 3 + int(n % 3 != 0)
		if size is None:
			cell_size = infer_size_from_img(imgs[0], DEF_CELLSIZE, redux, constrain_max=True)
			size = (cell_size[0] * 3, cell_size[1] * k)
		_make_grid(imgs, size, k, 3)


# experimental : frame creations 
# ideas : write parameters inside canvas ? write img ID ?
def frame(img):
	return img


def show_grad(colors, grad_func=linear_interp, dims=DEF_GRAD_DIMS):
	colors = [np.array(col) for col in colors]
	canvas = make_canvas(dims)
	d = dims[1]
	for j in range(d):
		t = float(j) / (d - 1)
		canvas[:, j] = grad_func(colors, t)
	show(canvas)


# by default, axes origin is on the bottom left corner.
# match_image_coords displays dots with origin in top left corner and x/y-axis inverted (to match image display).
def display_dots(coords, canvas_size=None, redux=1., match_image_coords=False):
	if isinstance(coords, list):
		coords = np.array(coords)
	assert coords.ndim == 2
	assert coords.shape[1] == 2
	if canvas_size is None:
		h = np.max([x for x, _ in coords])
		w = np.max([y for _, y in coords])
		canvas_size = (h, w)
	size = infer_size(canvas_size, DEF_FIGSIZE, redux)
	plt.figure(figsize=size)
	if match_image_coords:
		plt.scatter(coords[:, 1], canvas_size[0] - coords[:, 0])
		plt.xlim(0, canvas_size[1])
		plt.ylim(0, canvas_size[0])
	else:
		plt.scatter(coords[:, 0], coords[:, 1])
		plt.xlim(0, canvas_size[0])
		plt.ylim(0, canvas_size[1])
	plt.tick_params(left=None, bottom=None, labelleft=None, labelbottom=None)
	plt.show()
