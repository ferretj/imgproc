import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

DEF_CELLSIZE = 6.
DEF_FIGSIZE = 10.
DEF_TWOSIZE = 9.


def infer_size(img, def_size, redux=1., constrain_max=False):
	h, w = img.shape[:2]
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


def show(img, size=None, redux=1.):
	if img.ndim == 2:
		print('WARNING: colormap applied (since displaying a two-dim image)')
	if size is None:
		size = infer_size(img, DEF_FIGSIZE, redux)
	plt.figure(figsize=size)
	plt.imshow(img)
	plt.tick_params(left=None, bottom=None, labelleft=None, labelbottom=None)
	plt.show()


def _make_grid(imgs, figsize, n_rows, n_cols):
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
			cell_size = infer_size(imgs[0], DEF_TWOSIZE, redux, constrain_max=True)
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
			cell_size = infer_size(imgs[0], DEF_CELLSIZE, redux, constrain_max=True)
			size = (cell_size[0] * 3, cell_size[1] * k)
		_make_grid(imgs, size, k, 3)


# experimental : frame creations 
# ideas : write parameters inside canvas ? write img ID ?
def frame(img):
	return img
