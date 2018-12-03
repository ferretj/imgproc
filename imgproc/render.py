import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

DEF_CELLSIZE = 6.
DEF_FIGSIZE = 10.


def show(img, size=None, redux=1.):
	if img.ndim == 2:
		print('WARNING: colormap applied (since displaying a two-dim image)')
	h, w = img.shape[:2]
	if size is None:
		aspect_ratio = float(w) / h
		size = (redux * DEF_FIGSIZE, redux * (DEF_FIGSIZE / aspect_ratio))
	plt.figure(figsize=size)
	plt.imshow(img)
	plt.tick_params(left=None, bottom=None, labelleft=None, labelbottom=None)
	plt.show()


# assumption : square images
def grid(imgs, size=None, redux=1.):
	n = len(imgs)
	k = n // 3 + int(n % 3 != 0)
	if imgs[0].shape[0] != imgs[0].shape[1]:
		raise ValueError('Images are not square.')
	if size is None:
		size = (redux * DEF_CELLSIZE * 3, redux * DEF_CELLSIZE * k)
	fig = plt.figure(figsize=size)
	grid = ImageGrid(fig, 111,
	                 nrows_ncols=(k, 3),
	                 axes_pad=0.1,
	                 label_mode="1",
	                 share_all=True)
	for i, img in enumerate(imgs):
		grid[i].imshow(img)
	plt.tick_params(left=None, bottom=None, labelleft=None, labelbottom=None)
	plt.show()
