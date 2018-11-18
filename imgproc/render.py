import matplotlib.pyplot as plt

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

