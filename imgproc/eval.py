from imgproc.io import list_img_files, load_rgb
from imgproc.render import grid
import numpy as np
import os
# most amount of colors
# least amount of colors
# most global entropy
# least global entropy
# brightest
# darkest
# heaviest
# lightest
# most "unified" (least mean euclidean dist between pixs and mode)
# least "unified"
# red
# blue
# white
# green


def rank_folder_imgs(dirpath, func, k=None):
	if k is not None:
		if not isinstance(k, int):
			raise TypeError('k should be an integer.')
	imgfiles = list_img_files(dirpath)
	scores = []
	for imgfile in imgfiles:
		scores.append(func(load_rgb(imgfile)))
	rank_idxs = np.argsort(scores)[::-1]
	ranked_imgfiles = [imgfiles[ind] for ind in rank_idxs]
	if k is not None:
		ranked_imgfiles = ranked_imgfiles[:k]
	return ranked_imgfiles


def rank_grid(dirpath, func, k)
	ranked_imgfiles = rank_folder_imgs(dirpath, func, k)
	imgs = [load_rgb(img) for img in ranked_imgfiles]
	grid(imgs)
	print('\nRanking :')
	for rank, imgfile in enumerate(ranked_imgfiles):
		print('{}: {}'.format(rank + 1, imgfile))
