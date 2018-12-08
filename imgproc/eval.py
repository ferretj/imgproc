from imgproc.io import list_img_files, load_rgb
from imgproc.render import grid, show
from imgproc.scan import glob_entropy, glob_luminance, num_colors, avg_dist_to_color, avg_dist_to_mode
from imgproc.utils import identify_filesize
import numpy as np
import os

DEF_DESCS = [
	'Most amount of colors',
	'Least amount of colors',
	'Most entropy',
	'Least entropy',
	'Brightest',
	'Darkest',
	'Heaviest',
	'Lightest',
	'Most color variance',
	'Least color variance',
	'Red dominance',
	'Blue dominance',
	'White dominance',
	'Green dominance',
	'Black dominance',
]
DEF_FUNCS = [
	num_colors,
	lambda img: -num_colors(img),
	glob_entropy,
	lambda img: -glob_entropy(img),
	glob_luminance,
	lambda img: -glob_luminance(img),
	lambda imgfile: identify_filesize(imgfile)[0],
	lambda imgfile: -identify_filesize(imgfile)[0],
	lambda img: -avg_dist_to_mode(img),
	avg_dist_to_mode,
	lambda img: -avg_dist_to_color(img, (255, 0, 0)),
	lambda img: -avg_dist_to_color(img, (0, 255, 0)),
	lambda img: -avg_dist_to_color(img, (0, 0, 255)),
	lambda img: -avg_dist_to_color(img, (255, 255, 255)),
	lambda img: -avg_dist_to_color(img, (0, 0, 0)),
]
DEF_RANKED = 1


#TODO: better pattern to handle img based funcs vs imgfile based funcs 
def rank_folder_imgs(dirpath, func, k=None, return_scores=False):
	if k is not None:
		if not isinstance(k, int):
			raise TypeError('k should be an integer.')
	imgfiles = list_img_files(dirpath)
	scores = []
	for imgfile in imgfiles:
		try:
			scores.append(func(imgfile))
		except TypeError:
			scores.append(func(load_rgb(imgfile)))
	rank_idxs = np.argsort(scores)[::-1]
	ranked_imgfiles = [imgfiles[ind] for ind in rank_idxs]
	if k is not None:
		ranked_imgfiles = ranked_imgfiles[:k]
		rank_idxs = rank_idxs[:k]
	if return_scores:
		scores = [scores[ind] for ind in rank_idxs]
		return ranked_imgfiles, scores
	return ranked_imgfiles


def rank_grid(dirpath, func, desc, k):
	ranked_imgfiles, scores = rank_folder_imgs(dirpath, func, k, return_scores=True)
	imgs = [load_rgb(img) for img in ranked_imgfiles]
	print('{} :'.format(desc))
	for rank, (imgfile, score) in enumerate(zip(ranked_imgfiles, scores)):
		print('{}: {} -- score : {}'.format(rank + 1, imgfile, np.abs(score)))
	grid(imgs, warn=False)


def leaderboard(dirpath, funcs=DEF_FUNCS, descs=DEF_DESCS, k=DEF_RANKED):
	exp_name = os.path.basename(dirpath)
	print('-' * (35 + len(exp_name)))
	print('|   Leaderboard for experiment {}   |'.format(exp_name))
	print('-' * (35 + len(exp_name)) + '\n')
	for func, desc in zip(funcs, descs):
		rank_grid(dirpath, func, desc, k)
