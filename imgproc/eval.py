from collections import defaultdict
from imgproc.io import list_files, load_folder_imgs, load_rgb, open_yaml_as_dict
from imgproc.render import grid, show
from imgproc.scan import glob_entropy, glob_luminance, num_colors, dist_to_color, dist_to_mode
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
	lambda img: -dist_to_mode(img),
	dist_to_mode,
	lambda img: -dist_to_color(img, (255, 0, 0)),
	lambda img: -dist_to_color(img, (0, 255, 0)),
	lambda img: -dist_to_color(img, (0, 0, 255)),
	lambda img: -dist_to_color(img, (255, 255, 255)),
	lambda img: -dist_to_color(img, (0, 0, 0)),
]
DEF_RANKED = 1


#TODO: better pattern to handle img based funcs vs imgfile based funcs 
def rank_imgs(imgs, imgfiles, func, k=None, return_scores=False):
	if k is not None:
		if not isinstance(k, int):
			raise TypeError('k should be an integer.')
	scores = []
	for img, imgfile in zip(imgs, imgfiles):
		try:
			scores.append(func(imgfile))
		except TypeError:
			scores.append(func(img))
	rank_idxs = np.argsort(scores)[::-1]
	ranked_imgfiles = [imgfiles[ind] for ind in rank_idxs]
	if k is not None:
		ranked_imgfiles = ranked_imgfiles[:k]
		rank_idxs = rank_idxs[:k]
	if return_scores:
		scores = [scores[ind] for ind in rank_idxs]
		return ranked_imgfiles, scores, rank_idxs
	return ranked_imgfiles, rank_idxs


def rank_folder_imgs(dirpath, func, k=None, return_scores=False):
	imgs, imgfiles = load_folder_imgs(dirpath)
	return rank_imgs(imgs, imgfiles, func, k=k, return_scores=return_scores)


def rank_grid(dirpath, func, desc, k):
	ranked_imgfiles, scores, _ = rank_folder_imgs(dirpath, func, k, return_scores=True)
	imgs = [load_rgb(img) for img in ranked_imgfiles]
	print('{} :'.format(desc))
	for rank, (imgfile, score) in enumerate(zip(ranked_imgfiles, scores)):
		print('{}: {} -- score : {}'.format(rank + 1, imgfile, np.abs(score)))
	grid(imgs, warn=False)


def rank_grid_preload(imgs, imgfiles, func, desc, k):
	ranked_imgfiles, scores, rank_idxs = rank_imgs(imgs, imgfiles, func, k, return_scores=True)
	print('{} :'.format(desc))
	for rank, (imgfile, score) in enumerate(zip(ranked_imgfiles, scores)):
		print('{}: {} -- score : {}'.format(rank + 1, imgfile, np.abs(score)))
	grid([imgs[ind] for ind in rank_idxs], warn=False)


def leaderboard(dirpath, funcs=DEF_FUNCS, descs=DEF_DESCS, k=DEF_RANKED, preload=False):
	if preload:
		imgs, imgfiles = load_folder_imgs(dirpath)
	exp_name = os.path.basename(dirpath)
	print('-' * (35 + len(exp_name)))
	print('|   Leaderboard for experiment {}   |'.format(exp_name))
	print('-' * (35 + len(exp_name)) + '\n')
	for func, desc in zip(funcs, descs):
		if preload:
			rank_grid_preload(imgs, imgfiles, func, desc, k)
		else:
			rank_grid(dirpath, func, desc, k)


# assumption : all parameters subspaces are discrete
# what we should get as input (summary) :
# - paramspace (represent it in a separate YAML file ?)
# what is interesting to know :
# - which param values are dominant among the chosen ones (in proportion)
# - which param values are never chosen / which param values are always discarded
# - param -- %.2f correlation with binary target (chosen / discarded)
# - param :: value -- sampled %n times, chosen %n times (%n\% of the chosen), discarded %n times (%n\% of the discarded)
def param_analysis(dirpath, exp_name, indices, fmt='yml'):
	if fmt == 'yml':
		pspace_file = os.path.join(dirpath, '{}_paramspace.yml'.format(exp_name))
		summary_file = os.path.join(dirpath, '{}_summary.yml'.format(exp_name))
		paramspace = open_yaml_as_dict(pspace_file)
		summary = open_yaml_as_dict(summary_file)
	else:
		raise NotImplementedError
	n_exps = len(list_files(dirpath))
	pos_param_counts, neg_param_counts = [defaultdict(defaultdict(int)) for _ in range(2)]
	for i in range(n_exps):
		params = summary['graph_{}'.format(i)]['params']
		for param, value in params.items():
			if i in indices:
				pos_param_counts[param][value] += 1
			else:
				neg_param_counts[param][value] += 1
	for param, values in paramspace.items():
		correlation = param_correlation(param, indices)
		print('{} -- {.2f} correlation with binary target (chosen / discarded)'.format())
		for value in values:
			value_info = ('{} :: {} -- sampled {} times, chosen {} times ({}% of the chosen), ',
			              'discarded {} times ({}% of the discarded)')
			value_info = ''.join(value_info)
			value_count_pos = pos_param_counts[param][value]
			value_count_neg = neg_param_counts[param][value]
			value_count = value_count_pos + value_count_neg
			print(value_info.format(param, value, value_count, value_count_pos,
									value_ratio_pos, value_count_neg, value_ratio_neg))
	return pos_param_counts, neg_param_counts
