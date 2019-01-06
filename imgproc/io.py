from imgproc.utils import (check_img_arg, check_imgfile_arg, is_img_file, identify_format,
						   identify_dimensions, identify_filesize, pil_to_numpy)
import math
import matplotlib
import matplotlib.image as mplim
import numpy as np
import os
from PIL import Image
import shutil
import yaml

DEFAULT_DIR = '/Users/joh/Documents/Images/creations'


def list_files(dirpath, full_path=True):
	if not os.path.isdir(dirpath):
		raise IOError('{} does not refer to an existing folder.'.format(dirpath))
	if full_path:
		return [os.path.join(dirpath, file_) for file_ in os.listdir(dirpath)]
	else:
		return os.listdir(dirpath)


def list_img_files(dirpath, full_path=True):
	imgfiles = [file_ for file_ in list_files(dirpath, full_path) if is_img_file(file_)]
	return imgfiles


def load_folder_imgs(dirpath):
	imgfiles = list_img_files(dirpath)
	imgs = [load_rgb(imgfile) for imgfile in imgfiles]
	return imgs, imgfiles


def load_folder_imgs_from_nameparts(dirpath, nameparts):

	def has_tag_among(imgfile, nameparts):
		return any([namepart in imgfile for namepart in nameparts])

	imgfiles = [imgfile for imgfile in list_img_files(dirpath) if has_tag_among(imgfile, nameparts)]
	imgs = [load_rgb(imgfile) for imgfile in imgfiles]
	return imgs, imgfiles


def load_rgb(imgfile, show_info=False):
	check_imgfile_arg(imgfile)
	img = pil_to_numpy(Image.open(imgfile)) 
	if img.ndim == 2:
		raise ValueError('Numpy array has two dimensions only.')
	elif img.ndim == 3:
		if show_info:
			display_info(imgfile, img=img)
		if img.shape[2] == 3:
			return img
		elif img.shape[2] == 4:
			# if RGBA, we check that the transparency mask does
			# not filter anything
			mask = 255 * np.ones_like(img[..., 0])
			if np.allclose(img[..., -1], mask):
				return img[..., :3]
			else:
				raise ValueError('Detected transparency layer.')
	else:
		raise ValueError('Numpy array has more than 3 dimensions.')


def load_several_rgb(imgfiles, show_info=False):
	imgs = [load_rgb(imgfile, show_info=show_info) for imgfile in imgfiles]
	return imgs


# if img is not given, infer all from image file
def display_info(imgfile, img=None):
	check_imgfile_arg(imgfile)
	# size of file, format, dimension
	etype = imgfile.split('.')[-1].upper()
	ftype = identify_format(imgfile)
	fsize, size_descr = identify_filesize(imgfile)
	if img is not None:
		check_img_arg(img)
		fdim = img.shape[:2]
	else:
		fdim = identify_dimensions(imgfile)
	print('Extension :       {}'.format(etype))
	print('True format :     {}'.format(ftype))
	print('Dimensions :   {} x {}'.format(*fdim))
	print('Size of file : {} {}'.format(fsize, size_descr))


def default_dir(foldname=None):
	if foldname is not None:
		return os.path.join(DEFAULT_DIR, foldname)
	return DEFAULT_DIR


#TODO: write savefile check to a utils function ??
def save(img, savefile, use_default=False):
	check_img_arg(img)
	if use_default:
		savefile = os.path.join(DEFAULT_DIR, savefile)
	else:
		savedir = os.path.dirname(savefile)
		if not os.path.isdir(savedir):
			print('WARNING: creating directory {}'.format(savedir))
			os.mkdir(savedir)
	mplim.imsave(savefile, img)


def write_text(text, savefile):
	savedir = os.path.dirname(savefile)
	if not os.path.isdir(savedir):
		print('WARNING: creating directory {}'.format(savedir))
		os.mkdir(savedir)
	with open(savefile, 'w') as f:
		f.write(text)


def write_yml(dict_, savefile):
	savedir = os.path.dirname(savefile)
	if not os.path.isdir(savedir):
		print('WARNING: creating directory {}'.format(savedir))
		os.mkdir(savedir)
	with open(savefile, 'w') as f:
		yaml.dump(dict_, f, default_flow_style=False)


def erase_folder_contents(dirpath):
	if os.path.isdir(dirpath):
		shutil.rmtree(dirpath)
		os.makedirs(dirpath)


def open_yaml_as_dict(filepath):
	with open(savefile, 'w') as f:
		filedict = yaml.load(f)
	return filedict
