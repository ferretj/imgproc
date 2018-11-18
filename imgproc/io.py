from imgproc.utils import pil_to_numpy
import math
import numpy as np
import os
from PIL import Image


def load_rgb(imgfile, show_info=False):
	if not isinstance(imgfile, str):
		raise TypeError('Argument must be a string.')
	img = pil_to_numpy(Image.open(imgfile)) 
	if img.ndim == 2:
		raise ValueError('Numpy array has two dimensions only')
	elif img.ndim == 3:
		if show_info:
			display_info(img, imgfile)
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


def display_info(img, imgfile):
	if not isinstance(imgfile, str):
		raise TypeError('Imgfile argument must be a string.')
	# size of file, format, dimension
	ftype = imgfile.split('.')[-1].upper()
	fsize = os.path.getsize(imgfile)
	if fsize < 10 ** 3:
		size_descr = 'B'
	elif 10 ** 3 <= fsize < 10 ** 6:
		fsize = math.ceil(fsize / 10 ** 3)
		size_descr = 'KB'
	elif 10 ** 6 <= fsize < 10 ** 9:
		fsize = math.ceil(fsize / 10 ** 6)
		size_descr = 'MB'
	else:
		raise ValueError('File size past 1GB ??')
	print('Format :       {}'.format(ftype))
	print('Dimensions :   {} x {}'.format(*img.shape[:2]))
	print('Size of file : {} {}'.format(fsize, size_descr))
