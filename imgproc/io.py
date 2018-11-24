from imgproc.utils import pil_to_numpy
import math
import matplotlib
import numpy as np
import os
from PIL import Image


def load_rgb(imgfile, show_info=False):
	if not isinstance(imgfile, str):
		raise TypeError('Argument imgfile must be a string.')
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


# sometimes extensions are wrong indicators of the true nature of the file
# information is hidden in file signature
# see http://www.libpng.org/pub/png/spec/1.2/PNG-Rationale.html#R.PNG-file-signature
# and https://en.wikipedia.org/wiki/JPEG_File_Interchange_Format
def identify_format(imgfile):
	with open(imgfile, 'rb') as f:
		fline = f.next()
		if fline.startswith('ffd8'):
			ftype = 'JPG'
		elif fline.startswith('8950'):
			ftype = 'PNG'
		else:
			ftype = None
	if ftype is None:
		raise IOError('Unrecognized file format.')
	return ftype


def display_info(img, imgfile):
	if not isinstance(imgfile, str):
		raise TypeError('Argument imgfile must be a string.')
	# size of file, format, dimension
	etype = imgfile.split('.')[-1].upper()
	ftype = identify_format(imgfile)
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
	print('Extension :       {}'.format(etype))
	print('True format :     {}'.format(ftype))
	print('Dimensions :   {} x {}'.format(*img.shape[:2]))
	print('Size of file : {} {}'.format(fsize, size_descr))


def save(img, savefile):
	savedir = os.path.dirname(savefile)
	if not os.path.isdir(savedir):
		print('WARNING: creating directory {}'.format(savedir))
		os.mkdir(savedir)
	matplotlib.image.imsave(savefile, img)
