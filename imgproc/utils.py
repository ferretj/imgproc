import imghdr
import imagesize
import math
import numpy as np
import os
from PIL import Image
import shutil

GOLD_NB = (1. + math.sqrt(5)) / 2
HEX_CHARS = [str(d) for d in range(10)] + list('abcdef')
IMG_EXTS = ['jpg', 'png']
JPG_ALIASES = ['jpg', 'jpeg', 'JPG', 'JPEG']
PNG_ALIASES = ['png', 'PNG']


def numpy_to_pil(img):
	if img.dtype != 'uint8':
		warning = ('WARNING: Numpy array was automatically',
				   'converted to uint8 (dtype was {})')
		print(' '.join(warning).format(img.dtype)) 
		img = np.uint8(255 * img)
	pil_img = Image.fromarray(img)
	return pil_img


def pil_to_numpy(pil_img, from_grayscale=False):
	img = np.array(pil_img)
	if from_grayscale:
		img = np.repeat(img[..., np.newaxis], 3, axis=2)
	m, mx = np.min(img), np.max(img)
	if m < 0 or mx > 255:
		raise ValueError('Pixel values should be between 0 and 255 (included).')
	img = img.astype(np.uint8)
	return img


def has_values_in_range(arr, vmin=0, vmax=256):
	m, mx = np.min(arr), np.max(arr)
	if m >= vmin and mx < vmax:
		return True
	return False


def is_color(obj):
	if is_iterable(obj):
		if len(obj) == 3 and has_values_in_range(obj):
			return True
	return False


def is_pixel(obj):
	return is_color(obj)


def is_iterable(obj):
	try:
		iterator = iter(obj)
		return True
	except TypeError:
		return False


def is_img_file(file_):
	try:
		check_imgfile_arg(file_)
	except:
		return False
	return True


def is_rgb_image(obj):
	if hasattr(obj, 'dtype'):
		if obj.ndim == 3 and obj.dtype == 'uint8':
			return True
	return False


def identify_dimensions(imgfile):
	return imagesize.get(imgfile)


def identify_filesize(imgfile):
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
	return fsize, size_descr


# sometimes extensions are wrong indicators of the true nature of the file
# information is hidden in file signature
# see http://www.libpng.org/pub/png/spec/1.2/PNG-Rationale.html#R.PNG-file-signature
# and https://en.wikipedia.org/wiki/JPEG_File_Interchange_Format
def identify_format(imgfile):
	# with open(imgfile, 'rb') as f:
	# 	fline = f.next()
	# 	if fline.startswith('ffd8'):
	# 		ftype = 'JPG'
	# 	elif fline.startswith('8950'):
	# 		ftype = 'PNG'
	# 	else:
	# 		ftype = None
	# if ftype is None:
	# 	raise IOError('Unrecognized file format.')
	return imghdr.what(imgfile).upper()


def rgb_to_hex(rgb):
	if not is_iterable(rgb):
		raise TypeError('rgb argument should be an iterable.')
	elif len(rgb) != 3:
		raise IndexError('rgb argument should contain three integers.')
	rgb = tuple(int(c) for c in rgb)
	return '#%02x%02x%02x' % rgb


# converts 2-sized hexadecimal to number
# ex : ff -> 255
def hex_to_num(hexa):
	assert len(hexa) == 2
	return int(hexa, 16)


def hex_to_rgb(hexa):
	if not isinstance(hexa, str):
		raise TypeError('hexa argument should be a string.')
	hexa = hexa.lstrip('#')
	if len(hexa) != 6:
		err = ('hexa argument should be 6 chars long ',
			   '(not counting `#` at the beginning).')
		raise IndexError(''.join(err))
	return tuple(hex_to_num(hexa[i: i + 2]) for i in (0, 2 ,4))


def check_color(col, to_numpy=False):
	if not is_color(col):
		raise TypeError('Invalid color argument.')
	if to_numpy:
		return np.array(col)


def check_pixel(obj, to_numpy=False):
	col = check_color(obj, to_numpy=to_numpy)
	if to_numpy:
		return col


#TODO: add PIL format into account if needed
def check_img_arg(img):
	if not is_rgb_image(img):
		raise TypeError('Invalid image argument.')


def check_imgfile_arg(imgfile, ftype=None):
	if not isinstance(imgfile, str):
		raise TypeError('Argument imgfile must be a string.')
	elif not os.path.isfile(imgfile):
		raise IOError('Argument {} does not point towards an existing file.'.format(imgfile))
	fmt = identify_format(imgfile)
	if ftype is not None:
		if ftype in JPG_ALIASES:
			if fmt not in JPG_ALIASES:
				err = ('Image file format ({}) does not correspond',
					   ' to the one checked against ({})')
				raise IOError(''.join(err).format(fmt, ftype))
		elif ftype in PNG_ALIASES:
			if fmt not in PNG_ALIASES:
				err = ('Image file format ({}) does not correspond',
					   ' to the one checked against ({})')
				raise IOError(''.join(err).format(fmt, ftype))
		else:
			raise NotImplementedError('Only .jpg and .png support at the moment.')


# no assumption on length, only on the nature of characters
def check_hexadecimal(obj):

	def is_hexa_char(c):
		if not isinstance(c, str):
			raise TypeError('The argument must be a single character (not str here).')
		elif len(c) != 1:
			raise IndexError('The argument must be a single character (several here).')
		elif c in HEX_CHARS:
			return True
		return False

	if not all([is_hexa_char(c) for c in obj]):
		raise ValueError('Not strictly hexadecimal.')


def make_canvas(can_size, fill_value=255):
	if is_color(fill_value):
		fill_value = np.array(fill_value)[np.newaxis, np.newaxis, :]
		return np.tile(fill_value, can_size + (1,)).astype(np.uint8)
	elif isinstance(fill_value, int):
		return np.full(can_size + (3,), fill_value, dtype=np.uint8)
	else:
		raise TypeError('`fill_value` is supposed to be a RGB color array or an integer pixel value.')


def make_patch(patch_size, color):
	tiling_size = tuple(patch_size) + (1,)
	return np.tile(color, tiling_size)


# see https://stackoverflow.com/questions/123198/how-do-i-copy-a-file-in-python
def make_filecopy(f, return_dest=False):
	basedir = os.path.dirname(f)
	destname = os.path.basename(f).replace('.', '_copy.')
	dest = os.path.join(basedir, destname)
	shutil.copy2(f, dest)
	if return_dest:
		return dest


def hilo(a, b, c):
    if c < b: b, c = c, b
    if b < a: a, b = b, a
    if c < b: b, c = c, b
    return a + c


def hex_chars():
	return HEX_CHARS


def string_hash(obj):
	if is_rgb_image(obj):
		return [rgb_to_hex(obj[i, j]) for i in range(obj.shape[0]) for j in range(obj.shape[1])]
	elif is_pixel(obj):
		return rgb_to_hex(obj)
	else:
		raise TypeError('obj argument should be an image or a pixel value.')


def num_hash(obj):
	if is_rgb_image(obj):
		return obj[:, :, 0] + 256 * obj[:, :, 1] + (256 ** 2) * obj[:, :, 2]
	elif is_pixel(obj):
		return obj[0] + 256 * obj[1] + (256 ** 2) * obj[2]
	else:
		raise TypeError('obj argument should be an image or a pixel value.')


def num_hash_to_rgb(obj):
	if not isinstance(obj, int):
		raise TypeError('Input should be an integer.')
	return np.array([obj % 256, (obj % (256 ** 2)) // 256 , obj // (256 ** 2)])


def img_to_2d_num_hash(img):
	check_img_arg(img)
	return num_hash(img)


# see https://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color
def img_to_luminance(img):
	check_img_arg(img)
	return 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]


# path to imgproc base folder
# see https://stackoverflow.com/questions/4934806/how-can-i-find-scripts-directory-with-python
def pwd():
	return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


# return indices of all elements that are not in `coll`
def all_not_in(elems, coll):
	return [elem for elem in elems if elem not in coll]


# return indices of all elements that are not in `coll`
def index_all_not_in(elems, coll):
	return [i for i, elem in enumerate(elems) if elem not in coll]


def deg_to_rad(d):
	return (math.pi / 180) * d
