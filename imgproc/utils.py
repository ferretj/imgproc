import numpy as np
from PIL import Image


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


def is_iterable(obj):
	try:
		iterator = iter(obj)
		return True
	except TypeError:
		return False


def rgb_to_hex(rgb):
	if not is_iterable(rgb):
		raise TypeError('rgb argument should be an iterable.')
	elif len(rgb) != 3:
		raise IndexError('rgb argument should contain three integers.')
	rgb = tuple(int(c) for c in rgb)
	return '#%02x%02x%02x' % rgb


def hex_to_rgb(hexa):
	if not isinstance(hexa, str):
		raise TypeError('hexa argument should be a string.')
	hexa = hexa.lstrip('#')
	if len(hexa) != 6:
		err = ('hexa argument should be 6 chars long ',
			   '(not counting `#` at the beginning).')
		raise IndexError(''.join(err))
	return tuple(int(hexa[i : i + 2], 16) for i in (0, 2 ,4))
