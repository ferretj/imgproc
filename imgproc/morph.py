from copy import deepcopy
from imgproc.utils import numpy_to_pil, pil_to_numpy
from PIL import ImageEnhance, ImageFilter, ImageOps


def modify_vividness(img, factor):
	modifier = ImageEnhance.Color(numpy_to_pil(img))
	img_mod = modifier.enhance(factor)
	return pil_to_numpy(img_mod)


def modify_brightness(img, factor):
	modifier = ImageEnhance.Brightness(numpy_to_pil(img))
	img_mod = modifier.enhance(factor)
	return pil_to_numpy(img_mod)


def modify_contrast(img, factor, optimize=False):
	if optimize:
		img_mod = ImageOps.autocontrast(numpy_to_pil(img))
	else:
		modifier = ImageEnhance.Contrast(numpy_to_pil(img))
		img_mod = modifier.enhance(factor)
	return pil_to_numpy(img_mod)


def to_grayscale(img):
	img_mod = ImageOps.grayscale(numpy_to_pil(img))
	return pil_to_numpy(img_mod, from_grayscale=True)


def to_negative(img):
	img_mod = ImageOps.invert(numpy_to_pil(img))
	return pil_to_numpy(img_mod)


def flip(img, side='ud'):
	if side == 'ud':
		return img[::-1]
	elif side == 'rl':
		return img[:, ::-1]
	else:
		err = "Side argument takes either 'ud' or 'rl' as a value"
		raise ValueError(err)


def blur(img, radius=2.):
	f = ImageFilter.GaussianBlur(radius)
	img_mod = numpy_to_pil(img).filter(f)
	return pil_to_numpy(img_mod)


# if (xmod = 2, ymod = 0) then xstart = 2, xend = h
# TODO: test on real example
def translate(img, xmod, ymod, mode='mirror'):
	h, w = img.shape[:2]
	img_mod = np.zeros_like(img).astype(np.uint8)
	# copy of the image
	xstart, xend, nxstart, nxend = max(0, -xmod), min(h, h - xmod), max(0, xmod), min(h, h + xmod)
	ystart, yend, nystart, nyend = max(0, -ymod), min(w, w - ymod), max(0, ymod), min(w, w + ymod)
	img_mod[nxstart: nxend, nystart: nyend] = img[xstart: xend, ystart: yend]
	# processing part
	if mode == 'zero':
		img_mod[:nxstart] = 0
		img_mod[nxend:] = 0
		img_mod[:, :nystart] = 0
		img_mod[:, nyend:] = 0
	elif mode == 'ones':
		img_mod[:nxstart] = 255
		img_mod[nxend:] = 255
		img_mod[:, :nystart] = 255
		img_mod[:, nyend:] = 255
	elif mode == 'mirror':
		img_mod[:nxstart] = img[:nxstart:-1]
		img_mod[nxend:] = img[nxend::-1]
		img_mod[:, :nystart] = img[:, :nystart:-1]
		img_mod[:, nyend:] = img[:, nyend::-1]
	else:
		return NotImplementedError
	return img_mod


def to_channel(img, channel='r'):
	if channel.lower() == 'r':
		img_mod = deepcopy(img)
		img_mod[:, :, 1:] = 0
	elif channel.lower() == 'g':
		img_mod = deepcopy(img)
		img_mod[:, :, ::2] = 0
	elif channel.lower() == 'b':
		img_mod = deepcopy(img)
		img_mod[:, :, :2] = 0
	else:
		err = 'channel argument must be one of `r`, `g` or `b`.'
		raise ValueError(err)
	return img_mod
