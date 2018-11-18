from imgproc.utils import numpy_to_pil, pil_to_numpy
from PIL import ImageEnhance, ImageFilter, ImageOps


def modify_vividness(img, factor):
	modifier = ImageEnhance.Color(numpy_to_pil(img))
	im_mod = modifier.enhance(factor)
	return pil_to_numpy(im_mod)


def modify_brightness(img, factor):
	modifier = ImageEnhance.Brightness(numpy_to_pil(img))
	im_mod = modifier.enhance(factor)
	return pil_to_numpy(im_mod)


def modify_contrast(img, factor, optimize=False):
	if optimize:
		im_mod = ImageOps.autocontrast(numpy_to_pil(img))
	else:
		modifier = ImageEnhance.Contrast(numpy_to_pil(img))
		im_mod = modifier.enhance(factor)
	return pil_to_numpy(im_mod)


def to_grayscale(img):
	im_mod = ImageOps.grayscale(numpy_to_pil(img))
	return pil_to_numpy(im_mod, from_grayscale=True)


def to_negative(img):
	im_mod = ImageOps.invert(numpy_to_pil(img))
	return pil_to_numpy(im_mod)


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
	im_mod = numpy_to_pil(img).filter(f)
	return pil_to_numpy(im_mod)
