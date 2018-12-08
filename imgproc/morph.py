from copy import deepcopy
from imgproc.scan import background_color, num_hash
from imgproc.utils import numpy_to_pil, pil_to_numpy
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


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
def translate(img, xmod, ymod, fill='background', fill_value=None):
	h, w = img.shape[:2]
	img_mod = np.zeros_like(img).astype(np.uint8)
	# translating the image
	xstart, xend, nxstart, nxend = max(0, -xmod), min(h, h - xmod), max(0, xmod), min(h, h + xmod)
	ystart, yend, nystart, nyend = max(0, -ymod), min(w, w - ymod), max(0, ymod), min(w, w + ymod)
	img_mod[nxstart: nxend, nystart: nyend] = img[xstart: xend, ystart: yend]
	# processing missing values in translated image
	if fill_value is not None:
		img_mod[:nxstart] = fill_value
		img_mod[nxend:] = fill_value
		img_mod[:, :nystart] = fill_value
		img_mod[:, nyend:] = fill_value
	elif fill == 'black':
		img_mod[:nxstart] = 0
		img_mod[nxend:] = 0
		img_mod[:, :nystart] = 0
		img_mod[:, nyend:] = 0
	elif fill == 'white':
		img_mod[:nxstart] = 255
		img_mod[nxend:] = 255
		img_mod[:, :nystart] = 255
		img_mod[:, nyend:] = 255
	elif fill == 'background':
		bg_color = np.array(background_color(im)) 
		img_mod[:nxstart] = bg_color
		img_mod[nxend:] = bg_color
		img_mod[:, :nystart] = bg_color
		img_mod[:, nyend:] = bg_color
	elif fill == 'mirror':
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


def map_pixval(img, pix_start, pix_end):
	img_mod = deepcopy(img)
	imgh, pval = num_hash(img_mod), num_hash(pix_start)
	img_mod[imgh == pval] = pix_end
	return img_mod


#TODO: replace PIL functions by own functions !!
def pixelate(img):
	img_mod = numpy_to_pil(img)
	img_size = img_mod.size

	# boost saturation of image 
	sat_booster = ImageEnhance.Color(img_mod)
	img_mod = sat_booster.enhance(float(kwargs.get("saturation", 1.25)))

	# increase contrast of image
	contr_booster = ImageEnhance.Contrast(img_mod)
	img_mod = contr_booster.enhance(float(kwargs.get("contrast", 1.2)))

	# reduce the number of colors used in picture
	img_mod = img_mod.convert('P', palette=Image.ADAPTIVE, colors=int(kwargs.get("n_colors", 10)))

	# reduce image size
	superpixel_size = int(kwargs.get("superpixel_size", 10))
	reduced_size = (img_size[0] // superpixel_size, img_size[1] // superpixel_size)
	img_mod = img_mod.resize(reduced_size, Image.BICUBIC)

	# resize to original shape to give pixelated look
	img_mod = img_mod.resize(img_size, Image.BICUBIC)
	
	return img_mod


# similar to numpy sort but with an additional key argument
# key can either be a function or a 1D array / 2D matrix with values
def sort(img, key, order='increasing'):
	if callable(key):
		try:
			vals = key(img)
		except:
			h, w = img.shape[:2]
			vals = np.array([key(img[i, j]) for i in range(h) for j in range(w)])
	elif type(key) is np.ndarray:
		vals = key
	
	if vals.ndim == 1:
		indices = np.argsort(vals)
	elif vals.ndim == 2:
		indices = np.argsort(np.ravel(vals))
	else:
		raise ValueError('Indices for sorting should be a 1d or 2d array.')
	
	if order == 'reverse' or order == 'decreasing':
		indices = indices[::-1]

	img_mod = deepcopy(img).reshape(img.shape[0] * img.shape[1], img.shape[2])
	img_mod = img.mod[indices].reshape(img.shape[0], img.shape[1], img.shape[2])
	
	return img_mod


# from https://twitter.com/kGolid/status/1060841706105507840
def pixelsort(img):
	# select some starting pixel pst
	# draw randomly from neighborhood of processed pixels
	# pick the farthest from pst -> P
	# draw randomly from unprocessed
	# pick the most similar to the processed neighbor of P ->
	# swap P and S
	h, w = img.shape[:2]
	img_mod = deepcopy(img)
	x_start, y_start = np.random.randint(h), np.random.randint(w)
	orig = (x_start, y_start)
	proc = set(orig)
	unproc = set([(i, j) for i in range(h) for j in range(w)]) - proc
	neigh = set((x_start - 1, y_start),
				(x_start + 1, y_start),
				(x_start, y_start - 1),
				(x_start, y_start + 1))
	sources = sample_from_set(neigh, size=4)
	src = sources[np.argmax([dist(cd, orig) for cd in sources])]
	dests = sample_from_set(unproc, size=4)
	dst = dests[np.argmax([sim(cd, src) for cd in dests])]
	img_mod[src[0], src[1]], img_mod[dst[0], dst[1]] = img_mod[dst[0], dst[1]], img_mod[src[0], src[1]]
	proc.add(src)
	unproc.remove(src)
