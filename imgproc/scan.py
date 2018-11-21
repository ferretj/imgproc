from collections import Counter
from imgproc.utils import rgb_to_hex, hex_to_rgb
import numpy as np


#TODO: case where background color is not the mode pixel
def background_color(im):
	h, w = im.shape[:2]
	mc_pix = Counter([rgb_to_hex(im[i, j]) for i in range(h) for j in range(w)]).most_common(1)[0][0]
	return hex_to_rgb(mc_pix)