from imgproc.io import load_rgb, save
from imgproc.random import random_hex, sample_from_dict
import os

SERIAL_ID_LENGTH = 6


#TODO: use tmp dir or tempfile ??
#TODO: add ETA or loading info to be able to cancel 
def serigraph_mod(imgfile, func, paramspace, size, basename, savedir):
	im = load_rgb(imgfile)
	for _ in  range(size):
		params = sample_from_dict(paramspace)
		im_mod = func(im, **params)
		savename = '_'.join([basename, random_hex(SERIAL_ID_LENGTH)])
		savepath = os.path.join(savedir, savename)
		save(im_mod, savepath) 


#TODO: use tmp dir or tempfile ??
#TODO: add ETA or loading info to be able to cancel
def serigraph_gen(func, paramspace, size, basename, savedir):
	for _ in  range(size):
		params = sample_from_dict(paramspace)
		im_mod = func(**params)
		savename = '_'.join([basename, random_hex(SERIAL_ID_LENGTH)])
		savepath = os.path.join(savedir, savename)
		save(im_mod, savepath) 
