from imgproc.io import load_rgb, save
from imgproc.random import sample_from_discrete_dist

SERIAL_ID_LENGTH = 6


#TODO: use tmp dir or tempfile ??
#TODO: add ETA or loading info to be able to cancel 
def serigraph(imgfile, func, paramspace, size, basename, savedir):

	im = load_rgb(imgfile)
	for _ in  range(size):
		params = sample_from_discrete_dist(paramspace)
		im_mod = func(im, **params)
		savename = '_'.join([basename, hexaname(SERIAL_ID_LENGTH)])
		savepath = os.path.join(savedir, savename)
		save(im_mod, savepath) 
