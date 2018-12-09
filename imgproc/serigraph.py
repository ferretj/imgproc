from imgproc.io import erase_folder_contents, load_rgb, save, write_text
from imgproc.random import random_hex, sample_from_dict
import os
from timeit import default_timer as timer

SERIAL_ID_LENGTH = 6


def serigraph_summary(savenames, all_params, times):

	def _summarize_graph(savename, params, time, text=''):
		text += 'Img {} :\n'.format(savename)
		text += '-' * (len(savename) + 6) + '\n'
		for param, val in params.items():
			text += param + ': ' + str(val) + '\n'
		text += '\nProcessed in {} s'.format(int(time))
		return text

	assert len(all_params) == len(savenames)
	assert len(all_params) == len(times)
	summ = ''
	for (savename, params, time) in zip(savenames, all_params, times):
		summ = _summarize_graph(savename, params, time, text=summ)
		summ += '\n\n'
	return summ


#TODO: use tmp dir or tempfile ??
#TODO: add ETA or loading info to be able to cancel 
def serigraph_mod(imgfile, func, paramspace, size, basename, savedir, budget=None, erase_previous=False, make_summary=False):
	if size is None and budget is None:
		raise ValueError('Need to specify either size or budget.')
	elif budget is not None:
		size = 10000
	if erase_previous:
		erase_folder_contents(savedir)
	im = load_rgb(imgfile)
	chrono = 0
	savenames = []
	all_params = []
	times = []
	for _ in  range(size):
		t = timer()
		params = sample_from_dict(paramspace)
		im_mod = func(im, **params)
		savename = '_'.join([basename, random_hex(SERIAL_ID_LENGTH)]) + '.png' 
		savepath = os.path.join(savedir, savename)
		timelapse = timer() - t
		chrono += timelapse
		savenames.append(savename)
		all_params.append(params)
		times.append(timelapse)
		save(im_mod, savepath)
		if budget is not None:
			if chrono > budget:
				print('Went past allocated time.')
				break

	if make_summary:
		summ = serigraph_summary(savenames, all_params, times)
		summfile = '_'.join([basename, 'summary']) + '.txt'
		summpath = os.path.join(savedir, summfile)
		write_text(summ, summpath)



#TODO: use tmp dir or tempfile ??
#TODO: add ETA or loading info to be able to cancel
def serigraph_gen(func, paramspace, basename, savedir, size=None, budget=None, erase_previous=False, make_summary=False):
	if size is None and budget is None:
		raise ValueError('Need to specify either size or budget.')
	elif budget is not None:
		size = 10000
	if erase_previous:
		erase_folder_contents(savedir)
	chrono = 0
	savenames = []
	all_params = []
	times = []
	for _ in  range(size):
		t = timer()
		params = sample_from_dict(paramspace)
		print(params)
		im_mod = func(**params)
		savename = '_'.join([basename, random_hex(SERIAL_ID_LENGTH)]) + '.png'
		savepath = os.path.join(savedir, savename)
		timelapse = timer() - t
		chrono += timelapse
		savenames.append(savename)
		all_params.append(params)
		times.append(timelapse)
		save(im_mod, savepath)
		if budget is not None:
			if chrono > budget:
				print('Went past allocated time.')
				break

	if make_summary:
		summ = serigraph_summary(savenames, all_params, times)
		summfile = '_'.join([basename, 'summary']) + '.txt'
		summpath = os.path.join(savedir, summfile)
		write_text(summ, summpath)
