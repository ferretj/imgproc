from imgproc.random import sample_from_array
from imgproc.utils import hex_to_rgb, pwd
import json
import numpy as np
import os

DEF_GRAD_FILE = 'gradients_ui.json'


def linear_interp(colors, t):
	n_colors = len(colors)
	interval_bounds = np.linspace(0., 1., n_colors)
	idx = np.searchsorted(interval_bounds, t)
	idx += int(idx == 0)
	scol, tcol = colors[idx - 1], colors[idx]
	t = (t - interval_bounds[idx - 1]) * (n_colors - 1)
	return np.uint8((1. - t) * scol + t * tcol)


def quad_interp(colors, t):
	n_colors = len(colors)
	interval_bounds = np.linspace(0., 1., n_colors)
	idx = np.searchsorted(interval_bounds, t)
	idx += int(idx == 0)
	scol, tcol = colors[idx - 1], colors[idx]
	t = (t - interval_bounds[idx - 1]) * (n_colors - 1)
	return np.uint8((1. - t ** 2) * scol + (t ** 2) * tcol)


class ColorGradient:

	def __init__(self, colors, mixfunc=linear_interp, name=None):
		self.colors = colors
		self.mixfunc = mixfunc
		self.name = name

	@classmethod
	def from_json_dict(cls, jdict, mixfunc=linear_interp):
		name = jdict['name']
		colors = [np.array(hex_to_rgb(col)) for col in jdict['colors']]
		return cls(colors, mixfunc, name)

	@property
	def num_colors(self):
		return len(self.colors)

	# t is a float in [0, 1]
	def get_color(self, t):
		return self.mixfunc(self.colors, t)

	def sample_color(self):
		return sample_from_array(self.colors)


def fetch_gradient(name, grad_file=DEF_GRAD_FILE):
	grad_path = os.path.join(pwd(), grad_file)
	with open(grad_path, 'r') as f:
		data = json.load(f)
	for grad_dict in data:
		if grad_dict['name'].lower() == name.lower():
			return ColorGradient.from_json_dict(grad_dict)
	raise ValueError('{} was not found among the available gradients.'.format(name))


def fetch_random_gradient(grad_file=DEF_GRAD_FILE):
	grad_path = os.path.join(pwd(), grad_file)
	with open(grad_path, 'r') as f:
		data = json.load(f)  
	grad_dict = sample_from_array(data)
	return ColorGradient.from_json_dict(grad_dict)
