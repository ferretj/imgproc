from imgproc.utils import hex_chars, is_iterable
import numpy as np
import random


def sampling_amount(size):
	if isinstance(size, int):
		return size
	elif isinstance(size, float):
		print('WARNING: size argument containing floats.')
		return int(size)
	elif is_iterable(size):
		return np.prod(size)
	else:
		raise TypeError('size argument should be an int or an iterable.')


def sample_from_array(elems, size=1, weights=None):
	if weights is not None and len(elems) != len(weights):
		raise IndexError('Must have as many elements as weights.')
	if size == 1:
		return np.random.choice(elems)
	return list(np.random.choice(elems, size=size, p=weights))


def sample_from_arrays_alternating(elems, size=1):
	n = len(elems)
	n_samples = sampling_amount(size)
	start = random.randint(0, n - 1)
	samples = []
	for i in range(n_samples):
		k = (start + i) % n
		sample = sample_from_array(elems[k])
		samples.append(sample) 
	return samples


def _sample_from_dict_once(d):
	dres = dict()
	for item, elems in d.items():
		dres[item] = np.random.choice(elems)
	return dres


# space is a dictionary
def sample_from_dict(d, size=1):
	n_samples = sampling_amount(size)
	if size == 1:
		return _sample_from_dict_once(d)
	else:
		samples = []
		for _ in range(n_samples):
			samples.append(_sample_from_dict_once(d))
		return samples


def random_hex(size):
	return ''.join([str(c) for c in np.random.choice(hex_chars(), size)])


def random_color():
	return np.random.randint(256, size=(3,))
