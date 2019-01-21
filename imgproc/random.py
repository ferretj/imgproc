from imgproc.utils import del_all_selected, hex_chars, is_iterable
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


# np.random.choice but not converting to numpy floats and ints
def choice_onedim(elems, size=1, p=None, return_indices=False):
	n = len(elems)
	n_samples = sampling_amount(size)
	if p is not None:
		assert is_iterable(p)
		assert len(p) == n
	indices = np.random.choice(np.arange(n), size=n_samples, p=p)
	
	if return_indices:
		if size == 1:
			return elems[indices[0]], indices[0]
		return [elems[ind] for ind in indices], indices
	if size == 1:
		return elems[indices[0]]
	return [elems[ind] for ind in indices]


# generalization of np.random.choice to numpy array with arbitrary dimensionality
# sole purpose is to specify axis
def choice_multidim(elems, size=1, p=None, axis=0, return_indices=False):
	if isinstance(elems, list):
		elems = np.array(elems)
	n = elems.shape[axis]
	n_samples = sampling_amount(size)
	if p is not None:
		assert is_iterable(p)
		assert len(p) == n
	indices = np.random.choice(np.arange(n), size=n_samples, p=p)
	
	if return_indices:
		if size == 1:
			return np.take(elems, indices[0], axis=axis), indices[0]
		return np.array([np.take(elems, ind, axis=axis) for ind in indices]), indices
	if size == 1:
		return np.take(elems, indices[0], axis=axis)
	return np.array([np.take(elems, ind, axis=axis) for ind in indices])


def sample_from_array(elems, size=1, weights=None):
	if len(elems) == 0:
		raise IndexError('elems argument must be non-empty.')
	elif weights is not None and len(elems) != len(weights):
		raise IndexError('Must have as many elements as weights.')
	
	if isinstance(elems[0], np.ndarray):
		choice = choice_multidim
	else:
		choice = choice_onedim
	
	if size == 1:
		return choice(elems)
	return list(choice(elems, size=size, p=weights))


def sample_pop_from_array(elems, size=1, weights=None):
	if len(elems) == 0:
		raise IndexError('elems argument must be non-empty.')
	elif weights is not None and len(elems) != len(weights):
		raise IndexError('Must have as many elements as weights.')
	
	if isinstance(elems[0], np.ndarray):
		choice = choice_multidim
	else:
		choice = choice_onedim
	
	if size == 1:
		sample, idx = choice(elems, return_indices=True)
		del elems[idx]
		return sample
	else:
		#samples, indices = choice(elems, size=size, p=weights, return_indices=True)
		#samples = list(samples)
		#del_all_selected(elems, indices)
		raise NotImplementedError  # above code breaks because choice samples with replacement
		return samples


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
	for item, val in d.items():
		if is_iterable(val):
			dres[item] = sample_from_array(val)
		else:
			dres[item] = val 
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


def random_grayscale():
	return np.array([random.randint(0, 255)] * 3)
