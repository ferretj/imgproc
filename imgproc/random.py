from imgproc.utils import is_iterable
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


def alternate_uniform_sample(elems, size=1):
	n = len(elems)
	n_samples = sampling_amount(size)
	start = random.randint(0, n - 1)
	samples = []
	for i in range(n_samples):
		k = (start + i) % n
		sample = np.random.choice(elems[k])
		samples.append(sample) 
	return samples
