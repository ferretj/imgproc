import numpy as np


class Band:

	def __init__(self, img):
		self.img = img

	def __len__(self):
		return len(self.img)

	# mfunc for morphing function ? better name ?
	def apply(self, mfunc):
		self.img = mfunc(self.img)

	@property
	def shape(self):
		return self.img.shape
	

# question : does it take additional memory to give Band an indexed numpy array
#            or is it only a view (in that case, do not really need slice attribute in Band)
# answer : no copy but have to be careful not to erase original values in image
class BandDivider:

	def __init__(self, img, indices, btype='horizontal'):
		self.img = img
		self.indices = indices
		self._remove_duplicates(sort=True)
		self.btype = btype
		self.bands = self._divide_into_bands()

	def __getitem__(self, i):
		if i > self.n_bands - 1:
			raise IndexError('Exceeded actual amount of bands.')
		return self.bands[i]

	def __len__(self):
		return self.n_bands

	@property
	def n_bands(self):
		if self.indices is None:
			raise TypeError('Must define band indices first.')
		return len(self.indices) + 1

	def _remove_duplicates(self, sort=False):
		self.indices = np.array(list(set(self.indices)))
		if sort:
			self.indices.sort()

	def _divide_into_bands(self):
		return [self._get_band(i) for i in range(self.n_bands)]

	def _get_band(self, i):
		nb = self.n_bands
		if self.btype == 'horizontal':
			if i == 0:
				return Band(self.img[:self.indices[0]])
			elif i == nb - 1:
				return Band(self.img[self.indices[-1]:])
			else:
				return Band(self.img[self.indices[i - 1]: self.indices[i]])
		elif self.btype == 'vertical':
			if i == 0:
				return Band(self.img[:, :self.indices[0]])
			elif i == nb - 1:
				return Band(self.img[:, self.indices[-1]:])
			else:
				return Band(self.img[:, self.indices[i - 1]: self.indices[i]])
		else:
			err = 'Only `horizontal` and `vertical` modes are supported at the moment.'
			raise NotImplementedError(err)

	def stitch(self):
		if self.btype == 'horizontal':
			# print('Printing shapes :\n')
			# for i, band in enumerate(self.bands):
			# 	print('Band {} : {}'.format(i + 1, band.shape))
			return np.concatenate([band.img for band in self.bands], axis=0)
		elif self.btype == 'vertical':
			return np.concatenate([band.img for band in self.bands], axis=1)
		else:
			raise NotImplementedError
