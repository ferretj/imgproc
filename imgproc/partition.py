import numpy as np


# TODO: can use a better pattern to manage the mod attribute
class Band:

	def __init__(self, img, img_slice):
		self.img = img
		self.img_slice = img_slice
		self.mod = None

	# mfunc for morphing function ? better name ?
	def apply(self, mfunc):
		if self.mod is None:
			self.mod = mfunc(self.img[self.img_slice])
		else:
			self.mod = mfunc(self.mod)

	@property
	def data(self):
		if self.mod is None:
			return self.img[self.img_slice]
		return self.mod


# question : does it take additional memory to give Band an indexed numpy array
#            or is it only a view (in that case, do not really need slice attribute in Band)
class BandDivider:

	def __init__(self, img, indices, btype='horizontal'):
		self.img = img
		self.indices = indices
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

	def _divide_into_bands(self):
		bands = []
		for i in self.n_bands:
			bands.append(self._get_band(i))
		return bands

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
			return np.stack([band.data for band in self.bands], axis=0)
		elif self.btype == 'vertical':
			return np.stack([band.data for band in self.bands], axis=1)
		else:
			raise NotImplementedError
