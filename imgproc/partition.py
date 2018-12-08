from imgproc.utils import check_color, is_iterable
import numpy as np


class Cell:

	def __init__(self, img):
		self.img = img

	def __len__(self):
		return len(self.img)

	@property
	def shape(self):
		return self.img.shape

	# mfunc for morphing function ? better name ?
	def apply(self, mfunc):
		self.img = mfunc(self.img)

	def paint(self, col):
		check_color(col)
		self.img = np.tile(col, (self.shape[0], self.shape[1], 1))
	

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
		self.curr = 0

	def __iter__(self):
		return self

	def __next__(self):
		if self.curr > self.n_bands - 1:
			raise StopIteration
		else:
			self.curr += 1
			return self.bands[self.curr - 1]

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
				return Cell(self.img[:self.indices[0]])
			elif i == nb - 1:
				return Cell(self.img[self.indices[-1]:])
			else:
				return Cell(self.img[self.indices[i - 1]: self.indices[i]])
		elif self.btype == 'vertical':
			if i == 0:
				return Cell(self.img[:, :self.indices[0]])
			elif i == nb - 1:
				return Cell(self.img[:, self.indices[-1]:])
			else:
				return Cell(self.img[:, self.indices[i - 1]: self.indices[i]])
		else:
			err = 'Only `horizontal` and `vertical` modes are supported at the moment.'
			raise NotImplementedError(err)

	def apply_to_all(self, func):
		for band in self:
			band.apply(func)

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


class RegularGridDivider:

	def __init__(self, img, cell_size):
		self.img = img
		self.cell_size = cell_size
		self._check_cell_size()
		self.cells = self._divide_into_cells()
		self.curr = 0

	def __iter__(self):
		return self

	def __next__(self):
		if self.curr > self.n_cells - 1:
			raise StopIteration
		else:
			i, j = self.curr // self.n_row_cells, self.curr % self.n_row_cells
			self.curr += 1
			return self.bands[i, j]

	def __getitem__(self, iter_):
		i, j = iter_
		if i > self.n_row_cells - 1:
			raise IndexError('Exceeded actual amount of cells per row.')
		elif j > self.n_col_cells - 1:
			raise IndexError('Exceeded actual amount of cells per column.')
		return self.cells[i][j]

	def __len__(self):
		return self.n_row_cells

	@property
	def n_row_cells(self):
		return self.img.shape[0] // self.cell_size[0]

	@property
	def n_col_cells(self):
		return self.img.shape[1] // self.cell_size[1]

	@property
	def n_cells(self):
		return self.n_row_cells * self.n_col_cells

	@property
	def n_bands(self):
		if self.indices is None:
			raise TypeError('Must define band indices first.')
		return len(self.indices) + 1

	def _check_cell_size(self):
		if not is_iterable(self.cell_size):
			raise TypeError('Cell size should be an iterable.')
		elif len(self.cell_size) != 2:
			raise IndexError('Cell size should have a length of 2.')
		elif self.img.shape[0] % self.cell_size[0] != 0:
			raise IndexError('Image has height that is undivisible by cell size.')
		elif self.img.shape[1] % self.cell_size[1] != 0:
			raise IndexError('Image has width that is undivisible by cell size.')

	def _divide_into_cells(self):
		return [[self._get_cell(i, j) for j in range(self.n_col_cells)] for i in range(self.n_row_cells)]

	def _get_cell(self, i, j):
		nr, nc = self.n_row_cells, self.n_col_cells
		sr, sc = self.cell_size
		return Cell(self.img[i * sr: (i + 1) * sr, j * sc: (j + 1) * sc])

	def apply_to_all(self, func):
		for cell in self:
			cell.apply(func)

	def stitch(self):
		canvas = np.zeros_like(self.img)
		sr, sc = self.cell_size
		for i in range(self.n_row_cells):
			for j in range(self.n_col_cells):
				canvas[i * sr: (i + 1) * sr, j * sc: (j + 1) * sc] = self.cells[i][j].img
		return canvas
