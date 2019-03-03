from collections import defaultdict
from copy import deepcopy
from imgproc.utils import check_color, is_iterable, make_canvas
import itertools
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors


# defines a rectangular patch of image that can be represented as a numpy array
class Block:

	# coords are coordinates of opposite corners (as a 2D array or alike)
	def __init__(self, img, coords=None):
		self.img = img
		self.coords = coords

	def __len__(self):
		return len(self.img)

	@property
	def center(self):
		if self.coords is None:
			# raise AttributeError('Need to specify `coords` attribute when defining Block.')
			return None
		return np.mean(coords, axis=0)

	@property
	def shape(self):
		return self.img.shape

	#TODO: mfunc for morphing function ? better name ?
	def apply(self, mfunc):
		self.img = mfunc(self.img, **self._kwargs())

	def paint(self, col):
		check_color(col)
		self.img = np.tile(col, (self.shape[0], self.shape[1], 1))

	def _kwargs(self):
		kwargs = dict(
			center=self.center, 
			shape=self.shape,
		)
		return kwargs


# defines connected group of pixels of arbitrary shape
# inspiration : sparse matrix representation
class Cell:

	def __init__(self, vals, rows, cols):
		assert len(vals) == len(rows)
		assert len(vals) == len(cols)
		self.vals = vals
		self.rows = rows
		self.cols = cols

	def __len__(self):
		return len(self.vals)

	@property
	def mrow(self):
		return np.min(self.rows)

	@property
	def mcol(self):
		return np.min(self.cols)

	@property
	def vshape(self):
		rm, rx = self.mrow, np.max(self.rows)
		cm, cx = self.mcol, np.max(self.cols)
		return (rx - rm + 1, cx - cm + 1, 3)

	def _view(self):
		view = np.zeros(self.vshape).astype(np.uint8)
		view[self.rows - self.mrow, self.cols - self.mcol] = self.vals
		return view

	#TODO: mfunc for morphing function ? better name ?
	def apply(self, mfunc):
		mod_view = mfunc(self._view())
		self.vals = mod_view[self.rows - self.mrow, self.cols - self.mcol]
	
	def paint(self, col):
		check_color(col)
		self.vals = np.tile(col, (len(self), 1))


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
		self._curr = 0

	def __iter__(self):
		return self

	def __next__(self):
		if self._curr > self.n_bands - 1:
			self._curr = 0
			raise StopIteration
		else:
			self._curr += 1
			return self.bands[self._curr - 1]

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
				return Block(self.img[:self.indices[0]])
			elif i == nb - 1:
				return Block(self.img[self.indices[-1]:])
			else:
				return Block(self.img[self.indices[i - 1]: self.indices[i]])
		elif self.btype == 'vertical':
			if i == 0:
				return Block(self.img[:, :self.indices[0]])
			elif i == nb - 1:
				return Block(self.img[:, self.indices[-1]:])
			else:
				return Block(self.img[:, self.indices[i - 1]: self.indices[i]])
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

	# cell_size is a tuple/iterable of two elememnts
	def __init__(self, img, block_size):
		self.img = img
		self.block_size = block_size
		self._check_block_size()
		self.blocks = self._divide_into_blocks()
		self._curr = 0

	def __iter__(self):
		return self

	def __next__(self):
		if self._curr > self.n_blocks - 1:
			self._curr = 0
			raise StopIteration
		else:
			i, j = self._curr // self.n_col_blocks, self._curr % self.n_col_blocks
			self._curr += 1
			return self.blocks[i][j]

	def __getitem__(self, iter_):
		i, j = iter_
		if i > self.n_row_blocks - 1:
			raise IndexError('Exceeded actual amount of blocks per row.')
		elif j > self.n_col_blocks - 1:
			raise IndexError('Exceeded actual amount of blocks per column.')
		return self.blocks[i][j]

	def __len__(self):
		return self.n_row_blocks

	@property
	def n_row_blocks(self):
		return self.img.shape[0] // self.block_size[0]

	@property
	def n_col_blocks(self):
		return self.img.shape[1] // self.block_size[1]

	@property
	def n_blocks(self):
		return self.n_row_blocks * self.n_col_blocks

	@property
	def n_bands(self):
		if self.indices is None:
			raise TypeError('Must define band indices first.')
		return len(self.indices) + 1

	def _check_block_size(self):
		if not is_iterable(self.block_size):
			raise TypeError('block size should be an iterable.')
		elif len(self.block_size) != 2:
			raise IndexError('block size should have a length of 2.')
		elif self.img.shape[0] % self.block_size[0] != 0:
			raise IndexError('Image has height that is undivisible by block size.')
		elif self.img.shape[1] % self.block_size[1] != 0:
			raise IndexError('Image has width that is undivisible by block size.')

	def _divide_into_blocks(self):
		return [[self._get_block(i, j) for j in range(self.n_col_blocks)] for i in range(self.n_row_blocks)]

	def _get_block(self, i, j):
		nr, nc = self.n_row_blocks, self.n_col_blocks
		sr, sc = self.block_size
		return Block(self.img[i * sr: (i + 1) * sr, j * sc: (j + 1) * sc])

	def _random_block_index(self):
		return (random.randint(0, self.n_row_blocks - 1), random.randint(0, self.n_col_blocks - 1))

	def _random_block_indices(self, num_samples):
		all_block_indices = list(itertools.product(np.arange(self.n_row_blocks), np.arange(self.n_col_blocks)))
		samples = np.random.permutation(np.arange(self.n_blocks))[:num_samples]
		return [all_block_indices[ind] for ind in samples]

	def apply_to_all(self, func):
		for block in self:
			block.apply(func)

	def apply_to_random_sample(self, func, num_samples):
		if isinstance(num_samples, float):
			if 0. <= num_samples <= 1.:
				num_samples = int(num_samples * self.n_blocks)
			else:
				error = 'Expecting `num_samples` to be an integer or a (0, 1)-float.'
				raise ValueError(error)
		indices = self._random_block_indices(num_samples)
		for (i, j) in indices:
			self.blocks[i][j].apply(func)

	def apply_to_random(self, func):
		i, j = self._random_block_index()
		self.blocks[i][j].apply(func)

	def apply_to_selected(self, indices, func):
		assert isinstance(indices, list)
		for (i, j) in indices:
			self.blocks[i][j].apply(func)

	def paint_all(self, col_func):
		for block in self:
			col = col_func()
			block.paint(col)

	def paint_random_sample(self, col_func, num_samples):
		if isinstance(num_samples, float):
			if 0. <= num_samples <= 1.:
				num_samples = int(num_samples * self.n_blocks)
			else:
				error = 'Expecting `num_samples` to be an integer or a (0, 1)-float.'
				raise ValueError(error)
		indices = self._random_block_indices(num_samples)
		for (i, j) in indices:
			col = col_func()
			self.blocks[i][j].paint(col)

	def paint_random(self, col_func):
		i, j = self._random_block_index()
		col = col_func()
		self.blocks[i][j].paint(col)

	def paint_selected(self, indices, col_func):
		assert isinstance(indices, list)
		for (i, j) in indices:
			col = col_func()
			self.blocks[i][j].paint(col)

	def stitch(self):
		canvas = np.zeros_like(self.img)
		sr, sc = self.block_size
		for i in range(self.n_row_blocks):
			for j in range(self.n_col_blocks):
				canvas[i * sr: (i + 1) * sr, j * sc: (j + 1) * sc] = self.blocks[i][j].img
		return canvas


class CircularDivider:

	def __init__(self, img, rads, center=None):
		self.img = img
		self.rads = rads
		if center is None:
			self.center = self._default_center()
		else:
			self.center = np.array(center)
		self._remove_duplicates(sort=True)
		self._dist_map = self._map_dist_to_center() 
		self.cells = self._divide_into_cells()
		self._curr = 0

	def __iter__(self):
		return self

	def __next__(self):
		if self._curr > self.n_cells - 1:
			self._curr = 0
			raise StopIteration
		else:
			self._curr += 1
			return self.cells[self._curr - 1]

	def __getitem__(self, i):
		if i > self.n_cells - 1:
			raise IndexError('Exceeded actual amount of cells.')
		return self.cells[i]

	def __len__(self):
		return self.n_cells

	@property
	def n_cells(self):
		if self.rads is None:
			raise TypeError('Must define radiuses first.')
		return len(self.rads) + 1

	def _remove_duplicates(self, sort=False):
		self.rads = np.array(list(set(self.rads)))
		if sort:
			self.rads.sort()

	def _divide_into_cells(self):
		return [self._get_cell(i) for i in range(self.n_cells)]

	def _default_center(self):
		h, w = self.img.shape[:2]
		return np.array([int(h // 2), int(w // 2)])

	#TODO: can make faster by large margin
	def _map_dist_to_center(self):
		h, w = self.img.shape[:2]
		ijmap = np.array([[[i, j] for j in range(w)] for i in range(h)])
		return np.linalg.norm(ijmap - self.center, axis=-1)

	@property 
	def _max_rad(self):
		h, w = self.img.shape[:2]
		return np.max([self.center[0],
					   h - 1 - self.center[0],
					   self.center[1],
					   w - 1 - self.center[1]])

	def _get_cell(self, i):
		nb = self.n_cells
		if i == 0:
			rad_low, rad_high = 0, self.rads[i]
		elif i == nb - 1:
			rad_low, rad_high = self.rads[i - 1], self._max_rad
		else:
			rad_low, rad_high = self.rads[i - 1], self.rads[i]
		rows, cols = np.where((self._dist_map >= rad_low) & (self._dist_map < rad_high))
		vals = np.array([self.img[r, c] for r, c in zip(rows, cols)])
		return Cell(vals, rows, cols)

	def apply_to_all(self, func):
		for cell in self:
			cell.apply(func)

	def paint_all(self, col_func):
		for cell in self:
			col = col_func()
			cell.paint(col)

	def stitch(self):
		canvas = deepcopy(self.img)
		for cell in self:
			canvas[cell.rows, cell.cols] = cell.vals
		return canvas


class ArbitraryDivider:

    def __init__(self, img, cell_rows, cell_cols):
        self.img = img
        self.cell_rows = cell_rows
        self.cell_cols = cell_cols
        self._check_non_overlapping()
        self._check_complete()
        self.cells = self._divide_into_cells()
        self._curr = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr > self.n_cells - 1:
            self._curr = 0
            raise StopIteration
        else:
            self._curr += 1
            return self.cells[self._curr - 1]

    def __getitem__(self, i):
        if i > self.n_cells - 1:
            raise IndexError('Exceeded actual amount of cells.')
        return self.cells[i]

    def __len__(self):
        return self.n_cells

    @property
    def n_cells(self):
        assert len(self.cell_rows) == len(self.cell_cols)
        return len(self.cell_rows)
    
    def _concat(self, cell_rows, cell_cols):
        all_elems = []
        for row_arr, col_arr in zip(cell_rows, cell_cols):
            all_elems.extend(list(zip(row_arr, col_arr)))
        return all_elems
    
    def _check_non_overlapping(self):
        all_elems = self._concat(self.cell_rows, self.cell_cols)
        if len(all_elems) != len(set(all_elems)):
            n_overlap = len(all_elems) - len(set(all_elems))
            raise ValueError('Overlapping elements detected ({}).'.format(n_overlap))
    
    def _check_complete(self):
        all_elems = self._concat(self.cell_rows, self.cell_cols)
        n_pixels = np.prod(self.img.shape[:2])
        if len(all_elems) != n_pixels:
            n_missing = n_pixels - len(all_elems)
            raise ValueError('Missing elements detected ({}).'.format(n_missing))

    def _divide_into_cells(self):
        return [self._get_cell(i) for i in range(self.n_cells)]

    def _get_cell(self, i):
        rows, cols = self.cell_rows[i], self.cell_cols[i]
        vals = np.array([self.img[r, c] for r, c in zip(rows, cols)])
        return Cell(vals, rows, cols)

    def apply_to_all(self, func):
        for cell in self:
            cell.apply(func)
    
    def paint_all(self, col_func):
        for cell in self:
            col = col_func()
            cell.paint(col)

    def stitch(self):
        canvas = deepcopy(self.img)
        for cell in self:
            canvas[cell.rows, cell.cols] = cell.vals
        return canvas


class VoronoiDivider(ArbitraryDivider):

    def __init__(self, img, coords, k, ordered=True):
        cell_rows, cell_cols = self._voronoi(img, coords, k, ordered)
        super().__init__(img, cell_rows, cell_cols)
    
    def _grid_coords(self, grid_size):
        return np.array(np.meshgrid(np.arange(grid_size[0]),
                                    np.arange(grid_size[1]))).T.reshape(-1, 2)
    
    def _ordered_tuple(self, elems):
        elems.sort()
        return tuple(elems)

    def _voronoi(self, img, coords, k, ordered):
        h, w = img.shape[:2]
        if isinstance(coords, list):
            coords = np.array(coords)
        nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
        img_coords = self._grid_coords((h, w))
        _, indices = nbrs.kneighbors(img_coords)
        if ordered:
            indices = [self._ordered_tuple(idxs) for idxs in indices]
        else:
            indices = [tuple(idxs) for idxs in indices]
        cell_rows = defaultdict(list)
        cell_cols = defaultdict(list)
        for coords, idx in zip(img_coords, indices):
            cell_rows[idx].append(coords[0])
            cell_cols[idx].append(coords[1])
        cell_rows = [cell_rows[idx] for idx in set(indices)]
        cell_cols = [cell_cols[idx] for idx in set(indices)]
        return cell_rows, cell_cols
