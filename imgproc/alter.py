from imgproc.utils import deg_to_rad, numpy_to_pil, pil_to_numpy
import math
from PIL import ImageDraw


def invert_iterable(iter_):
	if isinstance(iter_, tuple):
		return tuple([elem for elem in iter_][::-1])
	return iter_[::-1]


def invert_all_coords(arr):
	return [invert_iterable(coords) for coords in arr]


#TODO: inplace
def draw_rectangle(img, coords, color):
	img_mod = numpy_to_pil(img)
	draw = ImageDraw.Draw(img_mod)
	draw.rectangle(coords, fill=tuple(color))
	return pil_to_numpy(img_mod)


#TODO: inplace
# def draw_rectangle_from_base(img, base_coords, length, color, dir='right'):
# 	delta_x = abs(base_coords[0][0] - base_coords[1][0])
# 	delta_y = abs(base_coords[0][1] - base_coords[1][1])
# 	theta = deg_to_rad(90) - math.atan2(delta_y, delta_x)
# 	return draw_rectangle_from_blcorner_angle(img, base_coords[0], length, theta, color, degree_fmt=False)


#TODO: inplace
# theta is the angle between the diagonal and the x-axis (oriented towards increasing xs)
# length is the norm of the diagonal
def draw_rectangle_from_blcorner_angle(img, xy, length, theta, color, degree_fmt=True, debug=False):
	img_mod = numpy_to_pil(img)
	if degree_fmt:
		theta = deg_to_rad(theta)
	opp_xy = (xy[0] + length * math.cos(theta), xy[1] + length * math.sin(theta))
	if debug:
		print('Opposite point coords : {}'.format(opp_xy))
	coords = [invert_iterable(xy), invert_iterable(opp_xy)]
	draw = ImageDraw.Draw(img_mod)
	draw.rectangle(coords, fill=tuple(color))
	return pil_to_numpy(img_mod)


#TODO: inplace
def draw_general_rectangle(img, coords, color):
	img_mod = numpy_to_pil(img)
	draw = ImageDraw.Draw(img_mod)
	draw.polygon(coords, fill=tuple(color))
	return pil_to_numpy(img_mod)


#TODO: inplace
def draw_general_rectangle_from_base(img, base_coords, length, color, dir_='right'):
	delta_x = abs(base_coords[0][0] - base_coords[1][0])
	delta_y = abs(base_coords[0][1] - base_coords[1][1])
	theta = deg_to_rad(90) - math.atan2(delta_y, delta_x)
	if dir_ == 'right':
		opp_coords = [(xy[0] - length * math.cos(theta), xy[1] + length * math.sin(theta)) for xy in base_coords[::-1]]
	else:
		opp_coords = [(xy[0] + length * math.cos(theta), xy[1] - length * math.sin(theta)) for xy in base_coords[::-1]]
	coords = invert_all_coords(base_coords) + invert_all_coords(opp_coords)
	return draw_general_rectangle(img, coords, color)


#TODO: inplace
def draw_general_rectangle_from_point(img, xy, length, width, theta, color, degree_fmt=True, debug=False):
	if degree_fmt:
		theta = deg_to_rad(theta)
	alpha = deg_to_rad(90) - theta
	opp_xy = (xy[0] + length * math.cos(theta), xy[1] + length * math.sin(theta))
	if debug:
		print('Opposite point coords : {}'.format(opp_xy))
	coords = [xy, opp_xy]
	coords.append((opp_xy[0] - width * math.cos(alpha), opp_xy[1] + width * math.sin(alpha)))
	coords.append((xy[0] - width * math.cos(alpha), xy[1] + width * math.sin(alpha)))
	coords = invert_all_coords(coords)
	return draw_general_rectangle(img, coords, color)
