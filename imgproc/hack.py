from imgproc.utils import check_hexadecimal, check_imgfile_arg, hex_to_num, make_filecopy
import os

# one byte = two hexadecimal symbols e.g. ff
JPG_NBYTES_LINE = 16
JPG_BYTE_START = 384


# assumes imgfile data always starts at the same
# position (because of standardized header)
#
# assumes always same number of characters per line 
#
# pos=k refers to the (k + 1)th sequence of symbols of the same size as hexa
#
#TODO: can skip lines ?
#      use a temp file ?
#      load numpy array in memory and erase file ??
def switch_bytes_jpeg(imgfile, positions, hexas, make_copy=True, return_dest=False):
	check_imgfile_arg(imgfile, ftype='jpeg')
	for hexa in hexas:
		check_hexadecimal(hexa)
	assert all([len(hexa) >= 2 for hexa in hexas])
	assert all([len(hexa) % 2 == 0 for hexa in hexas])

	# makes a copy of the file by default
	if make_copy:
		newfile = make_filecopy(imgfile, return_dest=True)
	else:
		newfile = imgfile
	# read in the file
	# bytearray used to turn bytes object into smth mutable
	with open(newfile, 'rb') as f:
		filedata = bytearray(f.read())

	sizes = [len(hexa) // 2 for hexa in hexas]
	ns = [JPG_BYTE_START + pos * size for pos, size in zip(positions, sizes)]
	# modify the target filedata
	for hexa, n, size in zip(hexas, ns, sizes):
		filedata[n: n + size] = [hex_to_num(hexa[i: i + 2]) for i in range(0, 2 * size, 2)]
	# write the file out again
	with open(newfile, 'wb') as f:
		f.write(filedata)
	
	# line_num, line_pos = n // JPG_NCHAR_LINE, n % JPG_NCHAR_LINE 
	# with open(imgfile, 'rb+') as f:
	# 	for i, line in enumerate(f):
	# 		if i == line_num:
	# 			break

	if return_dest:
		return newfile
