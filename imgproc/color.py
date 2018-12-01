from imgproc.utils import hilo


def complementary_color(rgb):
    hl = hilo(*rgb)
    return np.array([hl - c for c in rgb])


def complement(img):
	h, w = img.shape[:2]
    img_mod = np.array([complementary_color(img[i, j]) for i in range(h) for j in range(w)])
    return img_mod.reshape(img.shape)
