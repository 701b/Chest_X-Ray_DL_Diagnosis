from skimage import exposure


def normalize(img):
    normalized_img = exposure.equalize_adapthist(img)
    return normalized_img
