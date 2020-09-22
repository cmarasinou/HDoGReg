import numpy as np
from skimage.draw import circle
from skimage.filters.edges import convolve

def img_preprocessing(img, kernel_size=7):
    k_sz = kernel_size
    k_radius = k_sz/2
    kernel = np.zeros((k_sz, k_sz), dtype=np.uint8)
    rr,cc=circle(k_sz//2,k_sz//2,k_radius)
    kernel[rr,cc]=1
    img_subtr = img-convolve(img,kernel)/kernel.sum()
    img_stdr = (img_subtr-img_subtr.mean())/img_subtr.std()
    return img_stdr


class ContextPaperPreprocess():
    """ Transforms the first image according to the preprocessing in paper
    A context-sensitive deep learning approach for microcalcification detection in mammograms
    """
    
    def __init__(self, kernel_size=7):
        self.ks = kernel_size

    def __call__(self, imgs):
        """
        Args:
            imgs (list of np.arrays): first in the list should be the actual image
        Returns:
            tranformed images (only first image changes)
        """
        imgs[0] = img_preprocessing(imgs[0], self.ks)

        return imgs