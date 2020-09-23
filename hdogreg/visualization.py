import matplotlib as mlt
import numpy as np
import torch
from PIL import Image

def generate_masked_image(img_arr, mask_arr_list, color_list, alpha_list):
    ''' Generates raw image overlayed by mask as PIL.Image
    Args:
        img_arr (np.array, 2-dim): raw image
        mask_arr_list (list of np.array, 2-dim): binary mask
        color_list (list of RGB tuples)
        alpha_list (alpha for each mask)
    Returns:
        overlayed PIL.Image
    '''
    img_arr = (img_arr*(2**8-1)).astype(np.uint8)
    img = Image.fromarray(img_arr).convert('RGBA')
    
    for mask_arr, color, alpha in zip(mask_arr_list, color_list, alpha_list):
        alpha_channel = (mask_arr!=0)*(2**8-1)*alpha
        color_channel = mask_arr.astype(np.uint8)
        r,g,b = color
        mask_arr = np.stack((r*color_channel,
                             g*color_channel,
                             b*color_channel,
                             alpha_channel),
                           axis=2)
        mask = Image.fromarray((mask_arr).astype(np.uint8))

        img = Image.alpha_composite(img,mask)
    
    return img


def pil_from_array(im_array, cmap='gray'):
    if isinstance(im_array, torch.Tensor):
        im_array = im_array.squeeze().numpy()
    if im_array.dtype.name == 'bool':
        im_array = im_array.astype(float) 
    cm= mlt.cm.get_cmap(cmap)
    im = cm(im_array)
    im = np.uint8(im * 255)
    im = Image.fromarray(im)
    return im