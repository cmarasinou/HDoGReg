import numpy as np

class ExpProximityFunction():
    """ Creates exponential proximity function.
        Args:
            radius (float): Appears as xi in the paper, extend of proximity function
            alpha (float): Decay rate parrameter
    """
    
    def __init__(self, radius=10, alpha=1):
        self.radius = radius
        self.alpha = alpha

    def __call__(self, imgs):
        """
        Args:
            imgs (list of np.arrays, 2-dim): second in the list should be the binary mask
        Returns:
            tranformed images (only second image changes)
        """
        if self.radius == 0:
            return imgs

        exp_map_func = make_exponential_mask_from_binary_mask
        imgs[1] = exp_map_func(imgs[1][:,:,0], 
            radius=self.radius, alpha=self.alpha)
        imgs[1] = np.expand_dims(imgs[1],2)

        return imgs


    
def make_exponential_mask(img, locations, radius, alpha, INbreast=False):
    """Creating exponential proximity function mask given the locations of objects.
    Args:
        img (np.array, 2-dim): the image, only it's size is important
        locations (np.array, 2-dim): array should be (n_locs x 2) in size and 
            each row should correspond to a location [x,y]. Don't need to be integer,
            truncation is applied.
            NOTICE [x,y] where x is row number (distance from top) and y column number
            (distance from left)
        radius (int): radius of the exponential pattern
        alpha (float): decay rate
        INbreast (bool, optional): Not needed anymore, handled when parsing INbreast dataset
    Returns:
        mask (np.array, 0.0-1.0): Exponential proximity function
    """
    # create kernel which we will be adding at locations
    # Kernel has radial exponential decay form
    kernel = np.zeros((2*radius+1,2*radius+1))
    for i in range(0, kernel.shape[0]):
        for j in range(0, kernel.shape[1]):
            d = np.sqrt((i-radius)**2+(j-radius)**2)
            if d<= radius:
                kernel[i,j]=(np.exp(alpha*(1-d/radius))-1)/(np.exp(alpha)-1)
                
    # pad original img to avoid out of bounds errors
    img  = np.pad(img, radius+1, 'constant').astype(float)

    # update locations
    locations = np.array(locations)+radius+1
    locations = np.round(locations).astype(int)

    # initialize mask
    mask = np.zeros_like(img)    

    for location in locations:
        if INbreast:
            y, x = location
        else:
            x, y = location
        # add kernel
        mask[x-radius:x+radius+1, y-radius:y+radius+1] =np.maximum(mask[x-radius:x+radius+1, y-radius:y+radius+1],kernel)
        
    # unpad
    mask  = mask[radius+1:-radius-1,radius+1:-radius-1]
    
    return mask

def make_exponential_mask_from_binary_mask(mask, radius, alpha):
    """Creating exponential proximity function mask given the binary mask.
    Args:
        mask (np.array, 2-dim): binary mask
        radius (int): radius of the exponential pattern
        alpha (float): decay rate
    Returns:
        mask (np.array, 0.0-1.0): Exponential proximity function
    """
    locations = np.array(np.where(mask)).T
    mask = make_exponential_mask(mask, locations, radius, alpha, INbreast=False)
    
    return mask