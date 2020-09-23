from math import log
import numpy as np
from skimage import filters
from skimage.morphology import remove_small_objects, remove_small_holes, reconstruction, extrema
from skimage.morphology import opening, closing, square, dilation,disk, binary_erosion,binary_dilation, erosion
from skimage.transform import pyramid_reduce, pyramid_expand
from skimage.measure import regionprops
from skimage.segmentation import watershed, find_boundaries
from skimage.util import img_as_float
from skimage.feature.blob import _prune_blobs
from skimage.feature import peak_local_max
from scipy import ndimage
import cv2

def delineate_breast(image, otsu_percent=0.2, bdry_remove=0.0, erosion=0, return_slice = False):
    '''Given a mammogram image, it outputs the delineation mask
    Args:
        image (np.array, 2-dim): breast image
        otsu_percent (float, optional): thresholding factor (threshold = otsu_percent*otsu_threshold)
        bdry_remove (float, optional): percentage of dimension to be removed
        return_slice (bool, optional): if true performs erosion of the object
    Returns: 
        binary mask of breast (and optionally slices describing the bounding box)
    '''

    image_original = image
    thr = filters.threshold_otsu(image)
    image = image>otsu_percent*thr
    # Denoising
    image = remove_small_objects(remove_small_holes(image))
    image, nlabels = ndimage.label(image)
    # Find size of each component
    labels, size = np.unique(image, return_counts=True)
    labels_by_size = np.argsort(-size) # Descending order
    # Assumption: The two largest components correspond to background
    # and breast tissue
    # Assumption: Out of the two the component with the highest median 
    # intensity is the breast
    c0 = image_original[image==labels_by_size[0]]
    c1 = image_original[image==labels_by_size[1]]
    c0_brightness = np.median(c0)
    c1_brightness = np.median(c1)
    if c1_brightness>c0_brightness:
        mask = (image==labels_by_size[1])
    else:
        mask = (image==labels_by_size[0])
    # Fill holes within structure
    mask = ndimage.binary_fill_holes(mask)

    # if gives all image fails
    #if np.sum(mask) == mask.shape[0]*mask.shape[1]:
    #    return None 

    # remove mask on the image boundaries
    h, w = mask.shape
    h_remove, w_remove = int(bdry_remove*h), int(bdry_remove*w)

    mask[0:h_remove,:]=0
    mask[h-h_remove:h,:]=0
    mask[:,0:w_remove]=0
    mask[:,w-w_remove:w]=0

    mask = binary_erosion(mask,disk(erosion))

    if return_slice:       
        return mask, ndimage.find_objects(mask)[0]
    else:
        return mask


def blob_detector_hdog(image, min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold=2.0,
             overlap=.5, hessian_thr=None):
    """Segments blobs of arbitrary shape using DoG and Hessian Analysis.
    Args:
        image (np.array, 2-dim)
        min_sigma (float, optional): DoG parameter
        max_sigma (float, optional): DoG parameter
        sigma_ratio (float, optional): DoG parameter
        threshold (float, optional): DoG parameter
        overlap (float, optional): DoG parameter, 0.0-1.0
        hessian_thr (float, optional): Hessian Analysis parameter.
            Must be positive. Larger values give more tubular blobs
    Returns:
        segmented mask (np.array, 2-dim): the blob segmentation mask as a boolean array;
            same size as `image` 
    """
 
    image = img_as_float(image)

    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(log(float(max_sigma) / min_sigma, sigma_ratio)) + 1

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio ** i)
                           for i in range(k + 1)])

    gaussian_images = [ndimage.gaussian_filter(image, s) for s in sigma_list]

    # computing difference between two successive Gaussian blurred images
    # multiplying with standard deviation provides scale invariance
    dog_images = [(gaussian_images[i] - gaussian_images[i + 1])
                  * sigma_list[i]/(sigma_list[i+1]-sigma_list[i]) 
                  for i in range(k)]

    image_cube = np.stack(dog_images, axis=-1)

    # local_maxima = get_local_maxima(image_cube, threshold)
    local_maxima = peak_local_max(image_cube, threshold_abs=threshold,
                                  footprint=np.ones((3,) * (image.ndim + 1)),
                                  threshold_rel=0.0,
                                  exclude_border=False)
    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 3))
    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)
    # Convert the last index to its corresponding scale value
    lm[:, -1] = sigma_list[local_maxima[:, -1]]
    
    # Hessian for segmentation
    if hessian_thr is None:
        hessian_images = [hessian_negative_definite(dog_image) for dog_image in dog_images]
    else:
        hessian_images = [hessian_criterion(dog_image, hessian_thr) for dog_image in dog_images]

    # Denoise
    #hessian_images = [remove_small_objects(hessian_images[i], min_size = 2*sigma_list[i]**2) for i in range(k)]
    #hessian_images = [remove_large_objects(hessian_images[i], min_size = 3*sigma_list[i]**2) for i in range(k)]
    blobs = _prune_blobs(lm, overlap)
    
    hessian_max_sup = np.zeros_like(image)
    for i in range(k):
        lbl, _ = ndimage.label(hessian_images[i])
        rprops = regionprops(lbl)
        s = sigma_list[i]
        for blob in blobs:
            if blob[2]==s:
                x,y = int(blob[0]), int(blob[1])
                # If we hit the background we cannot count it
                lbl_id = lbl[x,y]
                if lbl_id == 0:
                    continue
                x1,y1,x2,y2 = rprops[lbl_id-1].bbox
                hessian_max_sup[x1:x2,y1:y2] = np.logical_or(hessian_max_sup[x1:x2,y1:y2], lbl[x1:x2,y1:y2]==lbl_id)
    
    return hessian_max_sup


def hessian_criterion(img_f, thr):
    """Computes Hessian of array and applies constraint on it's Hessian
    Args: 
        img_f (np.array, 2-dim): float
        thr (float): positive; parameter h_{thr} as in paper
    Returns:
        segmented mask (np.array, 2-dim): the blob segmentation mask as a boolean array;
        same size as `image` 
    """
    img_hes = hessian(img_f)
    img_hes_det = img_hes[0,0,:,:]*img_hes[1,1,:,:]-img_hes[1,0,:,:]*img_hes[0,1,:,:]
    img_hes_trace = img_hes[0,0,:,:]+img_hes[1,1,:,:]
    img_hes_neg = np.logical_and(img_hes_det>0,img_hes_trace<0)
    img_hes_other = np.abs(img_hes_det)/np.power(img_hes_trace+1e-20,2)
    img_hes_other = np.logical_and(img_hes_other<thr,img_hes_trace<0)
    #img_hes_other = np.logical_and(img_hes_det<0,img_hes_trace<0)
    img_hes_neg = np.logical_or(img_hes_neg, img_hes_other)
    return img_hes_neg


def hessian_negative_definite(img_f):
    """Determines at what locations Hessian of `img_f` is negative definite"""
    img_hes = hessian(img_f)
    img_hes_det = img_hes[0,0,:,:]*img_hes[1,1,:,:]-img_hes[1,0,:,:]*img_hes[0,1,:,:]
    img_hes_trace = img_hes[0,0,:,:]+img_hes[1,1,:,:]
    img_hes_neg = np.logical_and(img_hes_det>0,img_hes_trace<0)
    return img_hes_neg

def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Args:
       x (np.array, 2-dim)
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian


def hybrid_approach(pred_mask,breast_mask,blob_mask,hybrid_combining='multiplication',
    hybrid_combining_overlap=0.3, extra_mask=None):
    '''Takes prediction, breast mask and blob mask,
    applies correspodning thresholding and combining to give
    a final prediction mask
    '''

    # for removing structures at the boundary
    #breast_mask = binary_erosion(breast_mask,square(45))
    breast_mask = cv2.erode(breast_mask.astype(np.float),cv2.getStructuringElement(cv2.MORPH_RECT,(45,45)))
    pred_mask = pred_mask*breast_mask
    blob_mask = blob_mask*breast_mask
    if extra_mask is not None:
        pred_mask = pred_mask*extra_mask
        blob_mask = blob_mask*extra_mask        
    if hybrid_combining=='multiplication':
        pred_mask = blob_mask*pred_mask
    elif hybrid_combining=='none':
        pred_mask = pred_mask
    elif hybrid_combining=='hdog':
        pred_mask = blob_mask
    else:
        pred_mask = overlap_based_combining(pred_mask, blob_mask, 
            thr=hybrid_combining_overlap)
        pred_mask = ndimage.binary_fill_holes(pred_mask)
    return pred_mask


def overlap_based_combining(region_mask, object_mask, thr=0.1):
    '''Finds objects in the object_mask that have a percentage
    of ovelap with the region_mask and retains only these.
    '''
    if region_mask.sum() == 0:
        return region_mask
    final_mask = np.zeros_like(region_mask)
    object_label, n_objects = ndimage.label(object_mask)
    props = regionprops(object_label)
    for prop in props:
        x1,y1,x2,y2 = prop.bbox
        bb = (x1,x2,y1,y2)
        lbl = prop.label
        obj_mask = object_label[x1:x2,y1:y2]==lbl
        region_patch = region_mask[x1:x2,y1:y2]
        overlap = np.logical_and(region_patch,obj_mask).sum()
        overlap /= obj_mask.sum()
        if overlap>thr:
            final_mask[x1:x2,y1:y2] += obj_mask
    return final_mask


def get_bounding_boxes(img, kernel_size, stride):
    """Gives all bounding boxes covering an image with a sliding window
    Args:
        img (np.array or torch.tensor)
        kernel_size (int): size of the kernel, kernel is a square
        stride (int): stride of the sliding window motion
    Returns:
        bounding boxes list (list of (x, xp,y, yp)), convention of bounding box
        (top, bottom, left, right)
    Notice: If not all image covered by sliding window, additional
    bounding boxes are created ending at the end of each dimension
    """
    h, w = img.shape
    h_p, w_p = kernel_size, kernel_size
    s = stride

    # All x locations
    x = [i*s for i in range(0, int((h-h_p)/s)+1)]
    # Add one more point if didn't cover all range
    if x[-1]+h_p!=h:
        x.append(h-h_p)
    # All y locations
    y = [j*s for j in range(0, int((w-w_p)/s)+1)]
    # Add one more point if didn't cover all range
    if y[-1]+w_p!=w:
        y.append(w-w_p)

    # All bounding boxes in the form (x,xp,y,yp)
    # x,y: the top left corner/ xp,yp: the bottom right corner
    bbList = [(xi,xi+h_p, yi, yi+w_p) for xi in x for yi in y]

    return bbList


def detect_calcifications_Ciecholewski(img, thr=10., h=5,):
    """Morphological calcification detector, using method in 
    Ciecholewski, Marcin. 2017.
    `Microcalcification Segmentation from Mammograms: A Morphological Approach.`
    Args:
        img (np.array, 2-dim): breast image
        thr (float, 0.0-256.0, optional): for thresholding the filters
        h (float, optional): defined in paper 
    Returns:
        mask: binary np.array
    """
    ### Stage 1
    #### Step 1
    i2 = img*(img>21/255)
    #### Step 2
    #pyramid
    img1 = opening(closing(img, square(3)),square(3))
    img1 = pyramid_reduce(img1, multichannel=False)
    img2 = opening(closing(img1, square(3)),square(3))
    img2 = pyramid_reduce(img2, multichannel=False)
    # filters
    filter1 = img-np.minimum(pyramid_expand(img1, multichannel=False), img)
    filter2 = img-np.minimum(pyramid_expand(img2,upscale=4, multichannel=False), img)
    #thresholding
    filter1 = filter1>(thr/256.)
    filter1 = remove_small_objects(filter1,3)
    filter2 = filter2>(thr/256.)
    filter2 = remove_small_objects(filter2,3)
    #### Step 3
    # extended max
    iemax = emax(i2,h)

    #### Step 4
    marker1 = np.logical_and(filter1,iemax)

    marker2 = np.logical_and(filter2,iemax)

    result1 = reconstruction(marker1, iemax, method='dilation')
    result2 = reconstruction(marker2, iemax, method='dilation')
    result = np.logical_or(result1,result2)
    #### Filter objects
#     result = remove_small_objects(result,5)
#     small_objects = remove_small_objects(result,100)
#     result = np.logical_xor(result,small_objects)
    ### Stage 2
    # CO filtering
    img_co = opening(closing(i2, square(3)),square(3))
    # Inversion
    img_co = 1.-img_co

    img_min = rmin(img_co)

    # Internal marker
    int_marker = np.logical_and(img_min,result) 
    # Gradient of inverted image
    img_grad = grad(img_co) 
    # External marker with watershed
    ws_markers_bool = extrema.local_minima(img_co)
    ws_markers = ndimage.label(ws_markers_bool)[0]
    ext_marker = find_boundaries(watershed(img_co, markers=ws_markers), mode='outer')
    # Dilating external marker
    ext_marker_dil = binary_dilation(ext_marker,square(3))
    # Finding non-intersecting part of internal marker with external dilated marker
    intersection = np.logical_and(int_marker,ext_marker_dil)
    int_marker2 = np.logical_xor(int_marker,intersection)
    # Combine markers
    marker = np.logical_or(int_marker2,ext_marker)
    
    # Minimum imposition on gradient
    img_min_impo = min_imposition(img_grad,marker)
    # Watershed segmentation
    img_watershed = watershed(img_min_impo, markers=ndimage.label(int_marker2)[0])
    max_area=5000
    label_list = list()
    for rprop in regionprops(img_watershed):
        if rprop.area<=max_area:
            label_list.append(rprop.label)
    final_mask = np.isin(img_watershed,label_list)
    return final_mask


def detect_calcifications_whole_image(img, thr=10., erosion=10, method='pyramid'):
    """Morphological segmentation of calcifications
    """
    patches, padding = create_patches(img, (500,500))
    mask_patches = np.zeros_like(patches,dtype=int) 
    n_rows, n_columns = patches.shape[0], patches.shape[1]
    for i in range(0,n_rows):
        for j in range(0,n_columns):
            if method=='pyramid':
                mask_patches[i,j]=detect_calcifications(patches[i,j], thr=thr)
            else:
                mask_patches[i,j]=detect_calcifications_Ciecholewski(patches[i,j], thr=thr)
    #recostruct image
    mask_calc = image_from_patches(mask_patches, padding)
    mask_breast = delineate_breast(img, bdry_remove=0.)
    mask_breast = binary_erosion(mask_breast,disk(erosion))
    mask = np.logical_and(mask_calc,mask_breast)
    mask = ndimage.binary_fill_holes(mask)
    return mask


def image_from_patches(patches, padding):
    """Gluing patches to output whole image. 
    """
    h_patch, w_patch = patches.shape[2], patches.shape[3]
    n_rows, n_columns = patches.shape[0], patches.shape[1]
    h_pad, w_pad = padding
    # initialize image
    img = np.zeros((n_rows*h_patch, n_columns*w_patch))
    # combine patches
    for i in range(0,n_rows):
        for j in range(0,n_columns):
            top = i*h_patch
            bottom = (i+1)*h_patch
            left = j*w_patch
            right = (j+1)*w_patch
            img[top:bottom,left:right] = patches[i,j]
    # remove added padding
    h, w = img.shape
    img = img[0:h-h_pad,0:w-w_pad]
    
    return img

def create_patches(img, patch_size):
    """Creating patches, padding appropriately the image first. 
    """
    # add padding
    h, w = img.shape
    h_patch, w_patch = patch_size
    h_pad, w_pad = h_patch-np.mod(h,h_patch), w_patch-np.mod(w,w_patch)
    img_pad = np.zeros((h+h_pad, w+w_pad))
    img_pad[0:h,0:w] = img
    # create patches
    n_rows = int((h+h_pad)/h_patch)
    n_columns = int((w+w_pad)/w_patch)
    #initialize
    patches = np.zeros((n_rows,n_columns,patch_size[0], patch_size[1]))
    for i in range(0,n_rows):
        for j in range(0,n_columns):
            top = i*patch_size[0]
            bottom = (i+1)*patch_size[0]
            left = j*patch_size[1]
            right = (j+1)*patch_size[1] 
            patches[i,j] = img_pad[top:bottom,left:right]
    return patches, (h_pad, w_pad)


def detect_calcifications(img, thr=10.):
    """Morphological calcification detector, using pyramid scheme
    Args:
        img (np.array, 2-dim): breast image
        thr (float, 0.0-256.0, optional): for thresholding the filters
    Returns:
        mask: binary np.array
    """
    #pyramid
    img1 = opening(closing(img, square(3)),square(3))
    img1 = pyramid_reduce(img1, multichannel=False)
    img2 = opening(closing(img1, square(3)),square(3))
    img2 = pyramid_reduce(img2, multichannel=False)
    # filters
    filter1 = img-np.minimum(pyramid_expand(img1, multichannel=False), img)
    filter2 = img-np.minimum(pyramid_expand(img2,upscale=4, multichannel=False), img)
    #thresholding
    filter1 = filter1>(thr/256.)
    filter2 = filter2>(thr/256.)
    #combine
    filter_comb = np.logical_or(filter1,filter2)

    mask = remove_small_objects(filter_comb, min_size=3)
    mask = dilation(mask,disk(1))
    return mask

def emax(image, h=5):
    """Finds extended maximum
    """
    image = (image*255).astype(int)
    hmax = reconstruction(image-h, image, method='dilation')
    rmax =  hmax+1-reconstruction(hmax, hmax+1, method='dilation')
    return rmax

def rmin(image):
    image = (image*255).astype(int)
    return reconstruction(image+1, image, method='erosion')-image

def grad(image):
    return dilation(image,square(3))-erosion(image,square(3))

def min_imposition(image, marker):
    fm = (~marker)*image.max()
    pm = np.minimum(fm, image) #pointwise minimum
    return reconstruction(fm, pm, method='erosion')
