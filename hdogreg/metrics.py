import numpy as np
from skimage.morphology import remove_small_holes,\
    remove_small_objects, binary_erosion, disk, binary_dilation
from skimage.measure import regionprops
from scipy import ndimage


def get_confusion_matrix_2(pred_mask, gt_mask, 
                           distance_thr = 5, IoU_thr=0.3, small_object_area = 1, 
                           return_mask = False, return_dot_results=False):
    '''
    Strategy:
    1. Iterate over GT objects and determine which are the TP. Every time a detection
    occurs eliminate the object responsible from the Prediction
    2. All GT objects not detected are FN
    2. Remaining objects in prediction are FP
    '''
    # Cast to boolean
    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)
    
    # Denoise prediction
    pred_mask = remove_small_holes(pred_mask)
    #pred_mask = remove_small_objects(pred_mask, min_size=2)
    
    # Finding objects in masks
    label_gt, n_objects = ndimage.label(gt_mask)
    label_pred, n_preds = ndimage.label(pred_mask)
    props_gt = regionprops(label_gt)
    props_pred = regionprops(label_pred)
    # Unique pred labels
    pred_unique_labels = np.unique(label_pred)
    pred_unique_labels = np.setdiff1d(pred_unique_labels,np.array([0])) # to remove 0
    pred_unique_labels = list(pred_unique_labels)

    # Initialize
    tp, fp, fn = 0, 0, 0
    tp_objects = []
    fp_objects = []
    fn_objects = []

    tp_dot, fn_dot = 0,0
           
    # Perform detection 
    for i in range(0, len(props_gt)):
        prop_gt = props_gt[i]
        gt_area = prop_gt.area
        detection, gt_object_label,props_pred = small_object_criterion_2(prop_gt, props_pred, 
                label_pred, label_gt, distance_thr, pred_unique_labels=pred_unique_labels)
        if not detection:
            detection, gt_object_label,props_pred = large_object_criterion_2(prop_gt, props_pred, 
                label_gt, label_pred, IoU_thr, pred_unique_labels=pred_unique_labels)

        if detection:
            tp_objects.append(gt_object_label)
            tp += 1
            if gt_area==1.:
                tp_dot += 1
        else:
            fn_objects.append(gt_object_label)
            fn += 1
            if gt_area==1.:
                fn_dot += 1
    fp=n_preds-tp
        
    # Compute masks
    if return_mask:
        tp_mask = np.zeros_like(gt_mask)
        for idx in tp_objects:
            tp_mask[label_gt==idx] = 1
            
        fn_mask = np.zeros_like(gt_mask)
        for idx in fn_objects:
            fn_mask[label_gt==idx] = 1
            
        fp_mask = np.zeros_like(gt_mask)
        for idx in fp_objects:
            fp_mask[label_pred==idx] = 1
            
        return tp,fp,fn, tp_mask, fp_mask, fn_mask
    if return_dot_results:
        return tp,fp,fn,tp_dot,fn_dot
    return tp,fp,fn

def small_object_criterion_2(prop_gt, props_pred, label_pred, label_gt, distance_thr, pred_unique_labels):
    """
    Based on distance between centroids
    """
    #initialize
    min_distance = 10000
    d = distance_thr
    # Look on a patch around gt centroid
    h,w=label_pred.shape
    x_gt, y_gt = prop_gt.centroid
    bb=(max(0,int(x_gt)-d-2),min(h,int(x_gt)+d+2),max(0,int(y_gt)-d-2),min(w,int(y_gt)+d+2))
    bb = [int(b) for b in bb]
    lbl_patch = label_pred[bb[0]:bb[1],bb[2]:bb[3]]
    # Find pred unique labels within patch and their idx in regioprops list
    pred_labels = np.unique(lbl_patch)
    pred_labels = np.setdiff1d(pred_labels,np.array([0])) # to remove 0 if exists
    pred_labels = np.intersect1d(pred_labels,np.array(pred_unique_labels))
    pred_idx = pred_labels-1
    # Finding minimum distance between centroids
    for j in pred_idx:
        prop_pred = props_pred[j]
        x_pred, y_pred = prop_pred.centroid
        distance = np.sqrt((x_gt-x_pred)**2+(y_gt-y_pred)**2)
        if distance<=min_distance:

            min_distance = distance
            closest_object_label = j+1

    # condition
    detection=False
    if min_distance<=float(distance_thr):
        if pred_unique_labels:
            if closest_object_label in pred_unique_labels:
                detection = True
                pred_unique_labels.remove(closest_object_label)

        
    return detection, prop_gt.label, props_pred

def large_object_criterion_2(prop_gt, props_pred, label_gt, label_pred, IoU_thr, pred_unique_labels=None):
    """
    Based on IoU
    """
    # initialize
    max_IoU = 0.0
    # find max IoU
    gt_bb = prop_gt.bbox
    gt_bb = (gt_bb[0],gt_bb[2],gt_bb[1],gt_bb[3])
    gt_label = prop_gt.label
    # find pred labels it overlaps with
    lbl_patch = label_pred[gt_bb[0]:gt_bb[1],gt_bb[2]:gt_bb[3]]
    # Find pred unique labels within patch and their idx in regioprops list
    pred_labels = np.unique(lbl_patch)
    pred_labels = np.setdiff1d(pred_labels,np.array([0])) # to remove 0 if exists
    pred_labels = np.intersect1d(pred_labels,np.array(pred_unique_labels))
    pred_idx = pred_labels-1
    for j in pred_idx:
        prop_pred = props_pred[j]
        pred_bb = prop_pred.bbox
        pred_bb = (pred_bb[0],pred_bb[2],pred_bb[1],pred_bb[3])
        pred_label = prop_pred.label
        if bboxOverlap(gt_bb,pred_bb):
            bb = combineBboxes(pred_bb,gt_bb)
            pred_patch = label_pred[bb[0]:bb[1],bb[2]:bb[3]]==pred_label
            gt_patch = label_gt[bb[0]:bb[1],bb[2]:bb[3]]==gt_label
            iou = IoU(pred_patch,gt_patch)
            if iou>max_IoU:
                max_IoU=iou
                closest_object_label = j+1
    detection = False
    if max_IoU>=IoU_thr:
        if pred_unique_labels:
            if closest_object_label in pred_unique_labels:
                detection = True
                pred_unique_labels.remove(closest_object_label)

        
    return detection, prop_gt.label, props_pred


def bboxOverlap(bb1, bb2):
    
    x1,x2,y1,y2 = bb1
    x1b,x2b,y1b,y2b = bb2
    
    x_criterion = (x2 >= x1b) and (x2b >= x1)
    y_criterion = (y2 >= y1b) and (y2b >= y1)
    
    if x_criterion and y_criterion:
        return True
    return False


def combineBboxes(bb1,bb2):
    x1,x2,y1,y2 = bb1
    x1b,x2b,y1b,y2b = bb2
    
    bb = min(x1,x1b), max(x2,x2b), min(y1,y1b), max(y2,y2b)
    return bb


def IoU(y_pred, y_true):
    '''Returns IoU between two binary images in np format
    '''
    intersection = np.logical_and(y_pred,y_true)
    union = np.logical_or(y_pred,y_true)
    intersection = intersection.sum()
    union = union.sum()
    if union == 0:
        return 0
    return intersection/union


def mIoU(y_pred, y_true):
    '''Returns IoU between two binary images in np format
    '''
    iou1 = IoU(y_pred, y_true)
    iou2 = IoU(~y_pred, ~y_true)

    return (iou1+iou2)/2


def get_iou_per_object(pred_mask, gt_mask):
    '''
    Strategy:
    1. Iterate over GT objects and find max IoU with a predicted object for each
    '''
    # Cast to boolean
    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)
    
    # Denoise prediction
    # pred_mask = remove_small_holes(pred_mask)
    
    # Finding objects in masks
    label_gt, n_objects = ndimage.label(gt_mask)
    label_pred, n_preds = ndimage.label(pred_mask)
    props_gt = regionprops(label_gt)
    props_pred = regionprops(label_pred)

    data_dict=dict()
    for i in range(0, len(props_gt)):
        data_dict[i] = dict()
        prop_gt = props_gt[i]
        gt_area = prop_gt.area
        iou = max_iou(prop_gt, props_pred, label_gt, label_pred)
        data_dict[i]['area']=gt_area
        data_dict[i]['iou']=iou

    return data_dict



def max_iou(prop_gt, props_pred, label_gt, label_pred):
    """
    Based on IoU
    """
    # initialize
    max_IoU = 0.0
    
    # find max IoU
    gt_bb = prop_gt.bbox
    gt_bb = (gt_bb[0],gt_bb[2],gt_bb[1],gt_bb[3])
    gt_label = prop_gt.label
    for j in range(0, len(props_pred)):
        prop_pred = props_pred[j]
        pred_bb = prop_pred.bbox
        pred_bb = (pred_bb[0],pred_bb[2],pred_bb[1],pred_bb[3])
        pred_label = prop_pred.label
        if bboxOverlap(gt_bb,pred_bb):
            bb = combineBboxes(pred_bb,gt_bb)
            pred_patch = label_pred[bb[0]:bb[1],bb[2]:bb[3]]==pred_label
            gt_patch = label_gt[bb[0]:bb[1],bb[2]:bb[3]]==gt_label
            iou = IoU(pred_patch,gt_patch)
            if iou>max_IoU:
                max_IoU=iou
                closest_object = j
                    
    return max_IoU