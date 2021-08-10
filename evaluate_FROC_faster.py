# For evaluating HDoGReg with FROC analysis

import matplotlib
# To avoid using Xwindows backend
matplotlib.use('Agg') 
import ray
import os
from tqdm import tqdm
from math import log
import numpy as np

from ruamel.yaml import YAML
import torch
from skimage.morphology import remove_small_holes, binary_erosion, square, remove_small_objects
from skimage.measure import regionprops
from skimage.feature import blob_dog, peak_local_max
from skimage.util import img_as_float
from skimage.feature.blob import _prune_blobs
from scipy import ndimage
from hdogreg.image import  overlap_based_combining, hybrid_approach
from hdogreg.dataset import ImageDataset, bboxInImage
from hdogreg.metrics import get_confusion_matrix_2
from hdogreg.utils import arg_parsing

import pandas as pd
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
import torch
from skimage.draw import circle
import mlflow
import cv2
from sklearn.metrics import auc

from PIL import Image


def bboxIoU(x1,y1,x2,y2,x1b,y1b,x2b,y2b):
    '''
    x1,y1,x2,y2 are numbers
    x1b,y1b,x2b,y2b are np arrays 1-dim
    '''
    x1b,y1b,x2b,y2b = np.array(x1b),np.array(y1b),np.array(x2b),np.array(y2b)
    dx = np.minimum(x2,x2b)-np.maximum(x1,x1b)
    dy = np.minimum(y2,y2b)-np.maximum(y1,y1b)
    overlap = np.maximum(0,dx)*np.maximum(0,dy)
    union = (x2-x1)*(y2-y1)+(x2b-x1b)*(y2b-y1b)-overlap
    return overlap/union

def logging_params(params_list):
    for name, value in params_list:
        mlflow.log_param(name, value)

def logging_metrics(metrics_dict):
    for name, value in metrics_dict.items():
        mlflow.log_metric(name, value)

def make_froc_data(data_computed, total_area=None):
    if total_area is not None:
        per_unit_area=True
    else:
        per_unit_area=False
    
    n_samples=data_computed.shape[0]
    tp_all,fp_all,fn_all = np.array(data_computed).sum(axis=0)
    # The true positive rate
    llf = tp_all/(tp_all+fn_all)
    # The nlf
    if per_unit_area:
        nlf = fp_all/total_area
    else:
        nlf = fp_all/n_samples


    return list(nlf), list(llf)

def find_auc(FP,TPR, max_FP=1):
    x,y = FP,TPR
    x, y = np.array(x), np.array(y)
    x, y = x[x<=max_FP], y[x<=max_FP]
    x, y = np.append(x,0), np.append(y,0)
    x, y = x[np.argsort(x)], y[np.argsort(x)]
    x, y = np.append(x,max_FP), np.append(y,y[-1])
    
    return auc(x,y)


def make_froc_graphs(pickle_files, data_names,colors, max_FP = 10, bootstraps=None ,save=None):
    froc_data=dict()
    paucs = list()
    plt.figure(figsize=(10,7))
    plt.title('FROC curve', fontsize=20)
    plt.xlabel(r'FP per unit area ($cm^2$)', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('FROC curve', fontsize=20)
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.2', color='black')
    plt.ylim(0.0,1.0)
    plt.xlim(0.0,max_FP)
    plt.tight_layout()
    for pickle_file, name,clr in zip(pickle_files, data_names, colors):
        data_dict=pickle.load(open(pickle_file,'rb'))
        data_computed = data_dict['array']
        thr_list =  data_dict['thresholds']
        resolution = .007
        data_dict['breast_area']=np.array(data_dict['breast_area'])
        total_area = sum(data_dict['breast_area'])
        froc_data[name] = make_froc_data(data_computed, total_area)
        froc_data[name] = (froc_data[name][0], froc_data[name][1])
        plt.plot(*froc_data[name],color=clr)
        if bootstraps is None:
            paucs.append(find_auc(*froc_data[name], max_FP=max_FP))
        else:
            paucs.append(bootstrap_pauc(data_computed,max_FP=max_FP,breast_area=data_dict['breast_area'], bootstraps=bootstraps))
            y_errors = bootstrap_froc(data_computed,max_FP=max_FP,breast_area=data_dict['breast_area'], bootstraps=bootstraps)
            plt.fill_between(froc_data[name][0], 
                             froc_data[name][1]-y_errors,
                             froc_data[name][1]+y_errors, 
                             alpha=0.2, color=clr)
    if bootstraps is None:
        pauc = paucs[0]
    else:
        pauc = paucs[0][0]
    print(paucs)
    plt.legend(data_names, loc=4, frameon=False)
    if save is not None:
        plt.savefig(save, dpi=200)
        print(f'FILE SAVED. FROC curve was saved at: {save}')
    plt.show()
    return pauc
    


    
    
def bootstrap_pauc(data, max_FP=1, breast_area=None,  bootstraps=1000, seed=0):
    aucs = list()
    np.random.seed(seed)
    for i in range(bootstraps):
        idx_sampled = np.random.choice(np.arange(data.shape[0]), 
                                        size=data.shape[0],
                                       replace=True)
        data_sampled=data[idx_sampled]
        if breast_area is not None:
            breast_area_samples=breast_area[idx_sampled]
            total_area=sum(breast_area_samples)
        froc_data = make_froc_data(data_sampled, total_area)
        aucs.append(find_auc(*froc_data, max_FP=max_FP))
    return np.mean(aucs), np.std(aucs)

def bootstrap_froc(data, max_FP=1, breast_area=None,  bootstraps=1000, seed=0):
    froc_list = list()
    np.random.seed(seed)
    for i in range(bootstraps):
        idx_sampled = np.random.choice(np.arange(data.shape[0]), 
                                        size=data.shape[0],
                                       replace=True)
        data_sampled=data[idx_sampled]
        if breast_area is not None:
            breast_area_samples=breast_area[idx_sampled]
            total_area=sum(breast_area_samples)
        froc_data = make_froc_data(data_sampled, total_area)
        froc_list.append(froc_data)
    froc_list = np.array(froc_list)
    return np.std(froc_list[:,1,:],axis=0)

def run():
    # Load configuration contents
    config_fname = os.path.abspath(os.path.dirname(os.path.realpath(__file__))+'/validation_config.yaml')
    with open(config_fname) as config_f:
        config_dict = YAML().load(config_f)['confusion_matrix']

    # Update config settings using args
    config_dict = arg_parsing(config_dict)
    
    data_dir = config_dict['data_dir']
    thr_list = config_dict['threshold']
    csv_file = config_dict['val_csv']
    model_name = config_dict['model_name']
    radius = float(config_dict['radius'])
    alpha = float(config_dict['alpha'])
    save_dir = config_dict['save_dir']
    mask_col = config_dict['mask_col']
    num_workers = int(config_dict['num_workers'])

    out_csv_file = config_dict['out_csv_file']
    small_area = float(config_dict['small_area'])
    distance_thr = float(config_dict['distance_thr'])
    iou_thr = float(config_dict['iou_thr'])
    resolution = float(config_dict['resolution'])
    per_unit_area = int(config_dict['per_unit_area'])
    hybrid_combining = config_dict['hybrid_combining']
    hybrid_combining_overlap = float(config_dict['hybrid_combining_overlap'])
    regression = int(config_dict['regression'])


    hessian_thr = float(config_dict['hessian_thr'])
    comment = config_dict['comment']
    graph_only = int(config_dict['graph_only'])

    blob_dir = os.path.join(save_dir,'blob/')
    breast_dir = os.path.join(save_dir,'breast_segm/')
    pred_fpn_dir = os.path.join(save_dir,'predsFPN/')


    if not graph_only:
        logging_params([
        ("data_dir", os.path.basename(os.path.dirname(data_dir))),
        ('val_csv', csv_file), 
        ('model_name',model_name),
        ('radius', radius),
        ('alpha', alpha),
        ('hessian_thr', hessian_thr),
        ('comment',comment),
        ('out_csv_file',out_csv_file)
        ])


        ##########################################################
        # Load dataset
        print("\n------Loading Dataset------\n")
        ds = ImageDataset(data_dir, csv_file, img_format='png16',
                          images_list=[mask_col],
                         attributes_list=['full_image']
                        )
        print("\n------Initializing Ray------\n")
        ray.init(num_cpus=num_workers)#, webui_host='127.0.0.1')
        # put dataset on remote
        ds_id = ray.put(ds)
        ##########################################################
        # Defining the remote function, after ds in the remote
        @ray.remote
        def main_func(idx, ds_id):
            gt_mask, img_name = ds_id.__getitem__(idx)
            # Loading the segmentations of blobs and the breast mask
            blob_path = os.path.join(blob_dir,img_name+'.png')
            blob_mask = ds_id._img_to_numpy(blob_path)>0
            pred_fpn_path = os.path.join(pred_fpn_dir,img_name+'.png')
            # Open as uint32 array
            pred = np.array(Image.open(pred_fpn_path))

            overlap = hybrid_combining_overlap
            blob_data=list()


            lbl, n_blobs = ndimage.label(blob_mask)
            blob_rprops = regionprops(lbl)
            for rprop in blob_rprops:
                y1,x1,y2,x2=rprop.bbox
                y,x=rprop.centroid
                blob_mask_patch = blob_mask[y1:y2,x1:x2]
                pred_patch = pred[y1:y2,x1:x2]
                masked_data=pred_patch[blob_mask_patch]
                score = np.percentile(masked_data,(1-overlap)*100)
                blob_data.append({
                    'blob_id':rprop.label,
                    'y':y,
                    'x':x,
                    'score':score,
                    'x1':x1, 'y1':y1,'x2':x2,'y2':y2
                })
            blob_df = pd.DataFrame(blob_data)
            blob_df['gt_id']=np.NaN

            gt_lbl, n_gt = ndimage.label(gt_mask)
            gt_props = regionprops(gt_lbl)
            for rprop in gt_props:
                y1,x1,y2,x2=rprop.bbox
                y,x=rprop.centroid
                gt_id=rprop.label
                sq_distances = np.array((blob_df.x-x)**2+(blob_df.y-y)**2) # For distance criterion
                iou_list = bboxIoU(x1,y1,x2,y2,blob_df.x1,blob_df.y1,blob_df.x2,blob_df.y2)
                if np.min(sq_distances)<=distance_thr**2:
                    min_idx = np.argmin(sq_distances)
                    blob_record = blob_df.iloc[min_idx]
                    if not blob_record.gt_id in blob_df['gt_id']:
                        blob_df['gt_id'].iloc[min_idx]=gt_id
                elif np.max(iou_list)>=iou_thr:
                    max_idx = np.argmax(iou_list)
                    blob_record = blob_df.iloc[max_idx]
                    if not blob_record.gt_id in blob_df['gt_id']:
                        blob_df['gt_id'].iloc[max_idx]=gt_id
            # Compute tp,fp,fn counts on all thresholds           
            blob_df_fp = blob_df[blob_df.gt_id.isna()]
            blob_df_tp = blob_df[~blob_df.gt_id.isna()]
            threshold = np.arange(0,2**16+2)
            counts_fp, _ = np.histogram(np.array(blob_df_fp.score),bins=threshold)
            counts_tp, _ = np.histogram(np.array(blob_df_tp.score),bins=threshold)
            fp = np.cumsum(counts_fp[::-1])[::-1]
            tp = np.cumsum(counts_tp[::-1])[::-1]
            fn = n_gt-tp
            return tp,fp,fn
            

        out_data=[main_func.remote(idx,ds_id) for idx in range(ds.__len__())]
        out_data=ray.get(out_data)
        data_dict = {'array': np.array(out_data),
                    'thresholds':np.arange(0,2**16+1),
                    }



        ds = ImageDataset(data_dir, csv_file, img_format='png16',
                      images_list=['full_image'],
                     attributes_list=['full_image'])
        image_list = list(ds.df.full_image)
        if per_unit_area:
            @ray.remote
            def area_func(idx,ds_id):
                # Loading the segmentations of blobs and the breast mask
                img_name = image_list[idx]
                breast_path = os.path.join(breast_dir,img_name+'.png')
                breast_mask = ds_id._img_to_numpy(breast_path)
                breast_area = breast_mask.sum()
                # Area in cm^2
                breast_area = breast_area*resolution**2
                return breast_area
            data = list()
            for idx in range(0,ds.__len__()):
                data.append(area_func.remote(idx,ds_id))
            data = ray.get(data)
            total_area = np.sum(data)
            print("Overall area {} cm^2".format(total_area))
            data_dict['breast_area']=list(data)
        ray.shutdown()

        # Saving computations
        pickle_file = os.path.join(save_dir, out_csv_file+'FROC.p')
        pickle.dump(data_dict, open( pickle_file, "wb" ) )

    # Build FROC curve
    if graph_only:
        # Load data
        pickle_file = os.path.join(save_dir, out_csv_file+'FROC.p')
        data_dict = pickle.load( open( pickle_file, "rb" ) )


    save_path = os.path.join(save_dir,'{}FROC.png'.format(out_csv_file))
    if per_unit_area:
        pauc=make_froc_graphs([pickle_file],[model_name], ['blue'],
            max_FP=1, bootstraps=100, save=save_path)


    logging_metrics({
            'pauc':pauc,
            })

    print(f'RESULT FOUND. partial area under curve (pAUC) = {pauc}')


if __name__ == '__main__':
    mlflow.set_experiment('EvaluateFROC')
    with mlflow.start_run():
        run()

