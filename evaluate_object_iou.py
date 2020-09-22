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
from hdogreg.image import overlap_based_combining, hybrid_approach
from hdogreg.dataset import ImageDataset, bboxInImage
from hdogreg.metrics import get_iou_per_object
import pandas as pd
import argparse

import seaborn as sns
import matplotlib.pyplot as plt
import torch
from skimage.draw import circle
import mlflow
import pickle

def logging_params(params_list):
    for name, value in params_list:
        mlflow.log_param(name, value)

def logging_metrics(metrics_dict):
    for name, value in metrics_dict.items():
        mlflow.log_metric(name, value)


def arg_parsing(config_dict):
    arg_parser = argparse.ArgumentParser()
    for key in config_dict.keys():
        if key == 'threshold':
            # threshold is a list of threshold values
            arg_parser.add_argument('-{}'.format(key), nargs='+', type=float) 
        else: 
            arg_parser.add_argument('-{}'.format(key))
    args = vars(arg_parser.parse_args())
    for key in config_dict.keys():
        if args[key] is not None:
            config_dict[key] = args[key]
    return config_dict


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
    hybrid_combining = config_dict['hybrid_combining']
    hybrid_combining_overlap = float(config_dict['hybrid_combining_overlap'])
    regression = int(config_dict['regression'])
    pred_dir = config_dict['pred_dir']
    use_pred_dir = int(config_dict['use_pred_dir'])
    if not regression:
        thr_list = np.arange(0,1.,0.05) # classification thresholds
        thr_list = np.concatenate((np.logspace(-6,-1,5),np.linspace(0,0.9,10), 1-np.logspace(-6,-1,5)))
    else:
        thr_list = np.linspace(0,radius,20)
        thr_list = [(np.exp(alpha*(1-d/radius))-1)/(np.exp(alpha)-1) for d in thr_list]
        
    out_csv_file = config_dict['out_csv_file']
    small_area = float(config_dict['small_area'])
    distance_thr = float(config_dict['distance_thr'])
    iou_thr = float(config_dict['iou_thr'])
    evaluate_dots_separately = config_dict['evaluate_dots_separately']

    DoG_thr = float(config_dict['DoG_thr'])
    min_sigma = float(config_dict['min_sigma'])
    max_sigma = float(config_dict['max_sigma'])
    overlap = float(config_dict['overlap'])
    hessian_thr = float(config_dict['hessian_thr'])

    comment = config_dict['comment']
    graph_only = int(config_dict['graph_only'])

    blob_dir = os.path.join(save_dir,'blob/')
    breast_dir = os.path.join(save_dir,'breast_segm/')
    pred_fpn_dir = os.path.join(save_dir,'predsFPN/')


    logging_params([
    ("data_dir", os.path.basename(os.path.dirname(data_dir))),
    ('val_csv', csv_file), 
    ('model_name',model_name),
    ('radius', radius),
    ('alpha', alpha),
    ('min_sigma', min_sigma), 
    ('max_sigma', max_sigma),
    ('overlap', overlap),
    ('DoG_thr', DoG_thr),
    ('hessian_thr', hessian_thr),
    ('comment',comment),
    ('out_csv_file',out_csv_file)
    ])


    ##########################################################
    # Load dataset
    print("\n------Loading Dataset------\n")
    ds = ImageDataset(data_dir, csv_file, img_format='png16',
                      images_list=['full_image',
                                   mask_col],
                     attributes_list=['full_image']
                    )
    print("\n------Initializing Ray------\n")
    ray.init(num_cpus=num_workers, webui_host='127.0.0.1')
    # put dataset on remote
    ds_id = ray.put(ds)
    ##########################################################
    # Defining the remote function, right after ds in the remote
    @ray.remote
    def main_func(idx, thr_list, ds_id):
        img, mask, img_name = ds_id.__getitem__(idx)
        # Loading the segmentations of blobs and the breast mask
        blob_path = os.path.join(blob_dir,img_name+'.png')
        breast_path = os.path.join(breast_dir,img_name+'.png')
        blob_mask = ds_id._img_to_numpy(blob_path)
        breast_mask = ds_id._img_to_numpy(breast_path)
        pred_fpn_path = os.path.join(pred_fpn_dir,img_name+'.png')
        pred_fpn = ds_id._img_to_numpy(pred_fpn_path)
        # transforming CNN prediction to radial space
        data_list = list()
        for thr in thr_list:
            pred_mask = pred_fpn >= thr
            pred_mask = hybrid_approach(pred_mask,breast_mask,blob_mask,
                    hybrid_combining=hybrid_combining, 
                    hybrid_combining_overlap=hybrid_combining_overlap)
            data_list.append(get_iou_per_object(pred_mask, mask))

        return data_list

    data = list()

    n_data = ds.__len__()
    for idx in tqdm(range(0,n_data)):
        data.append(main_func.remote(idx, thr_list, ds_id))
    data = ray.get(data)
    out_file = os.path.join(save_dir, out_csv_file)
    pickle_dict = {
        'thresholds': thr_list,
        'iouperobject': data
    }
    pickle.dump( pickle_dict, open( out_file+"IoUperobject.p", "wb" ) )
    ray.shutdown()

    # Plot the graph as a boxplot
    object_ious = dict()
    for thr in thr_list:
        object_ious[thr] = list()
    for img_id in range(len(data)):
        for (thr_data,thr) in zip(data[img_id],thr_list):
            for i in range(len(thr_data)):
                if thr_data[i]['area']>1.:
                    object_ious[thr].append(thr_data[i]['iou'])
    df = pd.DataFrame(object_ious)
    plt.figure(figsize=(20,10))
    plt.xlabel('IoU per Object (Non-dot)', fontsize=15)
    plt.ylabel('Threshold', fontsize=15)
    plt.xticks(fontsize=15); plt.yticks(fontsize=15)
    sns.boxplot(data=df, orient='h', color = 'yellow')
    plt.savefig(os.path.join(save_dir,'{}IoUperobject.png'.format(out_csv_file)),dpi=200)
    plt.close()

    
    logging_metrics({
        'iou_max':df.mean().max()
        })
    print(df.mean().max())


if __name__ == '__main__':
    mlflow.set_experiment('EvaluateIoUperObject')
    with mlflow.start_run():
        run()
