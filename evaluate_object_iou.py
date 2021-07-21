# For evaluating HDoGReg computing IoU per object

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
from hdogreg.utils import arg_parsing
import pandas as pd
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


def run():
    # Load configuration contents
    config_fname = os.path.abspath(os.path.dirname(os.path.realpath(__file__))+'/validation_config.yaml')
    with open(config_fname) as config_f:
        config_dict = YAML().load(config_f)['confusion_matrix']

    # Update config settings using args
    config_dict = arg_parsing(config_dict)
    
    data_dir = config_dict['data_dir']
    csv_file = config_dict['val_csv']
    save_dir = config_dict['save_dir']
    mask_col = config_dict['mask_col']
    num_workers = int(config_dict['num_workers'])

    pred_dir = os.path.join(save_dir,'hybrid_approach/')


    logging_params([
    ("data_dir", os.path.basename(os.path.dirname(data_dir))),
    ('val_csv', csv_file), 
    ])


    ##########################################################
    # Load dataset
    print("\n------Loading Dataset------\n")
    ds = ImageDataset(data_dir, csv_file, img_format='png16',
                      images_list=[
                                   mask_col],
                     attributes_list=['full_image']
                    )
    print("\n------Initializing Ray------\n")
    ray.init(num_cpus=num_workers)#, webui_host='127.0.0.1')
    # put dataset on remote
    ds_id = ray.put(ds)
    ##########################################################
    # Defining the remote function, right after ds in the remote
    @ray.remote
    def main_func(idx, ds_id):
        mask, img_name = ds_id.__getitem__(idx)
        pred_mask_path = os.path.join(pred_dir,img_name+'.png')
        pred_mask = ds_id._img_to_numpy(pred_mask_path)

        return get_iou_per_object(pred_mask, mask)

    data = list()

    n_data = ds.__len__()
    for idx in tqdm(range(0,n_data)):
        data.append(main_func.remote(idx, ds_id))
    data = ray.get(data)
    out_file = os.path.join(save_dir, 'IoUperobject.pkl')
    pickle_dict = {
        'iouperobject': data
    }

    pickle.dump(pickle_dict, open(out_file, "wb" ) )
    ray.shutdown()

    # Plot the graph as a boxplot
    object_ious = list()
    for data_img in data:
            for mc_id in data_img.keys():
                if data_img[mc_id]['area']>1.:
                    object_ious.append(data_img[mc_id]['iou'])
    mean_iou = np.mean(object_ious)
    
    logging_metrics({
        'iou_per_object':mean_iou,
        })
    print(f'RESULT FOUND. IoU per Object = {mean_iou}')


if __name__ == '__main__':
    mlflow.set_experiment('EvaluateIoUperObject')
    with mlflow.start_run():
        run()
