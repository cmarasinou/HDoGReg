# For combining FPN and blob segmentation results to give final segmentation
# A fixed operating threshold is applied on the FPN output

import ray
import os
from tqdm import tqdm
from math import log
import numpy as np
import argparse
from ruamel.yaml import YAML

from hdogreg.dataset import ImageDataset, png16_saver
from hdogreg.image import hybrid_approach
from hdogreg.utils import arg_parsing


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
    num_workers = int(config_dict['num_workers'])
    model_name = config_dict['model_name']
    regression = int(config_dict['regression'])
    thr = float(config_dict['threshold'])

    blob_dir = os.path.join(save_dir,'blob/')
    breast_dir = os.path.join(save_dir,'breast_segm/')
    pred_fpn_dir = os.path.join(save_dir,'predsFPN/')

    hybrid_combining = config_dict['hybrid_combining']
    hybrid_combining_overlap = float(config_dict['hybrid_combining_overlap'])

    hybrid_approach_dir = os.path.join(save_dir, 'hybrid_approach/')
    os.makedirs(hybrid_approach_dir, exist_ok=True)
    ##########################################################
    # Load dataset
    print("\n------Loading Dataset------\n")
    ds = ImageDataset(data_dir, csv_file, img_format='png16',
                      images_list=['full_image'],
                     attributes_list=['full_image']
                    )
    print("\n------Initializing Ray------\n")
    ray.init(num_cpus=num_workers, webui_host='127.0.0.1')
    # put dataset on remote
    ds_id = ray.put(ds)
    ##########################################################
    # Defining the remote function, right after ds in the remote
    @ray.remote
    def main_func(idx, ds_id):
        img, img_name = ds_id.__getitem__(idx)
        # Loading the segmentations of blobs and the breast mask
        blob_path = os.path.join(blob_dir,img_name+'.png')
        breast_path = os.path.join(breast_dir,img_name+'.png')
        blob_mask = ds_id._img_to_numpy(blob_path)
        breast_mask = ds_id._img_to_numpy(breast_path)
        pred_fpn_path = os.path.join(pred_fpn_dir,img_name+'.png')
        pred_fpn = ds_id._img_to_numpy(pred_fpn_path)

        pred_mask = pred_fpn >= thr

        # combining all masks
        pred_mask = hybrid_approach(pred_mask,breast_mask,blob_mask,
                hybrid_combining=hybrid_combining, 
                hybrid_combining_overlap=hybrid_combining_overlap)
        
        img_path = os.path.join(hybrid_approach_dir,img_name+'.png')
        img_dir = os.path.dirname(img_path)
        os.makedirs(img_dir, exist_ok=True)
        png16_saver(pred_mask, img_path)

        return  None

    data = list()
    for idx in tqdm(range(0,ds.__len__())):
        data.append(main_func.remote(idx, ds_id))
    data = ray.get(data)
    ray.shutdown()


if __name__ == '__main__':
    run()
