# For evaluating HDoGReg computing mean IoU per image

import ray
import os
from tqdm import tqdm
from math import log
import numpy as np
import mlflow
import pickle
from ruamel.yaml import YAML

from hdogreg.dataset import ImageDataset
from hdogreg.metrics import mIoU
from hdogreg.utils import arg_parsing


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
    out_csv_file = config_dict['out_csv_file']

    pred_dir = os.path.join(save_dir,'hybrid_approach/')


    logging_params([
    ("data_dir", os.path.basename(os.path.dirname(data_dir))),
    ('val_csv', csv_file), 
    ('out_csv_file',out_csv_file)
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

        return mIoU(pred_mask.astype(bool), mask.astype(bool))

    data = list()
    n_data = ds.__len__()
    for idx in tqdm(range(0,n_data)):
        data.append(main_func.remote(idx, ds_id))
    data = ray.get(data)
    out_file = os.path.join(save_dir, out_csv_file)
    pickle_dict = {
        'miou': data
    }

    pickle.dump( pickle_dict, open(out_file+"mIoU.p", "wb" ) )
    ray.shutdown()

    mean_miou = np.mean(data)
    
    logging_metrics({
        'mean_miou':mean_miou,
        })
    print(f'RESULT FOUND. mean mIoU = {mean_miou}')

if __name__ == '__main__':
    mlflow.set_experiment('Evaluate-mIoU')
    with mlflow.start_run():
        run()