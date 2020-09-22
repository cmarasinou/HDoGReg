import ray
import os
from tqdm import tqdm
from math import log
import numpy as np
import argparse
from ruamel.yaml import YAML

from hdogreg.dataset import ImageDataset, png16_saver
from hdogreg.image import delineate_breast


def arg_parsing(config_dict):
    arg_parser = argparse.ArgumentParser()
    for key in config_dict.keys():
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
    csv_file = config_dict['val_csv']
    save_dir = config_dict['save_dir']
    num_workers = int(config_dict['num_workers'])

    ##########################################################  
    # Output directory
    breast_dir = os.path.join(save_dir, 'breast_segm/')
    os.makedirs(breast_dir, exist_ok=True)
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
    # Defining the remote function, after ds in the remote
    @ray.remote
    def main_func(idx, ds_id):
        img,  img_name = ds_id.__getitem__(idx)
        breast_mask = delineate_breast(img)
        img_path = os.path.join(breast_dir,img_name+'.png')
        img_dir = os.path.dirname(img_path)
        os.makedirs(img_dir, exist_ok=True)
        png16_saver(breast_mask, img_path)

        return  None

    data = list()
    for idx in tqdm(range(0,ds.__len__())):
        data.append(main_func.remote(idx, ds_id))
    data = ray.get(data)
    ray.shutdown()


if __name__ == '__main__':
    run()
