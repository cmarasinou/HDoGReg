# For segmenting blobs in mammograms

import ray
import os
from tqdm import tqdm
from ruamel.yaml import YAML

from hdogreg.dataset import ImageDataset, png16_saver
from hdogreg.image import blob_detector_hdog
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


    DoG_thr = float(config_dict['DoG_thr'])
    min_sigma = float(config_dict['min_sigma'])
    max_sigma = float(config_dict['max_sigma'])
    overlap = float(config_dict['overlap'])
    hessian_thr = float(config_dict['hessian_thr'])


    blob_dir = os.path.join(save_dir, 'blob/')
    os.makedirs(blob_dir, exist_ok=True)
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
        blob_mask = blob_detector_hdog(img, min_sigma=min_sigma,max_sigma=max_sigma,
            hessian_thr=hessian_thr, threshold=DoG_thr,overlap=overlap)
        img_path = os.path.join(blob_dir,img_name+'.png')
        img_dir = os.path.dirname(img_path)
        os.makedirs(img_dir, exist_ok=True)
        png16_saver(blob_mask, img_path)

        return  None

    data = list()
    for idx in tqdm(range(0,ds.__len__())):
        data.append(main_func.remote(idx, ds_id))
    data = ray.get(data)
    ray.shutdown()


if __name__ == '__main__':
    run()
