from ruamel.yaml import YAML
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
from hdogreg.dataset import *
from hdogreg.utils import arg_parsing


def generate_patches_dataset(data_dir, patches_dir,
                            kernel_size, stride, 
                            mask_col_name, 
                            in_csv, out_csv,
                            num_workers):

    print("\n------Generating patches dataset------\n")

    # Dataset to extract patches from
    ds = ImageDataset(data_dir, in_csv, img_format='png16', 
                  images_list=['full_image',mask_col_name],
                  attributes_list=['full_image',mask_col_name]
                  )

    # Dataset that extracts patches
    ds_patch = MakePatchesGeneral(ds, patches_dir, 
        img_name_func='default_naming', getboundingboxes_func='get_bbox_positive',
        bb_kargs={'kernel_size':kernel_size, 'stride':stride}
        )
    # Dataloader for parallel extraction
    dl_patch = DataLoader(ds_patch, batch_size=1, shuffle=False, num_workers=num_workers)
    
    # Producing patches
    for data in tqdm(dl_patch):
        continue

    # Separating the files in masks and regular images
    all_patch_files = list()
    for cdir, _, files in os.walk(ds_patch.img_dir):
        # remove extensions
        files = [f[:-4] for f in files]
        files.sort()
        files = [os.path.relpath(os.path.join(cdir,f),ds_patch.img_dir) for f in files]
        # keep only masks
        for f in files:
            all_patch_files.append(f)
    all_patch_files = pd.Series(all_patch_files)

    ds = ImageDataset(data_dir, in_csv, img_format='png16', images_list=['full_image']
                 )

    collect_patch_data = list()
    # for each image in initial dataset find corresponding extracted patches 
    for idx in range(0,ds.__len__()):
        record = ds.df.iloc[idx]
        # Adding "_" such that we make sure cases like: "image1", "image10" are not
        # picked together
        images = all_patch_files[all_patch_files.str.contains(record.full_image+'_')]
        # Sorting needed to make image patches match with mask patches
        images.sort_values(inplace=True)
        masks = all_patch_files[all_patch_files.str.contains(record[mask_col_name]+'_')]
        masks.sort_values(inplace=True,)
        df = pd.DataFrame({"full_image":list(images),
                    "mask_image":list(masks)})
        collect_patch_data.append(df)
    df_patch = pd.concat(collect_patch_data, ignore_index=True)
    csv_dir = os.path.join(patches_dir,'other/')
    df_patch.to_csv(os.path.join(csv_dir,out_csv), index=True)


def run():
    # Load configuration contents
    config_fname = os.path.abspath(os.path.dirname(os.path.realpath(__file__))\
        +'/preprocess_config.yaml')
    with open(config_fname) as config_f:
        config_dict = YAML().load(config_f)['default']

    # Update config settings using args
    config_dict = arg_parsing(config_dict)
        
    data_dir = config_dict['data_dir']
    patches_dir = config_dict['patches_dir']
    kernel_size = int(config_dict['kernel_size'])
    stride = int(config_dict['stride'])
    mask_col_name = config_dict['mask_col']
    train_csv = config_dict['train_csv']
    val_csv = config_dict['val_csv']
    num_workers = config_dict['num_workers']

    patches_dir = os.path.abspath(patches_dir)
    # process training images
    generate_patches_dataset(data_dir, patches_dir,  
        kernel_size, stride, mask_col_name, train_csv, 'train.csv', num_workers)
    # process validation images
    generate_patches_dataset(data_dir, patches_dir,  
        kernel_size, stride, mask_col_name, val_csv, 'val.csv', num_workers)


if __name__ == '__main__':
    run()
