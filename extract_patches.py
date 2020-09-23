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
    masks = list()
    full_images = list()
    for cdir, _, files in os.walk(ds_patch.img_dir):
        # remove extensions
        files = [f[:-4] for f in files]
        files.sort()
        files = [os.path.relpath(os.path.join(cdir,f),ds_patch.img_dir) for f in files]
        # keep only masks


        for f in files:
            if f.find('_mask')>=0:
                masks.append(f)
                f = f.replace('_mask_filled','')
                f = f.replace('_mask','')
                full_images.append(f)
            if f.find('FullMammoMasks')>=0:
                masks.append(f)
                f = f.replace('FullMammoMasks', 'FullMammoPng')
                full_images.append(f)
                
    df = pd.DataFrame({"full_image":full_images, mask_col_name:masks})
    ds = ImageDataset(data_dir, in_csv, img_format='png16', images_list=['full_image']
                 )
    # for each image in initial dataset find corresponding extracted patches 
    indices = []
    for idx in range(0,ds.__len__()):
        img_patches = df.full_image.str.contains(ds.df.full_image.iloc[idx])
        indices+=list(np.where(img_patches)[0])
    df_patch = df.iloc[indices]

    csv_dir = os.path.join(patches_dir,'other/')
    df_patch.rename(index=str, 
        columns={"full_image": "full_image", mask_col_name:"mask_image"}, 
        inplace=True)

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
    target_csv = config_dict['target_csv']
    train_csv = config_dict['train_csv']
    val_csv = config_dict['val_csv']
    num_workers = config_dict['num_workers']

    # process training images
    generate_patches_dataset(data_dir, patches_dir,  
        kernel_size, stride, mask_col_name, train_csv, 'train.csv', num_workers)
    # process validation images
    generate_patches_dataset(data_dir, patches_dir,  
        kernel_size, stride, mask_col_name, val_csv, 'val.csv', num_workers)



if __name__ == '__main__':
    run()
