# Script that takes model and dataset, infers on dataset, saves model outputs as images 
#
# Requirements:
# model_name, data_dir, val_csv, kernel_size, n_initial_filters
#
#

import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from ruamel.yaml import YAML
import torch
import argparse
import pandas as pd
import albumentations as albu
from torch.utils.data import Dataset as BaseDataset
import cv2
import segmentation_models_pytorch as smp
import skimage
from hdogreg.utils import arg_parsing



def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_bounding_boxes(img, kernel_size, stride):
    """Gives all bounding boxes covering an image with a sliding window
    Args:
        img (np.array or torch.tensor)
        kernel_size (int): size of the kernel, kernel is a square
        stride (int): stride of the sliding window motion
    Returns:
        bounding boxes list (list of (x, xp,y, yp)), convention of bounding box
        (top, bottom, left, right)
    Notice: If not all image covered by sliding window, additional
    bounding boxes are created ending at the end of each dimension
    """
    h, w = img.shape
    h_p, w_p = kernel_size, kernel_size
    s = stride

    # All x locations
    x = [i*s for i in range(0, int((h-h_p)/s)+1)]
    # Add one more point if didn't cover all range
    if x[-1]+h_p!=h:
        x.append(h-h_p)
    # All y locations
    y = [j*s for j in range(0, int((w-w_p)/s)+1)]
    # Add one more point if didn't cover all range
    if y[-1]+w_p!=w:
        y.append(w-w_p)

    # All bounding boxes in the form (x,xp,y,yp)
    # x,y: the top left corner/ xp,yp: the bottom right corner
    bbList = [(xi,xi+h_p, yi, yi+w_p) for xi in x for yi in y]

    return bbList

class Dataset(BaseDataset):
    """Dataset. Reads full images, apply preprocessing transformations.
    
    Args:
        data_dir (str): dataset directory
        csv_file (str): csv file within dataset listing full images and mask images
        classes (list): classes list
        preprocessing (albumentations.Compose): data preprocessing 
    """
    
    def __init__(
            self, 
            root_dir, 
            csv_file, 
            preprocessing=None,
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.csv_file = csv_file
        self.img_format = 'png16'
        
        self.img_dir = os.path.join(root_dir, "images/"+self.img_format)
        self.csv_file = os.path.join(self.root_dir,"other/"+self.csv_file)
        self.df = pd.read_csv(self.csv_file, 
                                low_memory=False, 
                                index_col=0)
        self.preprocessing = preprocessing
    
    def __getitem__(self, idx):
        
        # read data
        image_path = os.path.join(self.img_dir,self.df.iloc[idx]['full_image']+'.png')
        image = cv2.imread(image_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image,2)
        if self.preprocessing:
            image = globals()[self.preprocessing](image)
        ENCODER = 'inceptionv4'
        ENCODER_WEIGHTS = 'imagenet'
        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        preprocessing_fn.keywords.update({'mean':[0.5], 'std':[0.5]})
        image = get_preprocessing(preprocessing_fn)(image=image)['image']

        return image, self.df.iloc[idx]['full_image']
        
    def __len__(self):
        return len(self.df)


def predict_from_large_image(model, img, kernel_size=512, overlap = 0, batch_size=1):
    """Gives output of the network for a large image. Sliding window method is used
    to predict on individual patches
    Args:
        img (torch.tensor): Input image
        mask (torch.tensor, optional): Binary mask. Prediction is generated only for
            mask region
        kernel_size (int, optional): Sliding window patch size
        overlap (int, optional): Overlap between patches when moving in the sliding window
    Returns:
        network output for the whole image
    """
    #img = img.squeeze()
    out = torch.zeros(img.shape[1],img.shape[2]).to(DEVICE)
    o = overlap
    bs = batch_size
    bb_list = get_bounding_boxes(img[0], kernel_size,kernel_size-2*o)
    bb_batches = [bb_list[i:i + bs] for i in range(0, len(bb_list), bs)]
    for bb_batch in bb_batches:
        patch_batch = [img[:,bb[0]:bb[1],bb[2]:bb[3]] for bb in bb_batch]
        patch_batch_tensor = torch.stack(patch_batch,dim=0)
        batch_out = model.predict(patch_batch_tensor)
        for i in range(0,len(bb_batch)):
            patch_out = batch_out[i].squeeze()
            hp,wp = patch_out.size(0), patch_out.size(1)
            bb = bb_batch[i]
            if overlap == 0:
                out[bb[0]:bb[1],bb[2]:bb[3]]=patch_out
            else:
                # If there is overlap
                if (bb[0]==0) or (bb[2]==0) or (bb[1]==out.size(0)) or (bb[3]==out.size(1)):
                    # If touching the edge the edge doesn't have to be accounted in the overlap
                    dx1, dx2, dy1, dy2 = o, -o, o, -o
                    if bb[0]==0: dx1 = 0
                    if bb[1]==out.size(0): dx2 = 0
                    if bb[2]==0: dy1=0
                    if bb[3]==out.size(1): dy2=0
                    out[bb[0]+dx1:bb[1]+dx2,bb[2]+dy1:bb[3]+dy2]=patch_out[dx1:hp+dx2,dy1:wp+dy2]
                else:
                    out[bb[0]+o:bb[1]-o,bb[2]+o:bb[3]-o]=patch_out[o:-o,o:-o] 
    return out.cpu()


# Load configuration contents
config_fname = os.path.abspath(os.path.dirname(os.path.realpath(__file__))+'/validation_config.yaml')
with open(config_fname) as config_f:
    config_dict = YAML().load(config_f)['confusion_matrix']
#Update config settings using args
config_dict = arg_parsing(config_dict)

data_dir = config_dict['data_dir']
val_csv = config_dict['val_csv']
model_name = config_dict['model_name']
kernel_size = int(config_dict['kernel_size'])
n_initial_filters = int(config_dict['n_initial_filters'])
input_norm = float(config_dict['input_norm'])
img_format = config_dict['img_format']
overwrite_images = int(config_dict['overwrite_images'])
net_architecture = config_dict['net_architecture']
erosion=int(config_dict['erosion'])
gpu_number = int(config_dict['gpu_number'])
preprocessing = config_dict['preprocessing']
batch_size = int(config_dict['batch_size'])
save_dir = config_dict['save_dir']

print(data_dir)

out_dir = os.path.join(save_dir, 'predsFPN/')
os.makedirs(out_dir, exist_ok=True)
other_dir = os.path.join(data_dir, 'other/')

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)
DEVICE = 'cuda'
ENCODER = 'inceptionv4'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
preprocessing_fn.keywords.update({'mean':[0.5], 'std':[0.5]})

##########################################################
# Load dataset
ds = Dataset(data_dir, val_csv)
# Load model
model_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__))+'/models/'+model_name+'.pth')
print(model_path)
model = torch.load(model_path, map_location=torch.device('cpu') ).to(DEVICE)


##########################################################
# Infering and Saving predictions
print("\n------Making Predictions------\n")
for idx in tqdm(range(0,ds.__len__())):
    img, img_name = ds.__getitem__(idx)
    # Check if name exists
    fpath = os.path.join(out_dir,img_name+'.png')
    if not os.path.exists(fpath) or overwrite_images:
        h,w = img.shape[1], img.shape[2]
        x_tensor = torch.from_numpy(img).to(DEVICE)
        pred = predict_from_large_image(model,x_tensor,
                                        kernel_size=kernel_size,
                                        overlap =kernel_size//4,
                                        batch_size=batch_size)
        pred = pred.cpu().squeeze().numpy()
        pred = (pred*(2**16-1)).astype(np.uint32)
        pil_img = Image.fromarray(pred)
        img_dir = os.path.dirname(fpath)
        os.makedirs(img_dir, exist_ok=True)
        pil_img.save(fpath)


