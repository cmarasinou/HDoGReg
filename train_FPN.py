# For training the feature pyramidal network (FPN)

import os, ssl, sys
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch
import segmentation_models_pytorch as smp
import albumentations as albu
import yaml
import argparse
from hdogreg.utils import seed_everything
from skimage.morphology import remove_small_objects


def arg_parsing(config_dict):
    arg_parser = argparse.ArgumentParser()
    for key in config_dict.keys():
        arg_parser.add_argument('-{}'.format(key))
    args = vars(arg_parser.parse_args())
    for key in config_dict.keys():
        if args[key] is not None:
            config_dict[key] = args[key]
    return config_dict

class ExpProximityFunction():
    """ Creates exponential proximity function.
    """
    
    def __init__(self, radius=10, alpha=1):
        self.radius = radius
        self.alpha = alpha

    def __call__(self, imgs):
        """
        Args:
            imgs (list of np.arrays): second in the list should be the binary mask
        Returns:
            tranformed images (only second image changes)
        """
        if self.radius == 0:
            return imgs

        exp_map_func = make_exponential_mask_from_binary_mask
        imgs[1] = exp_map_func(imgs[1][:,:,0], 
            radius=self.radius, alpha=self.alpha)
        imgs[1] = np.expand_dims(imgs[1],2)

        return imgs


    
def make_exponential_mask(img, locations, radius, alpha, INbreast=False):
    """Creating exponential proximity function mask.
    Args:
        img (np.array, 2-dim): the image, only it's size is important
        locations (np.array, 2-dim): array should be (n_locs x 2) in size and 
            each row should correspond to a location [x,y]. Don't need to be integer,
            truncation is applied.
            NOTICE [x,y] where x is row number (distance from top) and y column number
            (distance from left)
        radius (int): radius of the exponential pattern
        alpha (float): decay rate
        INbreast (bool, optional): Not needed anymore, handled when parsing INbreast dataset
    Returns:
        mask (np.array, 0.0-1.0): Exponential proximity function
    """
    # create kernel which we will be adding at locations
    # Kernel has radial exponential decay form
    kernel = np.zeros((2*radius+1,2*radius+1))
    for i in range(0, kernel.shape[0]):
        for j in range(0, kernel.shape[1]):
            d = np.sqrt((i-radius)**2+(j-radius)**2)
            if d<= radius:
                kernel[i,j]=(np.exp(alpha*(1-d/radius))-1)/(np.exp(alpha)-1)
                
    # pad original img to avoid out of bounds errors
    img  = np.pad(img, radius+1, 'constant').astype(float)

    # update locations
    locations = np.array(locations)+radius+1
    locations = np.round(locations).astype(int)

    # initialize mask
    mask = np.zeros_like(img)    

    for location in locations:
        if INbreast:
            y, x = location
        else:
            x, y = location
        # add kernel
        mask[x-radius:x+radius+1, y-radius:y+radius+1] =np.maximum(mask[x-radius:x+radius+1, y-radius:y+radius+1],kernel)
        
    # unpad
    mask  = mask[radius+1:-radius-1,radius+1:-radius-1]
    
    return mask

def make_exponential_mask_from_binary_mask(mask, radius, alpha):
    """Creating exponential proximity function mask given the binary mask.
    Args:
        mask (np.array, 2-dim): binary mask
        radius (int): radius of the exponential pattern
        alpha (float): decay rate
    Returns:
        mask (np.array, 0.0-1.0): Exponential proximity function
    """
    locations = np.array(np.where(mask)).T
    mask = make_exponential_mask(mask, locations, radius, alpha, INbreast=False)
    
    return mask


class Dataset(BaseDataset):
    """Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
        
    
    """
    
    CLASSES = ['non-mc','mc']
    
    def __init__(
            self, 
            root_dir, 
            csv_file, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            np_transform=None
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.csv_file = csv_file
        self.img_format = 'png16'
        
        self.img_dir = os.path.join(root_dir, "images/"+self.img_format)
        self.csv_file = os.path.join(self.root_dir,"other/"+self.csv_file)
        self.df = pd.read_csv(self.csv_file, 
                                low_memory=False, 
                                index_col=0)
        
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.class_values = [255 if v==1 else v for v in self.class_values]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.np_transform = np_transform
    
    def __getitem__(self, idx):
        
        # read data
        image_path = os.path.join(self.img_dir,self.df.iloc[idx]['full_image']+'.png')
        mask_path = os.path.join(self.img_dir,self.df.iloc[idx]['mask_image']+'.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image,2)
        mask = cv2.imread(mask_path, 0)

        
        # extract classes from mask 
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')


        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        # apply other numpy tranformations
        if self.np_transform:
            imgs = [image, mask]
            imgs = self.np_transform(imgs)
            image, mask = imgs
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.df)


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


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
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


class MSELoss(smp.utils.base.Loss):

    def __init__(self, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = smp.utils.base.Activation(activation)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return torch.nn.MSELoss()(y_pr,y_gt)

def run():
    seed_everything()

    config_fname = os.path.abspath(os.path.dirname(os.path.realpath(__file__))+'/train_config.yaml')
    # Load settings from configuration file
    with open(config_fname) as config_f:
        config_dict = yaml.load(config_f, Loader=yaml.FullLoader)['default']
    # Update config settings using args
    config_dict = arg_parsing(config_dict)
            
    data_dir = config_dict['data_dir']
    batch_size = int(config_dict['batch_size'])
    train_csv = config_dict['train_csv']
    val_csv = config_dict['val_csv']
    model_name = config_dict['model_name']
    lr = float(config_dict['lr'])
    weight_decay = float(config_dict['wd'])
    n_epochs = int(config_dict['n_epochs'])
    radius = int(config_dict['radius'])
    alpha = float(config_dict['alpha'])
    wd = float(config_dict['wd'])
    comment = config_dict['comment']
    gpu_number = int(config_dict['gpu_number'])
    num_workers = int(config_dict['num_workers'])
    resume_training = int(config_dict['resume_training'])


    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)

    # Segmentation model settings
    ENCODER = 'inceptionv4'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['mc']
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda'

    # Create segmentation model with pretrained encoder
    model_starter_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))\
                +'/models/FPNinceptionBase1channel.pth'
    if resume_training:
        model_starter_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))\
                +'/models/'+model_name+'.pth'
        assert(os.path.exists(model_starter_path))
        model_name += '_new'
    if os.path.exists(model_starter_path):
        model = torch.load(model_starter_path)
    else:
        model = smp.FPN(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
            in_channels=1
        )

    # Preprocessing function switched to 1 channel
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessing_fn.keywords.update({'mean':[0.5], 'std':[0.5]})


    # Load datasets
    train_dataset = Dataset(
        data_dir, 
        train_csv, 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
        np_transform=ExpProximityFunction(radius=radius,alpha=alpha),
    )

    valid_dataset = Dataset(
        data_dir, 
        val_csv, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
        np_transform=ExpProximityFunction(radius=radius,alpha=alpha),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    loss = smp.utils.losses.DiceLoss()

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=lr, weight_decay=weight_decay),
    ])


    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    max_score = 0

    for i in range(0, n_epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            model_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))\
                +'/models/'+model_name+'.pth'
            torch.save(model, model_path)
            print('Model saved!')

            

if __name__ == '__main__':
  run()
            