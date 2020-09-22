import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms as trf
import pandas as pd
from PIL import Image
import numpy as np
import pydicom
from scipy import ndimage
from skimage.measure import regionprops
import random
from hdogreg.image import get_bounding_boxes

def png16_saver(img, path):
    """Saves image previously normalized to 1 as png at 16 bits
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    bits = 16
    norm = 2**bits-1
    img = (img*norm).astype(np.uint32)
    img = Image.fromarray(img)
    if path[-4:]!='.png':
        path=path+'.png'
    img.save(path)

def bboxInImage(bb, img_size):
    
    h,w = img_size
    x1, x2, y1, y2 = bb
    
    if x1<0 or x2>h-1 or y1<0 or y2>w-1:
        return False
    
    return True

class ImageDataset(Dataset):
    """Pytorch Dataset that can return images and tabular data. Images 
    can be returned as numpy arrays or torch tensors."""

    def __init__(self, root_dir,
                csv_file,
                img_format='png8',
                sample_size = None,
                images_list = ['full_image', 'mask_image'], 
                attributes_list = [],
                torch_tensor = False,
                transforms = None,
                np_transforms = None,
                tensor_norm = None,
                alb_transforms = None,
                crop = None
                ):
        """
        Args:
            root_dir (str): data directory path
            csv_file (str): csv file name in root_dir/others/
            img_format (str, optional): has to correspond to one of the 
                directories in /root_dir/images/. Regular options: 'png8', 
                'png16', 'dcm', 'npy'
            sample_size (float, 0.0-1.0, optional): sample fraction, sample
                is being drawn randomly
            images_list (list of str, optional): items correspond to the 
                columns in csv_file associated with file_names
            attributes_list (list of str, optional): items correspond to the
                columns in csv_file. For returning only values in csv_file.
            torch_tensor (bool, optional): If True images are given as torch.
                tensor, otherwise np.array
            transforms (torchvision.transforms, optional): Applied on images
                when in PIL format
            np_transforms (object, optional): Takes lists of np.array and
                transforms them
            tensor_norm (float, optional): If the output is a tensor, image 1 is normalized.
                if tensor_norm>0 then out = out * tensor_norm
                if tensor_norm == 0 then out = standardized(out)
                if tensor_norm <0 then out = abs(2*tensor_norm)*tensor_norm+tensor_norm, symmetric about 0
            alb_transforms (albumentations package transforms): They transform both iamge and mask
        """
        self.root_dir = os.path.abspath(root_dir)
        self.csv_file = csv_file
        self.img_format = img_format
        self.combining_masks = None
        self.images_list = images_list
        self.attributes_list = attributes_list
        self.torch_tensor = torch_tensor
        self.transforms = transforms
        self.np_transforms = np_transforms
        self.alb_transforms = alb_transforms
        self.tensor_norm = tensor_norm
        if self.tensor_norm == 1: self.tensor_norm=None
        self.crop = crop


        # Find images relative directory and extension
        img_extension_dict = {'png8': 'png', 
                            'png16':'png',
                            'dcm':'dcm',
                            'npy':'npy',
                            'tif':'tif'} 

        # Checking img_format
        # Finding suitabble img_reader function
        # Producing img_directory
        if img_format in img_extension_dict.keys():
            self.img_extension = img_extension_dict[img_format]
            img_reldir = "images/"+img_format
            if img_format=='dcm':
                self.img_reader = self._dicom_to_numpy
            elif img_format=='npy':
                self.img_reader = self._npy_to_numpy
            else:
                self.img_reader = self._img_to_numpy
        else:
            raise(NameError("img_format must be one of the following: %s" %(str(img_extension_dict.keys()))))
        self.img_dir = os.path.join(root_dir, img_reldir) 


        # Producing csv_path
        self.csv_file = os.path.join(self.root_dir,"other/"+self.csv_file)


        # Loading csv (with some filters)
        self.df = pd.read_csv(self.csv_file, 
                                low_memory=False, 
                                index_col=0)
        # Some manual filters
        if self.root_dir =='/home/cmarasinou/Documents/Data/CBIS-DDSM':
            print("Original DDSM dataset")
            self.df = self.df[self.df['image_size_mismatch']==0]
            self.df = self.df[self.df['bad_image']==0]

        # Get sample
        if sample_size is not None:
            self.df = self.df.sample(frac=sample_size, 
                random_state=123)

    def _dicom_to_numpy(self, img_path):
        """Reads DICOM image path and returns np.array normalized to 1
        """
        img = pydicom.dcmread(img_path)
        bits = img.BitsAllocated
        img = img.pixel_array

        img = img.astype(float)
        norm = 2**bits-1
        img /= norm

        if img.max()<=(1/2**4):
            img = img*2**4

        return img

    def _img_to_numpy(self, img_path):
        """Reads general image path and returns np.array normalized to 1
        """
        mode_to_bits = {'1':1, 'L':8, 'P':8, 'RGB':8, 'RGBA':32, 'CMYK':32, 'YCbCr':24, 'I':16, 'F':32}
        img = Image.open(open(img_path, 'rb'))
        bits = mode_to_bits[img.mode]

        img = np.array(img, dtype=float)
        norm = 2**bits-1
        img /= norm
        return img


    def _img_to_pil(self, img_path):
        """Reads general image path and returns np.array normalized to 1
        """
        img = Image.open(open(img_path, 'rb'))
        return img

    def  _pil_to_numpy(self,img):
        mode_to_bits = {'1':1, 'L':8, 'P':8, 'RGB':8, 'RGBA':32, 'CMYK':32, 'YCbCr':24, 'I':16, 'F':32}
        bits = mode_to_bits[img.mode]
        img = np.array(img, dtype=float)
        norm = 2**bits-1
        img /= norm
        return img

    def _npy_to_numpy(self,img_path):
        """Reads npy file path and returns the np.array
        """
        return np.load(img_path)

    def give_combined_masks(self):
        """For combining ground truth masks belonging to the same image
        """
        self.full_df = self.df
        # Retain only one entry for each unique greyscale image 
        # (masks usually multiple)
        self.df = self.df.drop_duplicates('full_image')
        self.combining_masks = True

    def _combine_binary_images(self,imgs):
        """Reads list of numpy binary imgs and adds them outputing a binary 
        mask
        """
        mask = imgs[0]
        if len(imgs)>1:
            for img in imgs[1:]:
                mask = np.logical_or(mask,img)
        return mask.astype(float)

    def change_dataframe(self,df):
        '''Changes the dataframe from which the dataset pulls data
        Args: 
            df (pd.DataFrame): the new Dataframe
        '''
        self.df = df
    

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Find image filenames and load images
        imgs = []
        if self.crop:
            self.img_reader = self._img_to_pil
        for img_type in self.images_list:
            if self.combining_masks and img_type=='mask_image':
                # many masks should be combined
                full_image = self.df.iloc[idx]['full_image']
                mask_img_names = [msk for msk in self.full_df[self.full_df.full_image==full_image]['mask_image']]
                mask_img_paths = [os.path.join(self.img_dir,
                                msk+"."+self.img_extension) for msk in mask_img_names]
                mask_imgs = [self.img_reader(msk) for msk in mask_img_paths]
                imgs.append(self._combine_binary_images(mask_imgs))
            else:
                img_path = os.path.join(self.img_dir,
                    str(self.df.iloc[idx][img_type])+"."+self.img_extension)
                imgs.append(self.img_reader(img_path))

        if self.crop:
            h,w = imgs[0].height, imgs[0].width
            assert(h >= self.crop)
            assert(w >= self.crop)
            while True:
                t = random.randint(0, h-self.crop)
                l = random.randint(0, w-self.crop)
                imgs_crops = [img.crop((l,t,l+self.crop,t+self.crop)) for img in imgs]
                imgs_crops = [self._pil_to_numpy(img) for img in imgs_crops]
                if imgs_crops[1].sum()>0:
                    imgs = imgs_crops
                    break
        # Find attributes
        attributes = []
        for attr_type in self.attributes_list:
            attributes.append(self.df.iloc[idx][attr_type])

        # Apply Albumentations to images
        if self.alb_transforms:
            imgs = self.alb_transforms(image=imgs[0], mask=imgs[1])
            imgs = [imgs['image'], imgs['mask']]
        # Apply np_transforms to images
        if self.np_transforms is not None:
            if isinstance(self.np_transforms,list):
                for np_transform in self.np_transforms:
                    imgs = np_transform(imgs)
            else:
                imgs = self.np_transforms(imgs)


        # Convert to tensors
        if self.torch_tensor:
            imgs = [torch.tensor(img).unsqueeze(0).to(torch.float) for img in imgs]
            if self.transforms is not None:
                imgs = [trf.ToPILImage()(img) for img in imgs]
                imgs = [self.transforms(img) for img in imgs]
                imgs = [trf.ToTensor()(img) for img in imgs]
            if self.tensor_norm is not None:
                if self.tensor_norm>0:
                    imgs[0] = self.tensor_norm*imgs[0] # We only want the first image normalized
                elif self.tensor_norm==0:
                    imgs[0] = trf.Normalize((0.47,),(0.0033,))(imgs[0])
                else:
                    imgs[0] = np.abs(2*self.tensor_norm)*imgs[0]+self.tensor_norm

            # Attributes shouldn't be a tensor for now ##attributes = [torch.tensor(attr) for attr in attributes]
            # Extra image tranforms
        
        return tuple(imgs+attributes)


class MakePatchesGeneral(Dataset):
    """Creates and saves a new dataset consisting of patches drawn from an
    existing dataset
    """
    def __init__(self, dataset, new_data_dir, img_rel_dir='images/png16/', 
                 img_saver='png16_saver',
                 getboundingboxes_func='get_roi_bboxes', bb_kargs={},
                img_name_func='idx_transformation', 
                save_patches_func='save_patches_regular', save_kargs={}):
        """
        Args:
            dataset (torch.Dataset): The initial dataset
            new_data_dir (str): Directory path for the new dataset. Doesn't
                have to exist.
        """
        self.ds = dataset
        self.new_data_dir = new_data_dir
        self.img_dir = os.path.join(self.new_data_dir, img_rel_dir)
        self.__imgsaver__ = globals()[img_saver]
        self.__getboundingboxes__ = globals()[getboundingboxes_func]
        self.bb_kargs=bb_kargs
        self.__imgnamefunc__ = globals()[img_name_func]
        self.__savepatchesfunc__ = globals()[save_patches_func]
        self.save_kargs = save_kargs
        
        
        # make dataset directories
        self.__makedatasetdirectories__()

    def __len__(self):
        return self.ds.__len__()

    def __getitem__(self, idx):
        
        data= self.ds.__getitem__(idx)
        n_images = int(len(data)/2)
        imgs, img_names = (data[0:n_images],data[n_images:2*n_images])
        bbox_list = self.__getboundingboxes__(imgs, **self.bb_kargs)
        img_names = self.__imgnamefunc__(img_names, idx)
        self.__savepatchesfunc__(imgs, img_names, bbox_list, self.__imgsaver__, self.img_dir, **self.save_kargs)
        
        #Just returning something to be able to use dataloaders
        return idx 
            
    def __makedatasetdirectories__(self):
        """Creating the new dataset directories for the images and the csv 
        files
        """
        try:
            os.makedirs(self.new_data_dir, exist_ok=True)
            os.makedirs(os.path.join(self.new_data_dir,"other/"), exist_ok=True)
            os.makedirs(os.path.join(self.new_data_dir,self.img_dir), exist_ok=True)
        except OSError:
            print("Directories not created")


def LocationInBbox(loc, bb):
    x,y = loc
    x1, x2, y1, y2 = bb
    if (x>=x1)&(x<=x2)&(y>=y1)&(y<=y2):
        return True
    return False

def PatchesFromBboxes(img, bboxes):
    patches = []
    for bb in bboxes:
        patches.append(img[bb[0]:bb[1],bb[2]:bb[3]])
    return patches
    
def npy_saver(img, path):
    """Saves image previously normalized to 1 as npy 
    """
    np.save(path+'.npy',img)

def get_roi_bboxes(imgs, size=512):
    """Extracts ROI bboxes based on objects appearing in the mask
    """
    mask = imgs[1]
    label, n_objects = ndimage.label(mask)
    regions = regionprops(label)
    bbox_list = []
    for region in regions:
        centroid = region.centroid
        x, y = int(centroid[0]), int(centroid[1])
        bbox = [x-size//2,x+size//2+1,y-size//2,y+size//2+1]
        if (bboxInImage(bbox,mask.shape)) and (region.bbox_area<= size**2):
            bbox_list.append(bbox)

    return bbox_list

def get_bbox_positive(imgs, kernel_size=512, stride=480):
    mask = imgs[1]
    bbList = get_bounding_boxes(mask,kernel_size, stride)
    bbListFiltered = []
    for bb in bbList:
        mask_patch = mask[bb[0]:bb[1],bb[2]:bb[3]]
        if (mask_patch==1.).sum() > 0:  
            bbListFiltered.append(bb)
    return bbListFiltered


def get_bbox_positive_and_negative(imgs, kernel_size=512, stride=480, ratio=1):
    img, mask = imgs
    bbList = get_bounding_boxes(mask,kernel_size, stride)
    bbListFiltered = []
    bbListNegative = []
    for bb in bbList:
        mask_patch = mask[bb[0]:bb[1],bb[2]:bb[3]]
        if (mask_patch==1.).sum() > 0:  
            bbListFiltered.append(bb)
        else:
            img_patch = img[bb[0]:bb[1],bb[2]:bb[3]]
            if (img_patch.std()>=1e-10):
                bbListNegative.append(bb)

    n_pos = len(bbListFiltered)
    n_neg = len(bbListNegative)
    neg_idx = np.random.choice(n_neg, size=min(n_neg,int(ratio*n_pos)), replace=False)
    bbListNegative = [bbListNegative[idx] for idx in neg_idx]
    bbListFiltered.extend(bbListNegative)
    return bbListFiltered

def get_roi_bboxes_centered(imgs, size=512):
    """Extracts ROI bboxes based on objects appearing in the mask
    If object is larger than bbox, bbox still returned
    """
    mask = imgs[1]
    label, n_objects = ndimage.label(mask)
    regions = regionprops(label)
    bbox_list = []
    for region in regions:
        centroid = region.centroid
        x, y = int(centroid[0]), int(centroid[1])
        bbox = [x-size//2,x+size//2+1,y-size//2,y+size//2+1]
        if (bboxInImage(bbox,mask.shape)):
            bbox_list.append(bbox)

    return bbox_list


def get_context_bboxes(imgs, k=20):
        
        size = 95
        mini_size=9
        n_pos_max = 150
        img, mask = imgs[0],imgs[1]
        # produce positive bboxes
        label, n_objects = ndimage.label(mask)
        regions = regionprops(label)
        bbox_list = []
        for region in regions:
            centroid = region.centroid
            x, y = int(centroid[0]), int(centroid[1])
            bbox = [x-size//2,x+size//2+1,y-size//2,y+size//2+1]
            # check if within image
            if bboxInImage(bbox,mask.shape):
                # check overlap of mask in mini box
                mini_bbox = [x-mini_size//2,x+mini_size//2+1,
                             y-mini_size//2,y+mini_size//2+1]
                if maskOverlap(mini_bbox, mask):
                    bbox_list.append(bbox)
        # keep only n_pos_max
        if len(bbox_list)>n_pos_max:
            bbox_list = random.sample(bbox_list,n_pos_max)
        # produce negative bboxes
        n_pos = len(bbox_list)
        n_neg = k*n_pos
        for i in range(0,n_neg):
            bbox_list.append(random_negative_bbox(img, mask, size))

        return bbox_list

def random_negative_bbox(img,mask,size):
    h,w = img.shape
    
    x,y = np.random.randint(size//2,h-size//2), np.random.randint(size//2,w+size//2)
    bb = [x-size//2,x+size//2+1,y-size//2,y+size//2+1]
    img_patch = img[bb[0]:bb[1],bb[2]:bb[3]]
    # conditions to keep it:
    # within image boundaries
    # does not contain positive regions
    # part of breast, i.e. patch not uniform
    if bboxInImage(bb,img.shape) and (not maskOverlap(bb,mask)) and (img_patch.std()>=1e-10):
        return bb
    else:
        return random_negative_bbox(img, mask,size)

def idx_transformation(img_names,idx):
    img_names=("full_image_{}".format(idx),
                "mask_image_{}".format(idx),
                
               "pred_image_{}".format(idx)
              )
    return img_names

def default_naming(img_names,idx):
    return img_names

def separate_dir_naming(img_names,idx):
    img_name = img_names[0]
    return ['imgs/'+img_name, 'masks/'+img_name]

def image_and_mask(img_names,idx):
    return (img_names[0], img_names[0]+'_mask')

def save_patches_regular(imgs, img_names, bbox_list, img_saver, img_dir):
    for img, img_name in zip(imgs, img_names):
        for i, bb in enumerate(bbox_list):
            patch = img[bb[0]:bb[1],bb[2]:bb[3]]
            fname = '{}_{}'.format(img_name,i)
            fpath = os.path.join(img_dir,fname)
            img_saver(patch, fpath)


def save_patches_binary(imgs, img_names, bbox_list, img_saver, img_dir, k=20, augmentations=False):
    # make the classification dirs
    os.makedirs(os.path.join(img_dir,'0/'), exist_ok=True)
    os.makedirs(os.path.join(img_dir,'1/'), exist_ok=True)
    # save all patches
    # not the masks
    imgs= imgs[0:1]
    img_names = img_names[0:1]
    n_pos = len(bbox_list)//(k+1)
    for img, img_name in zip(imgs, img_names):
        for i, bb in enumerate(bbox_list):
            patch = img[bb[0]:bb[1],bb[2]:bb[3]]
            if i <n_pos:
                fname = '1/{}_{}'.format(img_name,i)
            else:
                fname = '0/{}_{}'.format(img_name,i)
            fpath = os.path.join(img_dir,fname)
            img_saver(patch, fpath)
            if augmentations:
                patches, fpaths = make_augmentations(patch, fpath)
                for patch, fpath in zip(patches,fpaths):
                    img_saver(patch,fpath)

def make_augmentations(patch, fpath):
    hor_flip = np.flip(patch,1)  
    ver_flip = np.flip(patch,0)
    rot_90 = np.rot90(patch, 1)
    rot_180 = np.rot90(patch, 2)
    rot_270 = np.rot90(patch, 3)
    patches = [hor_flip, ver_flip, rot_90, rot_180, rot_270]
    suffix = ['hor_flip', 'ver_flip', 'rot_90', 'rot_180', 'rot_270']
    fpaths = [fpath+s for s in suffix]
    return patches, fpaths


def maskOverlap(bb, mask):
    
    if mask[bb[0]:bb[1],bb[2]:bb[3]].sum()==0:
        return False
    return True