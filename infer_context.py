import os 
from tqdm import tqdm
from ruamel.yaml import YAML
import argparse
from skimage.feature import blob_dog
from skimage.draw import circle
from hdogreg.dataset import ImageDataset, bboxInImage, png16_saver, npy_saver
from hdogreg.baseline.contextnet import contextNet
from hdogreg.baseline.transforms import ContextPaperPreprocess
import torch
import numpy as np

def arg_parsing(config_dict):
    arg_parser = argparse.ArgumentParser()
    for key in config_dict.keys():
        arg_parser.add_argument('-{}'.format(key))
    args = vars(arg_parser.parse_args())
    for key in config_dict.keys():
        if args[key] is not None:
            config_dict[key] = args[key]
    return config_dict


def apply_dog(img, min_sigma, max_sigma, DoG_thr, overlap):
    if isinstance(img, torch.Tensor):
        img = img.squeeze().numpy()
    blobs = blob_dog(img, min_sigma=min_sigma,
     max_sigma=max_sigma, threshold=DoG_thr, overlap=0)
    return blobs

def blobs_to_bbox(blobs,bbox_size=95):
    bbox_list =[]
    for blob in blobs:
        size = bbox_size
        x, y, r = blob
        x,y = int(x), int(y)
        bbox = [x-size//2,x+size//2+1,y-size//2,y+size//2+1]
        bbox_list.append(bbox)
    return bbox_list

def extract_patches(img,bboxes):
    patches=[]
    for bb in bboxes:
        patch = extract_patch(img,bb)
        patch = torch.Tensor(patch).unsqueeze(0)
        patches.append(patch)
    return torch.stack(patches, dim=0)

def extract_patch(img,bb):
    if isinstance(img, torch.Tensor):
        img = img.squeeze().numpy()
    h_bb, w_bb = bb[1]-bb[0], bb[3]-bb[2]
    assert(h_bb>0 and w_bb>0)
    if bboxInImage(bb, img.shape):
        return img[bb[0]:bb[1],bb[2]:bb[3]]
    patch = np.zeros((h_bb,w_bb))
    h,w = img.shape
    x1,x2,y1,y2 = bb
    dx1,dy1, dx2,dy2= 0,0,0,0
    if x1<0: dx1=-x1
    if x2>h: dx2=x2-h
    if y1<0: dy1=-y1
    if y2>w: dy2=y2-w
    patch[dx1:h_bb-dx2,dy1:w_bb-dy2]=img[x1+dx1:x2+dx2,y1+dy1:y2+dy2]
    return patch

def predict_batch(model, inputs, device='cuda'):
    model.to(device)
    inputs=inputs.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        y_prob = torch.softmax(logits, 1)[:,1]
    return y_prob

def load_model(model,  file_path):
    """Loads model weights
    """
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state'])

def run():
    # Load configuration contents
    config_fname = os.path.abspath(os.path.dirname(os.path.realpath(__file__))\
        +'/contextNet_config.yaml')
    with open(config_fname) as config_f:
        config_dict = YAML().load(config_f)['default']

    # Update config settings using args
    config_dict = arg_parsing(config_dict)
    
    data_dir = config_dict['data_dir']
    save_dir = config_dict['save_dir']
    csv = config_dict['csv']
    DoG_thr = float(config_dict['DoG_thr'])
    min_sigma = float(config_dict['min_sigma'])
    max_sigma = float(config_dict['max_sigma'])
    overlap = float(config_dict['overlap'])
    img_format = config_dict['img_format']
    model_name = config_dict['model_name']
    DoG_only = int(config_dict['DoG_only'])
    batch_size = int(config_dict['batch_size'])
    gpu_number = int(config_dict['gpu_number'])

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)

    ##########################################################
    # Load dataset
    print("\n------Loading Dataset------\n")
    ds = ImageDataset(data_dir, csv, img_format=img_format,
              torch_tensor=True,
              images_list=['full_image','full_image'],
              attributes_list=['full_image'],
              np_transforms=ContextPaperPreprocess())
    ##########################################################
    # Load model
    model_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__))\
        + '/models/'+model_name+'.tar')
    model = contextNet(1, 32, dropout=0.5)
    load_model(model, model_path)

    ##########################################################
    # Infering and Saving predictions
    print("\n------Making Predictions------\n")

    for idx in tqdm(range(0,ds.__len__())):
        img_name = ds.df.iloc[idx]['full_image']
        # Check if name exists
        img_pre, img, img_name = ds.__getitem__(idx)
        blobs = apply_dog(img, min_sigma,max_sigma,DoG_thr,overlap)
        n_blobs = blobs.shape[0]
        if DoG_only:
            probs = np.ones((n_blobs,1))
        else:
            bboxes = blobs_to_bbox(blobs)
            patches = extract_patches(img_pre,bboxes)
            batches = patches.size(0)//batch_size+1
            patches_batches = torch.chunk(patches,batches,dim=0)
            prob_list = []

            for inputs in patches_batches:
                out = predict_batch(model,inputs)
                out = out.cpu()
                prob_list.append(out)

            probs = torch.cat(prob_list, dim=0).unsqueeze(1).numpy()

        # Create pred image
        pred_image = np.zeros_like(img.squeeze())
        for blob, prob in zip(blobs, probs):
            x,y,r = blob
            bb = [x-r,x+r+1,y-r,y+r+1]
            if bboxInImage(bb,pred_image.shape):
                rr, cc = circle(x,y,r)
                pred_image[rr,cc]=prob

        # Saving pred image
        fpath = os.path.join(save_dir,'{}.png'.format( \
            img_name))
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        png16_saver(pred_image, fpath)



if __name__ == '__main__':
    run()