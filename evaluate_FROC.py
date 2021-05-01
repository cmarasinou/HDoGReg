# For evaluating HDoGReg with FROC analysis

import matplotlib
# To avoid using Xwindows backend
matplotlib.use('Agg') 
import ray
import os
from tqdm import tqdm
from math import log
import numpy as np

from ruamel.yaml import YAML
import torch
from skimage.morphology import remove_small_holes, binary_erosion, square, remove_small_objects
from skimage.measure import regionprops
from skimage.feature import blob_dog, peak_local_max
from skimage.util import img_as_float
from skimage.feature.blob import _prune_blobs
from scipy import ndimage
from hdogreg.image import  overlap_based_combining, hybrid_approach
from hdogreg.dataset import ImageDataset, bboxInImage
from hdogreg.metrics import get_confusion_matrix_2
from hdogreg.utils import arg_parsing

import pandas as pd
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
import torch
from skimage.draw import circle
import mlflow
import cv2
from sklearn.metrics import auc


def logging_params(params_list):
    for name, value in params_list:
        mlflow.log_param(name, value)

def logging_metrics(metrics_dict):
    for name, value in metrics_dict.items():
        mlflow.log_metric(name, value)

def make_froc_data(data_computed, total_area=None):
    if total_area is not None:
        per_unit_area=True
    else:
        per_unit_area=False
    thr_list =  data_computed.columns
    llf = dict()
    llf_region = dict()
    llf_dot = dict()
    nlf = dict()
    pos = dict()

    for thr in thr_list:
        tp_all, fp_all, fn_all, tp_dot_all, fn_dot_all, n_samples = 0,0,0,0,0,0
        for cfmtrix in data_computed[thr]:
            tp, fp, fn,tp_dot,fn_dot = cfmtrix
            tp_all += tp
            fp_all += fp
            fn_all += fn
            tp_dot_all += tp_dot
            fn_dot_all += fn_dot
            n_samples += 1
        if tp_all+fn_all==0:
            llf[thr]=0
        else:
            llf[thr] = tp_all/(tp_all+fn_all)
        tp_region_all = tp_all - tp_dot_all
        fn_region_all = fn_all - fn_dot_all
        if tp_region_all+fn_region_all==0:
            llf_region[thr]=0
        else:
            llf_region[thr] = tp_region_all/(tp_region_all+fn_region_all)
        if tp_dot_all+fn_dot_all==0:
            llf_dot[thr]=0
        else:
            llf_dot[thr] = tp_dot_all/(tp_dot_all+fn_dot_all)
        if per_unit_area:
            nlf[thr] = fp_all/total_area
            pos[thr] = (tp_all+fp_all)/total_area
        else:
            nlf[thr] = fp_all/n_samples
            pos[thr] = (tp_all+fp_all)/n_samples
    llf_list =[]
    llf_region_list =[]
    llf_dot_list =[]
    nlf_list = []
    pos_list = []
    thr_list = []
    for k in llf:
        llf_list.append(llf[k])
        llf_region_list.append(llf_region[k])
        llf_dot_list.append(llf_dot[k])
        nlf_list.append(nlf[k])
        pos_list.append(pos[k])
        thr_list.append(float(k))
    return nlf_list, llf_list

def find_auc(FP,TPR, max_FP=1):
    x,y = FP,TPR
    x, y = np.array(x), np.array(y)
    x, y = x[x<=max_FP], y[x<=max_FP]
    x, y = np.append(x,0), np.append(y,0)
    x, y = x[np.argsort(x)], y[np.argsort(x)]
    x, y = np.append(x,max_FP), np.append(y,y[-1])
    
    return auc(x,y)


def make_froc_graphs(pickle_files, data_names,colors, max_FP = 10, bootstraps=None ,save=None):
    froc_data=dict()
    paucs = list()
    plt.figure(figsize=(10,7))
    plt.title('FROC curve', fontsize=20)
    plt.xlabel(r'FP per unit area ($cm^2$)', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('FROC curve', fontsize=20)
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.2', color='black')
    plt.ylim(0.0,1.0)
    plt.xlim(0.0,max_FP)
    plt.tight_layout()
    for pickle_file, name,clr in zip(pickle_files, data_names, colors):
        data_dict=pickle.load(open(pickle_file,'rb'))
        data_computed = data_dict['dataframe']
        thr_list =  data_computed.columns
        resolution = .007
        total_area = data_dict['total_area']
        froc_data[name] = make_froc_data(data_computed, total_area)
        froc_data[name] = (froc_data[name][0]+[0], froc_data[name][1]+[0])
        x,y = froc_data[name]
        x, y = np.array(x), np.array(y)
        froc_data[name] = x[np.argsort(x)], y[np.argsort(x)] 
        sns.lineplot(*froc_data[name],color=clr)
        if bootstraps is None:
            paucs.append(find_auc(*froc_data[name], max_FP=max_FP))
        else:
            paucs.append(bootstrap_pauc(data_computed,max_FP=max_FP,total_area=total_area, bootstraps=bootstraps))
            y_errors = bootstrap_froc(data_computed,max_FP=max_FP,total_area=total_area, bootstraps=bootstraps)
            plt.fill_between(froc_data[name][0], 
                             froc_data[name][1]-np.append(y_errors,0),
                             froc_data[name][1]+np.append(y_errors,0), 
                                                          alpha=0.1, color=clr)
    if bootstraps is None:
        pauc = paucs[0]
    else:
        pauc = paucs[0][0]
    plt.legend(data_names, loc=4, frameon=False)
    if save is not None:
        plt.savefig(save, dpi=200)
        print(f'FILE SAVED. FROC curve was saved at: {save}')
    plt.show()
    return pauc
    

def make_froc_graphs_no_unit_area(pickle_files, data_names,colors, max_FP = 10, bootstraps=None ,save=None,
                                 no_zero_threshold=False):
    froc_data=dict()
    paucs = list()
    plt.figure(figsize=(10,7))
    plt.title('FROC curve', fontsize=20)
    plt.xlabel('FP per image', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('FROC curve', fontsize=20)
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.2', color='black')
    plt.ylim(0.0,1.0)
    plt.xlim(0.0,max_FP)
    plt.tight_layout()
    for pickle_file, name,clr in zip(pickle_files, data_names, colors):
        data_dict=pickle.load(open(pickle_file,'rb'))
        data_computed = data_dict['dataframe']
        if no_zero_threshold:
            data_computed = data_computed.drop(columns=0.0)
        
        thr_list =  data_computed.columns
        resolution = .007
        total_area = None
        froc_data[name] = make_froc_data(data_computed,total_area)
        froc_data[name] = (froc_data[name][0]+[0], froc_data[name][1]+[0])
        x,y = froc_data[name]
        x, y = np.array(x), np.array(y)
        froc_data[name] = x[np.argsort(x)], y[np.argsort(x)] 
        sns.lineplot(*froc_data[name],color=clr)
        if bootstraps is None:
            paucs.append(find_auc(*froc_data[name], max_FP=max_FP))
        else:
            paucs.append(bootstrap_pauc(data_computed,max_FP=max_FP,total_area=total_area, bootstraps=bootstraps))
            y_errors = bootstrap_froc(data_computed,max_FP=max_FP,total_area=total_area, bootstraps=bootstraps)
            plt.fill_between(froc_data[name][0], 
                             froc_data[name][1]-np.append(y_errors,0),
                             froc_data[name][1]+np.append(y_errors,0), 
                                                          alpha=0.1, color=clr)
    if bootstraps is None:
        pauc = [r'{} pAUC={:.3f}'.format(n,a) for n,a in zip(data_names,paucs)]
    else:
        pauc = [r'{} pAUC={:.3f}$\pm${:.3f}'.format(n,a[0],a[1]) for n,a in zip(data_names,paucs)]
    plt.legend(data_names, loc=4, frameon=False)
    if save is not None:
        plt.savefig(save, dpi=200)
        print(f'FILE SAVED. FROC curve was saved at: {save}')
    plt.show()
    return pauc[0]
    
    
def bootstrap_pauc(data, max_FP=1, total_area=None,  bootstraps=1000, seed=0):
    aucs = list()
    np.random.seed(seed)
    for i in range(bootstraps):
        data_sampled = data.sample(len(data), replace=True)
        froc_data = make_froc_data(data_sampled, total_area)
        aucs.append(find_auc(*froc_data, max_FP=max_FP))
    return np.mean(aucs), 2*np.std(aucs)

def bootstrap_froc(data, max_FP=1, total_area=None,  bootstraps=1000, seed=0):
    froc_list = list()
    np.random.seed(seed)
    for i in range(bootstraps):
        data_sampled = data.sample(len(data), replace=True)
        froc_data = make_froc_data(data_sampled, total_area)
        froc_list.append(froc_data)
    froc_list = np.array(froc_list)
    return np.std(froc_list[:,1,:],0)

def run():
    # Load configuration contents
    config_fname = os.path.abspath(os.path.dirname(os.path.realpath(__file__))+'/validation_config.yaml')
    with open(config_fname) as config_f:
        config_dict = YAML().load(config_f)['confusion_matrix']

    # Update config settings using args
    config_dict = arg_parsing(config_dict)
    
    data_dir = config_dict['data_dir']
    thr_list = config_dict['threshold']
    csv_file = config_dict['val_csv']
    model_name = config_dict['model_name']
    radius = float(config_dict['radius'])
    alpha = float(config_dict['alpha'])
    save_dir = config_dict['save_dir']
    mask_col = config_dict['mask_col']
    num_workers = int(config_dict['num_workers'])

    out_csv_file = config_dict['out_csv_file']
    small_area = float(config_dict['small_area'])
    distance_thr = float(config_dict['distance_thr'])
    iou_thr = float(config_dict['iou_thr'])
    resolution = float(config_dict['resolution'])
    per_unit_area = int(config_dict['per_unit_area'])
    hybrid_combining = config_dict['hybrid_combining']
    hybrid_combining_overlap = float(config_dict['hybrid_combining_overlap'])
    regression = int(config_dict['regression'])
    if not regression:
        thr_list = np.concatenate((np.logspace(-6,-1,5),np.linspace(0,0.9,10), 1-np.logspace(-6,-1,5)))
        # thr_list = [0.5]
    else:
        thr_list = np.linspace(0,radius,20)
        thr_list = [(np.exp(alpha*(1-d/radius))-1)/(np.exp(alpha)-1) for d in thr_list]

    hessian_thr = float(config_dict['hessian_thr'])
    comment = config_dict['comment']
    graph_only = int(config_dict['graph_only'])

    blob_dir = os.path.join(save_dir,'blob/')
    breast_dir = os.path.join(save_dir,'breast_segm/')
    pred_fpn_dir = os.path.join(save_dir,'predsFPN/')


    if not graph_only:
        logging_params([
        ("data_dir", os.path.basename(os.path.dirname(data_dir))),
        ('val_csv', csv_file), 
        ('model_name',model_name),
        ('radius', radius),
        ('alpha', alpha),
        ('hessian_thr', hessian_thr),
        ('comment',comment),
        ('out_csv_file',out_csv_file)
        ])


        ##########################################################
        # Load dataset
        print("\n------Loading Dataset------\n")
        ds = ImageDataset(data_dir, csv_file, img_format='png16',
                          images_list=['full_image',
                                       mask_col],
                         attributes_list=['full_image']
                        )
        print("\n------Initializing Ray------\n")
        ray.init(num_cpus=num_workers)#, webui_host='127.0.0.1')
        # put dataset on remote
        ds_id = ray.put(ds)
        ##########################################################
        # Defining the remote function, after ds in the remote
        @ray.remote
        def main_func(idx, thr, ds_id):
            img, mask, img_name = ds_id.__getitem__(idx)
            # Loading the segmentations of blobs and the breast mask
            blob_path = os.path.join(blob_dir,img_name+'.png')
            breast_path = os.path.join(breast_dir,img_name+'.png')
            blob_mask = ds_id._img_to_numpy(blob_path)
            breast_mask = ds_id._img_to_numpy(breast_path)
            pred_fpn_path = os.path.join(pred_fpn_dir,img_name+'.png')
            pred_fpn = ds_id._img_to_numpy(pred_fpn_path)

            pred_mask = pred_fpn >= thr

            pred_mask = hybrid_approach(pred_mask,breast_mask,blob_mask,
                hybrid_combining=hybrid_combining, hybrid_combining_overlap=hybrid_combining_overlap)
            #pred_mask = remove_small_objects(pred_mask.astype(bool), min_size=5)

            return get_confusion_matrix_2(pred_mask, mask, 
                               distance_thr = distance_thr, IoU_thr=iou_thr, 
                               small_object_area = small_area,
                               return_dot_results=1)
        data = dict()
        for thr in thr_list:
            # Loading predictions
            data[thr]=[]
            for idx in tqdm(range(0,ds.__len__())):
                data[thr].append(\
                    main_func.remote(idx, thr, ds_id))
        data_computed = dict()
        for thr in thr_list:
            data_computed[thr] = ray.get(data[thr])
        df = pd.DataFrame(data_computed)
        data_dict = {'dataframe': df}
        #csv_path = os.path.join(save_dir, out_csv_file)
        #df.to_csv(csv_path)
        #ray.shutdown()

        ds = ImageDataset(data_dir, csv_file, img_format='png16',
                      images_list=['full_image'],
                     attributes_list=['full_image']
                    )
        image_list = list(ds.df.full_image)

        #ray.init(num_cpus=num_workers, webui_host='127.0.0.1')
        if per_unit_area:
            @ray.remote
            def area_func(idx,ds_id):
                # Loading the segmentations of blobs and the breast mask
                img_name = image_list[idx]
                breast_path = os.path.join(breast_dir,img_name+'.png')
                breast_mask = ds_id._img_to_numpy(breast_path)
                breast_area = breast_mask.sum()
                # Area in cm^2
                breast_area = breast_area*resolution**2
                return breast_area
            data = list()
            for idx in range(0,ds.__len__()):
                data.append(area_func.remote(idx,ds_id))
            data = ray.get(data)
            total_area = np.sum(data)
            print("Overall area {} cm^2".format(total_area))
            data_dict['total_area']=total_area
        ray.shutdown()

        # Saving computations
        pickle_file = os.path.join(save_dir, out_csv_file+'FROC.p')
        pickle.dump(data_dict, open( pickle_file, "wb" ) )

    # Build FROC curve
    if graph_only:
        # Load data
        pickle_file = os.path.join(save_dir, out_csv_file+'FROC.p')
        data_dict = pickle.load( open( pickle_file, "rb" ) )
        if per_unit_area:
            total_area = data_dict['total_area']
    data_computed = data_dict['dataframe']


    save_path = os.path.join(save_dir,'{}FROC.png'.format(out_csv_file))
    if per_unit_area:
        pauc=make_froc_graphs([pickle_file],[model_name], ['blue'],
            max_FP=1, bootstraps=100, save=save_path)
    else:
        pauc=make_froc_graphs_no_unit_area([pickle_file],[model_name],['blue'],
            max_FP=100, bootstraps=100, save=save_path,
            no_zero_threshold=True)


    logging_metrics({
            'pauc':pauc,
            })
    print(f'RESULT FOUND. partial area under curve (pAUC) = {pauc}')


if __name__ == '__main__':
    mlflow.set_experiment('EvaluateFROC')
    with mlflow.start_run():
        run()
