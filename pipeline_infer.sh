# HDoG parameters
min_sigma=1.18
max_sigma=3.1
thr=0.006
ov=1.0
hessian_thr=1.4
# Combining threshold
comb_thr=0.00031622776601683794
# Files and Directories 
data_dir='./data/INbreast-sample/'
save_dir='./output/infer1/'
csv="test.csv"
# Other
num_workers=20
gpu_number=0
model_name="trainedFPN"

echo 'Segmenting breast area'
python infer_breast_segm.py -data_dir $data_dir -val_csv $csv\
  -save_dir $save_dir -num_workers $num_workers

echo 'Applying blob segmentation'
python infer_blob_segm.py -data_dir $data_dir -val_csv $csv\
	-save_dir $save_dir -num_workers $num_workers -DoG_thr $thr -max_sigma $max_sigma \
	-min_sigma $min_sigma -overlap $ov -hessian_thr $hessian_thr

echo 'Applying CNN'
python infer_FPN.py -model_name $model_name \
  -save_dir $save_dir -overwrite_images 1 -data_dir $data_dir\
  -gpu_number $gpu_number -val_csv $csv

echo 'Combining results of blob segmentation and CNN'
python infer_hybrid_approach.py -model_name $model_name\
  -data_dir $data_dir -save_dir $save_dir\
  -val_csv $csv -num_workers $num_workers \
  -hybrid_combining 'overlap_based_combining' -hybrid_combining_overlap 0.3\
  -threshold $comb_thr


