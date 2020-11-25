min_sigma=1.18
max_sigma=3.1
thr=0.006
ov=1.0
hessian_thr=1.4

# # python extract_patches.py

# python train_FPN.py -model_name 'trial' -data_dir '../../Data/INbreast-patches-512-split20-png'\
#   -batch_size 8 -num_workers 10 -radius 10 -alpha 1 -n_epochs 1 -gpu_number 0

# python infer_breast_segm.py -data_dir '../../Data/INbreast-pytorch/' -val_csv 'test-mini.csv'\
#   -save_dir './output/inbreast/' -num_workers 20

# python infer_blob_segm.py -data_dir '../../Data/INbreast-pytorch/' -val_csv 'test-mini.csv'\
# 	-save_dir './output/inbreast/' -num_workers 20 -DoG_thr $thr -max_sigma $max_sigma \
# 	-min_sigma $min_sigma -overlap $ov -hessian_thr $hessian_thr

# python infer_FPN.py -model_name 'trial' \
#   -save_dir './output/inbreast/' -overwrite_images 1 -data_dir '../../Data/INbreast-pytorch/'\
#   -gpu_number 6 -val_csv 'test-mini.csv'

# python infer_hybrid_approach.py -model_name 'trial'\
#   -data_dir '../../Data/INbreast-pytorch/' -save_dir './output/inbreast/'\
#   -val_csv 'test-mini.csv' -num_workers 10 \
#   -hybrid_combining 'overlap_based_combining' -hybrid_combining_overlap 0.3\
#   -threshold 0.00031622776601683794 


# python evaluate_FROC.py -model_name 'trial' -out_csv_file \
# 'trial' -data_dir '../../Data/INbreast-pytorch/' -save_dir './output/inbreast/' \
#   -num_workers 10 \
#   -graph_only 0  -per_unit_area 1 -resolution 0.007 -hessian_thr $hessian_thr\
#   -hybrid_combining 'overlap_based_combining' -hybrid_combining_overlap 0.3\
#   -regression 1  -val_csv 'test-mini.csv'\
#   -mask_col 'mask_image_filled'


# python evaluate_miou.py -model_name 'trial' -out_csv_file \
# 'trial' -data_dir '../../Data/INbreast-pytorch/' -save_dir './output/inbreast/' \
#   -num_workers 10 \
#   -graph_only 0  -per_unit_area 1 -resolution 0.007 -hessian_thr $hessian_thr\
#   -hybrid_combining 'overlap_based_combining' -hybrid_combining_overlap 0.3\
#   -regression 1  -val_csv 'test-mini.csv'\
#   -mask_col 'mask_image_filled'


# python evaluate_object_iou.py -model_name 'trial' -out_csv_file \
# 'trial' -data_dir '../../Data/INbreast-pytorch/' -save_dir './output/inbreast/' \
#   -num_workers 10 \
#   -graph_only 0  -per_unit_area 1 -resolution 0.007 -hessian_thr $hessian_thr\
#   -hybrid_combining 'overlap_based_combining' -hybrid_combining_overlap 0.3\
#   -regression 1  -val_csv 'test-mini.csv'\
#   -mask_col 'mask_image_filled'



# ################
# # Ciecholewski

# python infer_ciecholewski.py  -data_dir '../../Data/INbreast-pytorch/' -val_csv 'test-mini.csv'\
#    -save_dir './output/inbreast/' -num_workers 10


# ################
# # Wang and Yang

# python infer_context.py -model_name 'context2'  -DoG_thr 0.01 -min_sigma 1.6 -max_sigma 2.0 \
# 	-save_dir './output/inbreast/context/' -csv 'test-mini.csv'\
# 	-data_dir '../../Data/INbreast-pytorch/' -gpu_number 0 


##################
# UCLA data

model_name="FPNr10a1nh"
min_sigma=1.18
max_sigma=3.
thr=0.01
ov=1.0
hesssian_thr=0.5
val_csv="mag_annot_nocview.csv"
data_dir='../../Data/UCLAMammoPng/'
save_dir='./output/ucla/mag/untuned/'



# python infer_breast_segm.py -data_dir $data_dir -val_csv $val_csv\
#   -save_dir $save_dir -num_workers 20

# python infer_blob_segm.py -data_dir $data_dir -val_csv $val_csv\
# 	-save_dir $save_dir -num_workers 20 -DoG_thr $thr -max_sigma $max_sigma \
# 	-min_sigma $min_sigma -overlap $ov -hessian_thr $hessian_thr

# python infer_FPN.py -model_name $model_name \
#   -save_dir $save_dir -overwrite_images 1 -data_dir $data_dir\
#   -gpu_number 6 -val_csv $val_csv

# python infer_hybrid_approach.py -model_name $model_name\
#   -data_dir $data_dir -save_dir $save_dir\
#   -val_csv $val_csv -num_workers 20 \
#   -hybrid_combining 'overlap_based_combining' -hybrid_combining_overlap 0.3\
#   -threshold 0.00031622776601683794


model_name="FPNr10a1nh_new" # This is the UCLA tuned model
min_sigma=4.
max_sigma=6.
thr=0.001
ov=1.0
hesssian_thr=1.0
val_csv="mag_annot_nocview.csv"
data_dir='../../Data/UCLAMammoPng/'
save_dir='./output/ucla/mag/tuned/'



# python infer_breast_segm.py -data_dir $data_dir -val_csv $val_csv\
#   -save_dir $save_dir -num_workers 20

# python infer_blob_segm.py -data_dir $data_dir -val_csv $val_csv\
# 	-save_dir $save_dir -num_workers 20 -DoG_thr $thr -max_sigma $max_sigma \
# 	-min_sigma $min_sigma -overlap $ov -hessian_thr $hessian_thr

# python infer_FPN.py -model_name $model_name \
#   -save_dir $save_dir -overwrite_images 1 -data_dir $data_dir\
#   -gpu_number 6 -val_csv $val_csv

python infer_hybrid_approach.py -model_name $model_name\
  -data_dir $data_dir -save_dir $save_dir\
  -val_csv $val_csv -num_workers 20 \
  -hybrid_combining 'overlap_based_combining' -hybrid_combining_overlap 0.3\
  -threshold 0.00031622776601683794 
