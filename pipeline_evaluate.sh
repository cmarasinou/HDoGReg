# Files and Directories 
data_dir='./data/INbreast-sample/'
save_dir='./output/infer1/'
csv="test.csv"
out_csv="froc.csv"
# Other
num_workers=4
gpu_number=0
model_name="trainedFPN"

echo 'Performing FROC analysis'
python evaluate_FROC.py \
    -model_name $model_name \
    -out_csv_file $out_csv \
    -data_dir $data_dir \
    -save_dir $save_dir \
    -num_workers $num_workers \
    -graph_only 0  -per_unit_area 1 -resolution 0.007 \
    -hybrid_combining 'overlap_based_combining' -hybrid_combining_overlap 0.3 \
    -regression 1 -val_csv $csv \
    -mask_col 'mask_image'

echo 'Evaluating segmentation metrics'
python evaluate_miou.py \
    -model_name $model_name \
    -out_csv_file $out_csv \
    -data_dir $data_dir \
    -save_dir $save_dir \
    -num_workers $num_workers \
    -graph_only 0  -per_unit_area 1 -resolution 0.007 \
    -hybrid_combining 'overlap_based_combining' -hybrid_combining_overlap 0.3 \
    -regression 1 -val_csv $csv \
    -mask_col 'mask_image'


python evaluate_object_iou.py\
    -model_name $model_name \
    -out_csv_file $out_csv \
    -data_dir $data_dir \
    -save_dir $save_dir \
    -num_workers $num_workers \
    -graph_only 0  -per_unit_area 1 -resolution 0.007 \
    -hybrid_combining 'overlap_based_combining' -hybrid_combining_overlap 0.3 \
    -regression 1 -val_csv $csv \
    -mask_col 'mask_image'