# DoG parameters
min_sigma=1.18
max_sigma=3.1
threshold=0.01
# Files and Directories 
data_dir='./data/INbreast-sample/'
save_dir='./output/infer_other/'
csv="test.csv"
# Other
num_workers=20
gpu_number=0
model_name="wang_yang"

################
# Ciecholewski
echo 'Applying Segmentation, Ciecholewski'
python infer_ciecholewski.py \
    -data_dir $data_dir \
    -val_csv $csv \
    -save_dir $save_dir \
    -num_workers $num_workers


################
# Wang and Yang

echo 'Applying detection, Wang and Yang'
python infer_context.py \
    -model_name $model_name \
    -DoG_thr $threshold -min_sigma $min_sigma -max_sigma $max_sigma \
	  -save_dir $save_dir -csv $csv\
	  -data_dir $data_dir -gpu_number $gpu_number