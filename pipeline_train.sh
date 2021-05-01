# Proximity function parameters
radius=10
alpha=1
# Files and Directories 
data_dir='./data/INbreast-sample/'
patch_dir='./data/patches/'
# Training Parameters
num_workers=20
gpu_number=0
batch_size=8
n_epochs=1
model_name="new_model"

#Running
python extract_patches.py -data_dir $data_dir -patches_dir $patch_dir

python train_FPN.py -model_name $model_name -data_dir $patch_dir\
   -batch_size $batch_size -num_workers $num_workers\
  -radius $radius -alpha $alpha -n_epochs $n_epochs -gpu_number $gpu_number


