import os
import argparse
from torch.hub import download_url_to_file
import zipfile

data_url = 'https://github.com/cmarasinou/HDoGReg/releases/download/v0.1alpha/INbreast-sample.zip'
data_filename = 'INbreast-sample.zip'
local_filepath =  "../../Data/INbreast-sample/INbreast-sample.zip" #In case download fails

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description=\
        "Download sample data and extract them in target directory")
    parser.add_argument("-target_dir", type=str, help="Absolute directory to sae data")
    args = parser.parse_args()
    target_dir = args.target_dir
    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)


    # download data in target_dir
    filepath = os.path.join(target_dir, data_filename)
    try:
        download_url_to_file(data_url, filepath, progress=False)
    except:
        print('Cannot download {}'.format(data_url))
        filepath = local_filepath
        
    # extract data
    with zipfile.ZipFile(filepath, "r") as zip_file:
        zip_file.extractall(target_dir)