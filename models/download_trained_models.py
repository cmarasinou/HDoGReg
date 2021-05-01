from torch.hub import download_url_to_file
import os

model_urls = [
    'https://github.com/cmarasinou/HDoGReg/releases/download/v0.1/wang_yang.tar',
    'https://github.com/cmarasinou/HDoGReg/releases/download/v0.1/trainedFPN.pth',
    ]
model_filenames = [
    'wang_yang.tar',
    'trainedFPN.pth',
    ]

model_dir = os.path.dirname(os.path.realpath(__file__))

for filename, url in zip(model_filenames, model_urls):
    try:
        fpath = os.path.join(model_dir,filename)
        download_url_to_file(url,filename, progress=True)
    except:
        print('Cannot download {}'.format(url))