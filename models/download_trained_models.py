from torch.hub import download_url_to_file
import os

model_urls = [
    'https://github.com/cmarasinou/HDoGReg/releases/download/v0.1alpha/context2.tar',
    'https://github.com/cmarasinou/HDoGReg/releases/download/v0.1alpha/FPNinceptionBase1channel.pth',
    ]
model_filenames = [
    'context2.tar',
    'FPNinceptionBase1channel.pth',
    ]

model_dir = os.path.dirname(os.path.realpath(__file__))

for filename, url in zip(model_filenames, model_urls):
    try:
        fpath = os.path.join(model_dir,filename)
        download_url_to_file(url,filename, progress=True)
    except:
        print('Cannot download {}'.format(url))