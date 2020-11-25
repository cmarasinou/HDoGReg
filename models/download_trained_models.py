from torch.hub import download_url_to_file

model_urls = [
    'https://github.com/cmarasinou/HDoGReg/releases/download/v0.1alpha/context2.tar',
    'https://github.com/cmarasinou/HDoGReg/releases/download/v0.1alpha/FPNinceptionBase1channel.pth',
    ]
model_filenames = [
    'context2.tar',
    'FPNinceptionBase1channel.pth',
    ]


for filename, url in zip(model_filenames, model_urls):
    try:
        download_url_to_file(url,filename, progress=False)
    except:
        print('Cannot download {}'.format(url))