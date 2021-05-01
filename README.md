# HDoGReg: Segmentation and Grouping of Microcalcifications in 2D Mammography With Hessian Difference of Gaussians and Regression

This repository contains the code for our paper.

TODO:Paper in citation format


## Abstract

Accurate microcalcification (MC) segmentation in 2D full-field digital mammography has shown to benefit computerized diagnosis of breast cancer. However, the automated identification and segmentation of microcalcifications remains a challenging task with high false positive rate. We present Hessian difference-of-Gaussians regression (HDoGReg), a two stage multi-scale approach to MC segmentation. Candidate high optical density objects are first delineated using blob detection and Hessian analysis. A regression convolutional network, trained  to output a function with higher response near MCs, chooses the objects which constitute actual MCs. The method is trained and validated on 435 mammograms from two separate datasets. HDoGReg achieved a mean intersection over the union of 0.670$\pm$0.121 per image, intersection over the union per MC object of 0.607$\pm$0.250 and true positive rate of 0.75 at 0.4 false positive detections per $cm^2$. The results of HDoGReg perform better when compared to state-of-the-art MC segmentation and detection methods.

## Approach

![HDoGReg: Approach for segmenting individual microcalcifications.](https://github.com/cmarasinou/HDoGReg/blob/master/documentation/figures/SegmentationPipeline.png)

HDoGReg: Approach for segmenting individual MCs. Although the segmentation is performed on whole mammograms, for visualization purposes only a patch is being shown. In the upper branch, blob segmentation is per- formed to segment bright blob-like and tubular structures. In the lower branch, a regression convolutional neural network gives a continuous function with higher response close to MCs. A threshold is then applied to segment individ- ual MCs. The output from the two branches is combined based on an overlap criterion, resulting to the final segmentation mask.

## Results

Here are some segmentation examples applying HDoGReg and the comparison method. For visualization purposes we present 256x256 sized patches with the single-pixel
reference annotation objects dilated. From left to right: (a) Images; (b) Reference Annotation; (c) HDoGReg Predicted Segmentation; (d) Ciecholewski (2017)
Segmentation. The three top patches are from INbreast data and the rest are our data.

![Segmentation Examples](https://github.com/cmarasinou/HDoGReg/blob/master/documentation/figures/ResultExamples.png)


## Requirements & Containers

TODO

## Preparing your data

The dataset should have the following directory structure. A specific format for the image file names does not apply. 

```
dataset_directory/
├── images
│   └── png16
│       ├── image1.png
│       └── mask1.png
│           ...
└── other
    ├── test.csv
    ├── train.csv
    └── val.csv
```

The CSV files point to the files and should have the following structure:

```
train.csv

 , full_image, mask_image
0, image1, mask1
        ...
```

A sample dataset can be found in 

## Running the code

1. Follow the [Docker Instructions](https://github.com/cmarasinou/HDoGReg/blob/master/docker/README.md) to create and run a container.
2. Download Sample Data and Trained Models
```bash
python download_sample_data.py -target_dir './data/'
python models/download_trained_models.py
```
3. Run inference
```bash
./pipeline_infer.sh
```
4. Run training
```bash
./pipeline_train.sh
```
5. Run evaluation
TODO

## Tutorials

- [Tutorial: Individual components of HDoGReg are applied to produce MC segmentation](https://github.com/cmarasinou/HDoGReg/blob/master/notebooks/microcalcification-segmentation-tutorial.ipynb)