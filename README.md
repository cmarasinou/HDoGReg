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

TODO: Describe the directory structure of the dataset & provide sample from INbreast or all

## Running the code

TODO: How to run pipelines

## Tutorials

TODO: Point to notebook tutorials