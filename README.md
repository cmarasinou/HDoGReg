# HDoGReg: Segmentation and Grouping of Microcalcifications in 2D Mammography With Hessian Difference of Gaussians and Regression

This repository contains the code for our paper:
??

## Abstract
Accurate microcalcification (MC) segmentation in 2D full-field digital mammography has shown to benefit computerized diagnosis of breast cancer. However, the automated identification and segmentation of microcalcifications remains a challenging task with high false positive rate. We present Hessian difference-of-Gaussians regression (HDoGReg), a two stage multi-scale approach to MC segmentation. Candidate high optical density objects are first delineated using blob detection and Hessian analysis. A regression convolutional network, trained  to output a function with higher response near MCs, chooses the objects which constitute actual MCs. The method is trained and validated on 435 mammograms from two separate datasets. HDoGReg achieved a mean intersection over the union of 0.670$\pm$0.121 per image, intersection over the union per MC object of 0.607$\pm$0.250 and true positive rate of 0.75 at 0.4 false positive detections per $cm^2$. The results of HDoGReg perform better when compared to state-of-the-art MC segmentation and detection methods.

## Approach

??Figure??

