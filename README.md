# Apparent Age Estimation using ResNet34-CBAM

**Author:** Bhoomi Priya

## Overview
This repository contains a PyTorch implementation for apparent facial age estimation using the APPA-REAL dataset. The project adapts a pretrained ResNet34 architecture integrated with Convolutional Block Attention Modules (CBAM) for continuous age regression. 


## Dataset
* **APPA-REAL Dataset:** Contains facial images annotated with real and apparent ages.
* **Preprocessing:** Images resized to $224 \times 224$, normalized using ImageNet statistics, and augmented (random flips, rotations, and color jittering).

## Architecture & Training Strategy
* **Backbone:** ResNet34 (Pretrained on ImageNet)
* **Attention:** CBAM (Spatial & Channel Attention) inserted for enhanced facial feature extraction.
* **Loss Function:** Smooth L1 (Huber) Loss
* **Optimizer:** AdamW with Weight Decay ($10^{-4}$)
* **Differential Learning Rates:** * `1e-5` for the ResNet34 backbone (preserves robust feature extractors).
  * `1e-3` for the CBAM blocks and regression head (encourages rapid adaptation).
* **Callbacks:** ReduceLROnPlateau and Early Stopping (Patience = 10).

## Results
Evaluated on a test split of 1,978 unseen images, the model demonstrates highly stable and consistent predictive performance:

| Metric | Value |
| :--- | :--- |
| **MAE** | 5.6677 years |
| **RMSE** | 7.9546 years |
| **MSE** | 63.2763 |
| **$R^2$ Score** | 0.7974 |
| **Pearson Correlation** | 0.8975 |

## How to Run

**1. Train the model:**
```bash
python train.py --data_root ./dataset --model resnet34-cbam --batch_size 32
