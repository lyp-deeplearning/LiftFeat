## LiftFeat: 3D Geometry-Aware Local Feature Matching  
Training code is now available

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
 - [Inference](#inference)
 - [Training](#training)
 - [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## Introduction
This repository contains the official implementation of the paper: *[LiftFeat: 3D Geometry-Aware Local Feature Matching]*, to be presented at ICRA 2025.

**Overview of LiftFeat's achitecture**
<div style="background-color:white">
    <img align="center" src="./assert/achitecture.png" width=1000 />
</div>

## Installation
If you use conda as virtual environment,you can create a new env with:
```bash
git clone xxx
cd xxx
conda create -n liftfeat python=3.8
conda activate liftfeat

pip install -r requirements.txt
```

## Usage
### Inference
To run LiftFeat on an image,you can simply run with:
```bash
python demo.py --image_path <path to your image>
```

### Training
To train LiftFeat as described in the paper, you will need MegaDepth & COCO_20k subset of COCO2017 dataset as described in the paper *[XFeat: Accelerated Features for Lightweight Image Matching](https://arxiv.org/abs/2404.19174)*
You can obtain the full COCO2017 train data at https://cocodataset.org/.
However, we [make available](https://drive.google.com/file/d/1ijYsPq7dtLQSl-oEsUOGH1fAy21YLc7H/view?usp=drive_link) a subset of COCO for convenience. We simply selected a subset of 20k images according to image resolution. Please check COCO [terms of use](https://cocodataset.org/#termsofuse) before using the data.

To reproduce the training setup from the paper, please follow the steps:
1. Download [COCO_20k](https://drive.google.com/file/d/1ijYsPq7dtLQSl-oEsUOGH1fAy21YLc7H/view?usp=drive_link) containing a subset of COCO2017;
2. Download MegaDepth dataset. You can follow [LoFTR instructions](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md#download-datasets), we use the same standard as LoFTR. Then put the megadepth indices inside the MegaDepth root folder following the standard below:
```bash
{megadepth_root_path}/train_data/megadepth_indices #indices
{megadepth_root_path}/MegaDepth_v1 #images & depth maps & poses
```
3. Finally you can call training
```bash
python train.py --megadepth_root_path <path_to>/MegaDepth --synthetic_root_path <path_to>/coco_20k --ckpt_save_path /path/to/ckpts
```

### Evaluation
All evaluation code are in *evaluation*

**Homography Estimation**
```bash
python evaluation/hpatch_evaluation.py
```

**Relative Pose Estimation**

For *Megadepth1500* dataset:
```bash
python evaluation/megadepth_evaluation.py
```

For *ScanNet1500* dataset:
```bash
python evaluation/scannet_evaluation.py
```

**Visual Localization**
```bash
python evaluation/aachen_day_night_evaluation.py
```

## Citation
If you find this code useful for your research, please cite the paper:



## License
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
