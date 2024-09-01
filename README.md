# ViTAS
This is the official repo for our work 'Playing to Vision Foundation Model's Strengths in Stereo Matching'.  
[Paper](https://arxiv.org/abs/2404.06261)  

[Demo Video](https://www.youtube.com/watch?v=vL3xjHFYgP0)  

## Setup
We built and ran the repo with CUDA 11.8, Python 3.9.0, and Pytorch 2.1.0. For using this repo, please follow the instructions below:
```
pip install -r requirements.txt
```

If you have any problem with installing xFormers package, please follow the guidance in [DINOv2](https://github.com/facebookresearch/dinov2).

## Pre-trained models

Pretrained models leading to our [SoTA KITTI benchmark results](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) can be downloaded from [google drive](https://drive.google.com/file/d/15hwxnvN53PhV3-7GGH0N_AM4XlD9DEsQ/view?usp=sharing), and is supposed to be under dir: `toolkit/models/ViTASIGEV`.

Results of our KITTI benchmark results can be downloaded from [2012(google drive)](https://drive.google.com/file/d/1HxZIrBZvYjt8g4NOXisj3TxkQmrjvQ66/view?usp=drive_link) and [2015(google drive)](https://drive.google.com/file/d/15jCzdIa_2gxF7LLoulsdY3vwRNwBB6_t/view?usp=drive_link).

## Dataset Preparation
To train/evaluate ViTAStereo, you will need to download the required datasets.

* [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#:~:text=on%20Academic%20Torrents-,FlyingThings3D,-Driving) (Includes FlyingThings3D, Driving)
* [Virtual KITTI 2]([https://vision.middlebury.edu/stereo/data/](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/))
* [Middlebury](https://vision.middlebury.edu/stereo/data/)
* [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-test-data)
* [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)
* [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

You can create symbolic links to wherever the datasets were downloaded in the `$root/datasets` folder:

```shell
ln -s $YOUR_DATASET_ROOT datasets
```

Our folder structure is as follows:

```
├── datasets
    ├── ETH3D
    │   ├── testing
    │       ├── lakeside_1l
    │       └── ...
    │   └── training
    │       ├── delivery_area_1l
    │       └── ...
    │
    ├── KITTI
    │   ├── 2012
    │   │   ├── testing
    │   │   └── training
    │   └── 2015
    │       ├── testing
    │       └── training
    ├── middlebury
    │   ├── 2005
    │   ├── 2006
    │   ├── 2014
    │   ├── 2021
    │   └── MiddEval3
    └── sceneflow
        ├── driving
        │   ├── 15mm_focallength
        │   │   ├── scene_backwards
        │   │   └── scene_fowwards
        │   └── 35mm_focallength
        ├── flying
        │   ├── TRAIN
        │   │   ├── A
        │   │   ├── B
        │   │   └── C
        │   └── Test
```
## Training & Evaluation
The demos of training and evaluating function of our ViTAS  are integrated in the  `toolkit/main.py`. 

## Acknowledgment
Some of this repo come from [IGEV-Stereo](https://github.com/gangweiX/IGEV),[GMStereo](https://github.com/autonomousvision/unimatch), and [DINOv2](https://github.com/facebookresearch/dinov2).
