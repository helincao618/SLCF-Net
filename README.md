# SLCF-Net
[ICRA 2024] SLCF-Net: Sequential LiDAR-Camera Fusion for Semantic Scene Completion using a 3D Recurrent U-Net

[Helin (Henry) Cao](https://helincao618.github.io/),
[Sven Behnke](https://www.ais.uni-bonn.de/behnke/)  
University of Bonn, Bonn, Germany

If you find this work or code useful, please cite our [paper](https://arxiv.org/abs/2403.08885) and give this repo a star
```
@inproceedings{cao2024slcf,
    title={SLCF-Net: Sequential LiDAR-Camera Fusion for Semantic Scene Completion using a 3D Recurrent U-Net}, 
    author={Helin Cao and Sven Behnke},
    booktitle={ICRA},
    year={2024}
}
```

# Teaser

<img src="./teaser/SLCF-Net.gif"  />

# Table of Content
- [News](#news)
- [Preparing](#preparing)
  - [Installation](#installation)  
  - [Datasets](#datasets)
- [Running SLCF-Net](#running-slcfnet)
  - [Training](#training)
  - [Evaluating](#evaluating)
- [Inference & Visualization](#inference--visualization)
  - [Inference](#inference)
  - [Visualization](#visualization)
- [License](#license)

# News
- 10/04/2024: We will release the code soon!

# Preparing

## Installation

1. Create conda environment:

```
$ conda create -y -n slcfnet
$ conda activate slcfnet
```
2. This code was implemented with python 3.7, pytorch 1.7.1 and CUDA 10.2. Please install [PyTorch](https://pytorch.org/): 

```
$ conda install 
```

3. Install the additional dependencies:

```
$ cd SLCF-Net/
$ pip install -r requirements.txt
```


## Datasets

1. You need to download

      - The **Semantic Scene Completion dataset v1.1** (SemanticKITTI voxel data (700 MB)) from [SemanticKITTI website](http://www.semantic-kitti.org/dataset.html#download)
      -  The **KITTI Odometry Benchmark calibration data** (Download odometry data set (calibration files, 1 MB)) and the **RGB images** (Download odometry data set (color, 65 GB)) from [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
      - The dataset folder at **/path/to/semantic_kitti** should have the following structure:
    ```
    └── /path/to/semantic_kitti/
      └── dataset
        ├── poses
        └── sequences
    ```


2. Create a folder to store SemanticKITTI preprocess data at `/path/to/kitti/preprocess/folder`.

3. Store paths in environment variables for faster access (**Note: folder 'dataset' is in /path/to/semantic_kitti**):

```
$ export KITTI_PREPROCESS=/path/to/kitti/preprocess/folder
$ export KITTI_ROOT=/path/to/semantic_kitti 
```



# Running SLCF-Net

## Training


## Evaluating 

To evaluate SLCF-Net on SemanticKITTI validation set, type:


# Inference & Visualization

## Inference

## Visualization




# License
SLCF-Net is released under the [MIT license](./LICENSE).
