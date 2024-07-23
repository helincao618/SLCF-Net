<div align="center">
  <h1 align="center">SLCF-Net: Sequential LiDAR-Camera Fusion for Semantic Scene Completion using a 3D Recurrent U-Net</h1>

  <p align="center">
    <a href="https://helincao618.github.io/">Helin Cao</a> and <a href=https://www.ais.uni-bonn.de/behnke/ target=_blank rel=noopener>Sven Behnke</a>
      <br>
      University of Bonn and Lamarr Institute, Bonn, Germany
    <br />
    <strong>ICRA 2024</strong>
    <br />
    <a>[![arXiv](https://img.shields.io/badge/arXiv%20%2B%20supp-2112.00726-purple)](https://arxiv.org/abs/2403.08885)</a> | <a>[![Project page](https://img.shields.io/badge/Project%20Page-MonoScene-red)](https://sites.google.com/view/slcf-net)</a>
    <br />
  </p>
</div>

# Teaser

<img src="./teaser/SLCF-Net.gif"  />

# Table of Content
- [Preparing](#preparing)
  - [Setup](#setup)  
  - [Datasets](#datasets)
- [Running](#running)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
  - [Visualization](#visualization)
- [Citation](#citation)
- [License and Acknowledgement](#license-and-acknowledgement)

# Preparing

## Setup

We recommend you to use [anaconda](https://www.anaconda.com/) to manage dependencies. You may need to change the torch and cuda version according to your computer.

1. Create conda environment:
```
conda create --name slcf python=3.7
conda activate slcf
```

2. Please install [PyTorch](https://pytorch.org/): 
```
conda install pytorch=1.13.0 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

3. Install the additional dependencies:
```
cd SLCF-Net/
pip install -r requirements.txt
```

4. We use [dvis](https://github.com/SirWyver/dvis) for visualization, which is a lightweight but efficient tool with a web server. We recommend you to use another conda environment to visualize the result. 

```
conda create --name dvis python=3.8 requests matplotlib pyyaml tqdm imageio
conda activate dvis
pip install visdom
git clone git@github.com:SirWyver/dvis.git
cd dvis
pip install .
```

## Datasets

1. You need to download
  - The **Semantic Scene Completion dataset v1.1** (SemanticKITTI voxel data (700 MB)) from [SemanticKITTI website](http://www.semantic-kitti.org/dataset.html#download)
  -  The **KITTI Odometry Benchmark calibration data** (Download odometry data set (calibration files, 1 MB)) and the **RGB images** (Download odometry data set (color, 65 GB)) from [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
  - The dataset folder at **/path/to/semantic_kitti** should have the following structure:
    ```
    └── /path/to/semantic_kitti/
      └── dataset
        └── sequences
          ├── 00
          | ├── image_2
          | ├── labels
          | ├── velodyne
          | ├── voxels
          | ├── calib.txt
          | ├── poses.txt
          | └── times.txt
          ├── 01
          ...
    ```

2. Create a folder to store SemanticKITTI preprocess data at `/path/to/kitti/preprocess/folder`.

3. Store paths in environment variables for faster access (**Note: folder 'dataset' is in /path/to/semantic_kitti**):

```
$ export KITTI_PREPROCESS=/path/to/kitti/preprocess/folder
$ export KITTI_ROOT=/path/to/semantic_kitti 
```

# Running
All the scripts is controlled by the `/SLCF-Net/slcfnet/config/slcfnet.yaml`
## Preprocess
Before starting the training process, some preprocess is necessary
```
python preprocess.py
```

## Training
```
python train.py
```

## Evaluation
Put the checkpoints in the `/path/to/kitti/logdir/trained_models/kitti.ckpt`, then run:
```
python evaluation.py
```

## Inference
Put the checkpoints in the `/path/to/kitti/logdir/trained_models/kitti.ckpt`, then run:
```
python generate_output.py
```

## Visualization
Please follow the guide of [dvis](https://github.com/SirWyver/dvis), you need to setup the server before running the script.
```
python visualization.py
```

## Citation
If you find this work or code useful, please cite our paper and give this repo a star :)
```
@inproceedings{cao2024slcf,
  title = {{SLCF-Net}: Sequential {LiDAR}-Camera Fusion for Semantic Scene Completion using a {3D} Recurrent {U-Net}},
  author = {Cao, Helin and Behnke, Sven},
  booktitle = {IEEE Int. Conf. on Robotics and Automation (ICRA)},
  pages = {2767--2773},
  year = {2024},
}
```

# License and Acknowledgement
SLCF-Net is released under the [MIT license](./LICENSE). Our code follows several awesome repositories. We appreciate them for making their codes available to public.
- [MonoScene](https://github.com/astra-vision/MonoScene)
- [DVIS](https://github.com/SirWyver/dvis)
- [LMSCNet](https://github.com/astra-vision/LMSCNet)
- [SemanticKitti](https://github.com/PRBonn/semantic-kitti-api)