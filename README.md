# V2V4Real: A large-scale real-world dataset for Vehicle-to-Vehicle Cooperative Perception
[![website](https://img.shields.io/badge/Website-Explore%20Now-blueviolet?style=flat&logo=google-chrome)](https://research.seas.ucla.edu/mobility-lab/v2v4real/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2303.07601.pdf)
[![supplement](https://img.shields.io/badge/Supplementary-Material-red)](https://arxiv.org/pdf/2303.07601.pdf)
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)]()


This is the official implementation of CVPR2023 **Highlight** paper. "V2V4Real: A large-scale real-world dataset for Vehicle-to-Vehicle Cooperative Perception".
[Runsheng Xu](https://derrickxunu.github.io/),  [Xin Xia](https://scholar.google.com/citations?user=vCYqMTIAAAAJ&hl=en), [Jinlong Li](https://jinlong17.github.io/), [Hanzhao Li](), [Shuo Zhang](),  [Zhengzhong Tu](https://github.com/vztu), [Zonglin Meng](), [Hao Xiang](https://xhwind.github.io/), [Xiaoyu Dong](), [Rui Song](), [Hongkai Yu](https://scholar.google.com/citations?user=JnQts0kAAAAJ&hl=en), [Bolei Zhou](https://boleizhou.github.io/), [Jiaqi Ma](https://mobility-lab.seas.ucla.edu/)

Supported by the [UCLA Mobility Lab](https://mobility-lab.seas.ucla.edu/).

<p align="center">
<img src="imgs/scene1.png" width="600" alt="" class="img-responsive">
</p>

## Overview
- [Codebase Features](#codebase-features)
- [Data Download](#data-download)
- [Changelog](#changelog)
- [Devkit Setup](#devkit-setup)
- [Quick Start](#quick-start)
- [Benchmark](#benchmark)
- [Citation](#citation)
- [Acknowledgment](#known-issues)

## CodeBase Features
- Support both simulation and real-world cooperative perception dataset
    - [x] V2V4Real
    - [x] OPV2V
- Multiple Tasks supported
    - [x] 3D object detection
    - [ ] Cooperative tracking
    - [ ] Domain adaption
- SOTA model supported
    - [x] [Attentive Fusion [ICRA2022]](https://arxiv.org/abs/2109.07644)
    - [x] [Cooper [ICDCS]](https://arxiv.org/abs/1905.05265)
    - [x] [F-Cooper [SEC2019]](https://arxiv.org/abs/1909.06459)
    - [x] [V2VNet [ECCV2022]](https://arxiv.org/abs/2008.07519)
    - [x] [V2X-ViT [ECCV2022]](https://github.com/DerrickXuNu/v2x-vit)
    - [x] [CoBEVT [CoRL2022]](https://arxiv.org/abs/2207.02202)

## Data Download
Please check our [website](https://research.seas.ucla.edu/mobility-lab/v2v4real/) to download the data (OPV2V format).

After downloading the data, please put the data in the following structure:
```shell
├── v2v4real
│   ├── train
|      |── testoutput_CAV_data_2022-03-15-09-54-40_1
│   ├── validate
│   ├── test
```
## Changelog
- Mar. 23, 2023: The codebase for 3D object detection is released
- Mar. 19, 2023: The website is ready
- Mar. 14, 2023: Tha paper is release

## Devkit setup
V2V4Real's codebase is build upon [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD). Compared to OpenCOOD, this codebase supports both the simulation and real-world data and more perception tasks. Furthermore, this repo provides augmentations that OpenCOOD does not support. We highly recommend you to use this codebase to train your model on V2V4Real dataset

To set up the codebase environment, do the following steps:
#### 1. Create conda environment (python >= 3.7)
```shell
conda create -n v2v4real python=3.7
conda activate v2v4real
```
#### 2. Pytorch Installation (>= 1.12.0 Required)
Take pytorch 1.12.0 as an example:
```shell
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
#### 3. spconv 2.x Installation
```shell
pip install spconv-cu113
```
#### 4. Install other dependencies
```shell
pip install -r requirements.txt
python setup.py develop
```
#### 5.Install bbx nms calculation cuda version
```shell
python opencood/utils/setup.py build_ext --inplace
```

## Quick Start
### Data sequence visualization
To quickly visualize the LiDAR stream in the OPV2V dataset, first modify the `validate_dir`
in your `opencood/hypes_yaml/visualization.yaml` to the opv2v data path on your local machine, e.g. `opv2v/validate`,
and the run the following commond:
```python
cd ~/OpenCOOD
python opencood/visualization/vis_data_sequence.py [--color_mode ${COLOR_RENDERING_MODE} --isSim]
```
Arguments Explanation:
- `color_mode` : str type, indicating the lidar color rendering mode. You can choose from 'v2vreal', 'constant', 'intensity' or 'z-value'.
- `isSim` : bool type, if you are visualizing the simulation data, then claim this argument.

### Train your model
OpenCOOD uses yaml file to configure all the parameters for training. To train your own model
from scratch or a continued checkpoint, run the following commonds:
```python
python opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER} --half]
```
Arguments Explanation:
- `hypes_yaml`: the path of the training configuration file, e.g. `opencood/hypes_yaml/point_pillar_fax.yaml`, meaning you want to train
CoBEVT with pointpillar backbone. See [Tutorial 1: Config System](https://opencood.readthedocs.io/en/latest/md_files/config_tutorial.html) to learn more about the rules of the yaml files.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.
- `half` (optional): If set, the model will be trained with half precision. It cannot be set with multi-gpu training togetger.

To train on **multiple gpus**, run the following command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --use_env opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}]
```
### Test the model
Before you run the following command, first make sure the `validation_dir` in config.yaml under your checkpoint folder
refers to the testing dataset path, e.g. `v2v4real/test`.

```python
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} [--show_vis] [--show_sequence]
```
Arguments Explanation:
- `model_dir`: the path to your saved model.
- `fusion_method`: indicate the fusion strategy, currently support 'nofusion', 'early', 'late', and 'intermediate'.
- `show_vis`: whether to visualize the detection overlay with point cloud.
- `show_sequence` : the detection results will visualized in a video stream. It can NOT be set with `show_vis` at the same time.

The evaluation results  will be dumped in the model directory.

Important notes for testing:
1. Remember to change the `validation_dir` in config.yaml under your checkpoint folder to the testing dataset path, e.g. `v2v4real/test`.
2. To test under async mode, you need to set the `async_mode` in config.yaml to `True` and set the `async_overhead` to the desired delay time (default 100ms).

## Benchmark
### Results of Cooperative 3D object detection
| Method        | Backbone    | Sync AP@0.5 | Sync AP@0.7 | Async AP@0.5 | Async AP@0.7 | Bandwidth | Download Link                                                            |
|--------------|-------------|----------------|----------------|--------------|--------------|-----------|--------------------------------------------------------------------------|
| No Fusion    | PointPillar | 39.8           | 22.0          | 39.8          | 22.0          |      0.0     |    [url](https://drive.google.com/file/d/1spnCYEbzOiQaK4p9u9kD1K-hUCh6Me3-/view?usp=share_link)                                                                    |
| Late Fusion  | PointPillar | 55.0           | 26.7       | 50.2        | 22.4         |      0.003     |      [url](https://drive.google.com/file/d/1spnCYEbzOiQaK4p9u9kD1K-hUCh6Me3-/view?usp=share_link)                                                                         |
| Early Fusion | PointPillar  | 59.7          | 32.1         | 52.1        | 25.8       |      0.96     |       [url](https://drive.google.com/file/d/1v8aD_HyQnUddhGhZAlqAziLo43LwlOc0/view?usp=share_link)                       |
| [F-Cooper](https://arxiv.org/abs/1909.06459) | PointPillar | 60.7          | 31.8          | 53.6        | 26.7       |      0.20     |     [url](https://drive.google.com/file/d/1znq2xSa3bYrKg_KsqA4Ax34sbYcZ7YBe/view?usp=share_link)                                                                     |
| [Attentive Fusion](https://arxiv.org/abs/2109.07644)     | PointPillar | 64.5         | 34.3          | 56.4         | 28.5      |     0.20      |      [url](https://drive.google.com/file/d/1RudJFuJrKRwJpEVtEx-ZV05-yBj-HWlR/view?usp=share_link)                                                                    |
| [V2VNet](https://arxiv.org/abs/2008.07519)         |PointPillar | 64.7         | 33.6           | 57.7        | 27.5     |     0.20      |     [url](https://drive.google.com/file/d/1MtkaUHT5_LdwWs73g034pATa1sUJxHaf/view?usp=share_link)                                                                      |
| [V2X-ViT](https://arxiv.org/pdf/2203.10638.pdf)    | PointPillar | 64.9          | **36.9**           | 55.9       | 29.3       |   0.20        | [url](https://drive.google.com/file/d/1gtF_RHxhOLEAqhUVWaOlBJMLdUXFEIBb/view?usp=share_link)
| [CoBEVT](https://arxiv.org/abs/2207.02202)      | PointPillar |    **66.5**     |  36.0   | **58.6**  | **29.7**  | 0.20| [url](https://drive.google.com/file/d/1aTpADzAYvseyHDstePakh5mKXrW1-zaz/view?usp=share_link)|

### Results of Cooperative tracking
| Method       | AMOTA(↑) | AMOTP(↑) | sAMOTA(↑) | MOTA(↑)  | MT(↑)    | ML(↓)    |
|--------------|----------|----------|-----------|----------|----------|----------|
| No Fusion    | 16.08    | 41.60    | 53.84     | 43.46    | 29.41    | 60.18    |
| Late Fusion  | 29.28    | 51.08    | 71.05     | 59.89    | 45.25    | 31.22    |
| Early Fusion | 26.19    | 48.15    | 67.34     | 60.87    | 40.95    | 32.13    |
| F-Cooper     | 23.29    | 43.11    | 65.63     | 58.34    | 35.75    | 38.91    |
| AttFuse      | 28.64    | 50.48    | 73.21     | 63.03    | 46.38    | 28.05    |
| V2VNet       | 30.48    | 54.28    | 75.53     | **64.85**    | **48.19**    | 27.83    |
| V2X-ViT      | 30.85    | 54.32    | 74.01     | 64.82    | 45.93    | **26.47**    |
| CoBEVT       | **32.12**    | **55.61**    | **77.65**     | 63.75    | 47.29    | 30.32    |

### Results of Domain Adaption
| Method       | Domain Adaption | AP@0.5 | Download Link |
|--------------|----------|----------|-----------
| F-Cooper     | [1]   | 37.3   |     | 
| AttFuse      | [1]      | 23.4   |     | 
| V2VNet       | [1]      | 26.3   |     |
| V2X-ViT      | [1]     | 39.5   |    | 
| CoBEVT       | [1]      | **40.2**  |     |

[1]: Yuhua Chen, Wen Li, Christos Sakaridis, Dengxin Dai, and
Luc Van Gool. Domain adaptive faster r-cnn for object de-
tection in the wild. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pages 3339–3348, 2018.

## Citation
```shell
@inproceedings{xu2023v2v4real,
  title={V2V4Real: A Real-world Large-scale Dataset for Vehicle-to-Vehicle Cooperative Perception},
  author={Xu, Runsheng and Xia, Xin and Li, Jinlong and Li, Hanzhao and Zhang, Shuo and Tu, Zhengzhong and Meng, Zonglin and Xiang, Hao and Dong, Xiaoyu and Song, Rui and Yu, Hongkai and Zhou, Bolei and Ma, Jiaqi},
  booktitle={The IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR)},
  year={2023}
}
```

## Acknowledgment
This dataset belongs to the [OpenCDA ecosystem](https://arxiv.org/abs/2301.07325) family. The codebase is build upon [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD), which is the first Open Cooperative Detection framework for autonomous driving.