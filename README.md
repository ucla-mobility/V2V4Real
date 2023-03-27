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
- [Data Download](#data-download)
- [Changelog](#changelog)
- [Devkit setup](#devkit-setup)
- [Benchmark](#nuimages)
- [Citation](#citation)
- [Acknowledgment](#known-issues)



### Data Download
Please check our [website](https://research.seas.ucla.edu/mobility-lab/v2v4real/) to download the data.

### Changelog
- Mar. 19, 2023: The website is ready
- Mar. 14, 2023: Tha paper is release

### Devkit setup
V2V4Real's codebase is build upon [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD). To set up the codebase environment,
do the following steps:
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
pip install spconv-cu117
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