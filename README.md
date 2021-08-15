# PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers

Created by [Xumin Yu\*](https://yuxumin.github.io/), [Yongming Rao\*](https://raoyongming.github.io/), [Ziyi Wang](https://github.com/LavenderLA), [Zuyan Liu](https://github.com/lzy-19), [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=en&authuser=1), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1)

This repository contains PyTorch implementation for PoinTr (ICCV 2021 Oral Presentation).

PoinTr is a transformer-style architecture that reformulates point cloud completion as a set-to-set translation problem. Our architecture extracts the geometric information and semantic relation in the mean time, the new design of transformers helps us learn structural knowledge and preserve detailed information, which is crucial for point cloud completion.

![intro](fig/pointr.gif)

[[arXiv]]()

## Pretrained Models

 - PoinTr for PCN [[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/55b01b2990e040aa9cb0/?dl=1)] [[Google Drive](https://drive.google.com/file/d/182xUHiUyIQhgqstFTVPoCyYyxmdiZlxq/view?usp=sharing)]

## Usage

### Requirements

- PyTorch >= 1.7.0
- python >= 3.7
- CUDA >= 9.0
- GCC >= 4.9 
- torchvision
- timm
- open3d
- tensorboardX

```
pip install -r requirements.txt
```

#### Build Pytorch extension (chamfer distance and others)

NOTE: PyTorch >= 1.7,  and GCC >= 4.9 are required.

```
bash install.sh
```

#### Building C++/CUDA Extensions for PointNet++ and kNN
```
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

### Prepare Dataset

[DATASET](./DATASET.md)

### Evaluation

To evaluate a pre-trained PoinTr model on the Three Dataset with single GPU, run:

```
# bash ./scipt/test.sh <GPU_IDS> --ckpts <path> --config <config> --exp_name <name> [--mode <easy/median/hard>]
```
#### for example:
Test PoinTr pretrained ckpt on PCN benchmark
```
bash ./scipt/test.sh 0 --ckpts ./pretrained/PoinTr_PCN.pth --config ./cfgs/PCN_models/PoinTr.yaml --exp_name example
```
Test PoinTr pretrained ckpt on ShapeNet55 benchmark under easy mode
```
bash ./scipt/test.sh 0 --ckpts ./pretrained/PoinTr_ShapeNet55.pth --config ./cfgs/ShapeNet55_models/PoinTr.yaml --mode easy --exp_name example
```
Test PoinTr pretrained ckpt on KITTI benchmark
```
bash ./scipt/test.sh 0 --ckpts ./pretrained/PoinTr_KITTI.pth --config ./cfgs/KITTI_models/PoinTr.yaml --exp_name example
```

### Training

To train PointCompletion models from scratch, run:

```
# Using DDP
# bash ./scipt/dist_train.sh <NUM_GPU> <port> --config <config> --exp_name <name> [--resume] [--start_ckpts <path>]
```
####  for example:
Train a PoinTr model on PCN benchmark with 0,1 gpus
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 --config ./cfgs/PCN_models/PoinTr.yaml --exp_name example
```
Autoresume the ckpts
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 --config ./cfgs/PCN_models/PoinTr.yaml --exp_name example --resume
```
Train a GRNet on ShapeNet-55
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 --config ./cfgs/ShapeNet55_models/GRNet.yaml --exp_name example
```
Finetune a PoinTr on PCNCars
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 --config ./cfgs/KITTI_models/PoinTr.yaml --exp_name example --start_ckpts ./weight.pth
```

single GPU
```
bash ./scripts/train.sh 0 --config ./cfgs/KITTI_models/PoinTr.yaml --exp_name example
```

### Completion Results on ShapeNet55 and KITTI-Cars

![results](fig/VisResults.gif)

## License
MIT License
