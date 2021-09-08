# PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pointr-diverse-point-cloud-completion-with/point-cloud-completion-on-shapenet)](https://paperswithcode.com/sota/point-cloud-completion-on-shapenet?p=pointr-diverse-point-cloud-completion-with)

Created by [Xumin Yu](https://yuxumin.github.io/)\*, [Yongming Rao](https://raoyongming.github.io/)\*, [Ziyi Wang](https://github.com/LavenderLA), [Zuyan Liu](https://github.com/lzy-19), [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=en&authuser=1), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1)

[[arXiv]](https://arxiv.org/abs/2108.08839) [[Video]](https://youtu.be/mSGphas0p8g) [[Dataset]](./DATASET.md) [[Models]](#pretrained-models)

This repository contains PyTorch implementation for __PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers__ (ICCV 2021 Oral Presentation).

PoinTr is a transformer-based model for point cloud completion.  By representing the point cloud as a set of unordered groups of points with position embeddings, we convert the point cloud to a sequence of point proxies and employ a transformer encoder-decoder architecture for generation. We also propose two more challenging benchmarks [ShapeNet-55/34](./DATASET.md) with more diverse incomplete point clouds that can better reflect the real-world scenarios to promote future research.

![intro](fig/pointr.gif)

## Pretrained Models

We provide pretrained PoinTr models:
| dataset  | url|
| --- | --- |  
| ShapeNet-55 | [[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/4a7027b83da343bb9ac9/?dl=1)] / [[Google Drive](https://drive.google.com/file/d/1WzERLlbSwzGOBybzkjBrApwyVMTG00CJ/view?usp=sharing)] / [[BaiDuYun](https://pan.baidu.com/s/1T4NqN5HQkInDTlNAX2KHbQ)] (code:erdh) |
| ShapeNet-34 | [[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/ac82414f884d445ebd54/?dl=1)] / [[Google Drive](https://drive.google.com/file/d/1Xy6wZjgJNhOYe3wDA-SbLMmGwBJ0jcBz/view?usp=sharing)] / [[BaiDuYun](https://pan.baidu.com/s/1zAxYf_9ixixqR7lvnBsRNQ)] (code:atbb ) |
| PCN |  [[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/55b01b2990e040aa9cb0/?dl=1)] / [[Google Drive](https://drive.google.com/file/d/182xUHiUyIQhgqstFTVPoCyYyxmdiZlxq/view?usp=sharing)]  / [[BaiDuYun](https://pan.baidu.com/s/1iGenIM076akP8EgbYFBWyw)] (code:9g79) |
| KITTI | coming soon |

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

#### Building Pytorch Extensions for Chamfer Distance, PointNet++ and kNN

*NOTE:* PyTorch >= 1.7 and GCC >= 4.9 are required.

```
# Chamfer Distance
bash install.sh
# PointNet++
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

### Dataset

The details of our new ***ShapeNet-55/34*** datasets and other existing datasets can be found in [DATASET.md](./DATASET.md).

### Evaluation

To evaluate a pre-trained PoinTr model on the Three Dataset with single GPU, run:

```
bash ./scripts/test.sh <GPU_IDS> --ckpts <path> --config <config> --exp_name <name> [--mode <easy/median/hard>]
```

####  Some examples:
Test the PoinTr pretrained model on the PCN benchmark:
```
bash ./scripts/test.sh 0 --ckpts ./pretrained/PoinTr_PCN.pth --config ./cfgs/PCN_models/PoinTr.yaml --exp_name example
```
Test the PoinTr pretrained model on ShapeNet55 benchmark (*easy* mode):
```
bash ./scripts/test.sh 0 --ckpts ./pretrained/PoinTr_ShapeNet55.pth --config ./cfgs/ShapeNet55_models/PoinTr.yaml --mode easy --exp_name example
```
Test the PoinTr pretrained model on the KITTI benchmark:
```
bash ./scripts/test.sh 0 --ckpts ./pretrained/PoinTr_KITTI.pth --config ./cfgs/KITTI_models/PoinTr.yaml --exp_name example
```

### Training

To train a point cloud completion model from scratch, run:

```
# Use DistributedDataParallel (DDP)
bash ./scripts/dist_train.sh <NUM_GPU> <port> --config <config> --exp_name <name> [--resume] [--start_ckpts <path>] [--val_freq <int>]
# or just use DataParallel (DP)
bash ./scripts/train.sh <GPUIDS> --config <config> --exp_name <name> [--resume] [--start_ckpts <path>] [--val_freq <int>]
```
####  Some examples:
Train a PoinTr model on PCN benchmark with 2 gpus:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 --config ./cfgs/PCN_models/PoinTr.yaml --exp_name example
```
Resume a checkpoint:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 --config ./cfgs/PCN_models/PoinTr.yaml --exp_name example --resume
```

Finetune a PoinTr on PCNCars
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 --config ./cfgs/KITTI_models/PoinTr.yaml --exp_name example --start_ckpts ./weight.pth
```

Train a PoinTr model with a single GPU:
```
bash ./scripts/train.sh 0 --config ./cfgs/KITTI_models/PoinTr.yaml --exp_name example
```

We also provide the Pytorch implementation of several baseline models including GRNet, PCN, TopNet and FoldingNet. For example, to train a GRNet model on ShapeNet-55, run:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 --config ./cfgs/ShapeNet55_models/GRNet.yaml --exp_name example
```

### Completion Results on ShapeNet55 and KITTI-Cars

![results](fig/VisResults.gif)

## License
MIT License

## Acknowledgements

Our code is inspired by [GRNet](https://github.com/hzxie/GRNet) and [mmdetection3d](https://github.com/open-mmlab/mmdetection3d).

## Citation
If you find our work useful in your research, please consider citing: 
```
@inproceedings{yu2021pointr,
  title={PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers},
  author={Yu, Xumin, Rao, Yongming and Wang, Ziyi and Liu, Zuyan, and Lu, Jiwen and Zhou, Jie},
  booktitle={ICCV},
  year={2021}
}
```
