# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-09-06 11:35:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:20:36
# @Email:  cshzxie@gmail.com

import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from extensions.gridding import Gridding, GriddingReverse
from extensions.cubic_feature_sampling import CubicFeatureSampling
from extensions.chamfer_dist import ChamferDistanceL2
from extensions.gridding_loss import GriddingLoss
from .build import MODELS



class RandomPointSampling(torch.nn.Module):
    def __init__(self, n_points):
        super(RandomPointSampling, self).__init__()
        self.n_points = n_points

    def forward(self, pred_cloud, partial_cloud=None):
        if partial_cloud is not None:
            pred_cloud = torch.cat([partial_cloud, pred_cloud], dim=1)

        _ptcloud = torch.split(pred_cloud, 1, dim=0)
        ptclouds = []
        for p in _ptcloud:
            non_zeros = torch.sum(p, dim=2).ne(0)
            p = p[non_zeros].unsqueeze(dim=0)
            n_pts = p.size(1)
            if n_pts < self.n_points:
                rnd_idx = torch.cat([torch.randint(0, n_pts, (self.n_points, ))])
            else:
                rnd_idx = torch.randperm(p.size(1))[:self.n_points]
            ptclouds.append(p[:, rnd_idx, :])

        return torch.cat(ptclouds, dim=0).contiguous()

@MODELS.register_module()
class GRNet(torch.nn.Module):
    def __init__(self, config):
        super(GRNet, self).__init__()
        self.num_pred = config.num_pred
        self.gridding_scale = [config.gridding_loss_scales]
        self.gridding_alpha = [config.gridding_loss_alphas]
        self.loss_lambda = 0.
        self.gridding = Gridding(scale=64)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.fc5 = torch.nn.Sequential(
            torch.nn.Linear(16384, 2048),
            torch.nn.ReLU()
        )
        self.fc6 = torch.nn.Sequential(
            torch.nn.Linear(2048, 16384),
            torch.nn.ReLU()
        )
        self.dconv7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.dconv8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.dconv9 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.dconv10 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.ReLU()
        )
        self.gridding_rev = GriddingReverse(scale=64)
        self.point_sampling = RandomPointSampling(n_points=self.num_pred//8)
        self.feature_sampling = CubicFeatureSampling()
        self.fc11 = torch.nn.Sequential(
            torch.nn.Linear(1792, 1792),
            torch.nn.ReLU()
        )
        self.fc12 = torch.nn.Sequential(
            torch.nn.Linear(1792, 448),
            torch.nn.ReLU()
        )
        self.fc13 = torch.nn.Sequential(
            torch.nn.Linear(448, 112),
            torch.nn.ReLU()
        )
        self.fc14 = torch.nn.Linear(112, 24)
        # self.fc14 = torch.nn.Linear(112, 3)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func_1 = ChamferDistanceL2()
        self.loss_func_2 = GriddingLoss(
                        self.gridding_scale,
                        self.gridding_alpha)

    def get_loss(self, ret, gt):
        loss_coarse = self.loss_func_1(ret[0], gt) + self.loss_func_2(ret[0], gt) * self.loss_lambda
        loss_fine = self.loss_func_1(ret[1], gt)
        return loss_coarse, loss_fine

    def forward(self, xyz):
        # NOTE: # Avoid overflow while gridding on ShapeNet55
        partial_cloud = xyz * 0.5
        # print(partial_cloud.size())     # torch.Size([batch_size, 2048, 3])
        pt_features_64_l = self.gridding(partial_cloud).view(-1, 1, 64, 64, 64)
        # print(pt_features_64_l.size())  # torch.Size([batch_size, 1, 64, 64, 64])
        pt_features_32_l = self.conv1(pt_features_64_l)
        # print(pt_features_32_l.size())  # torch.Size([batch_size, 32, 32, 32, 32])
        pt_features_16_l = self.conv2(pt_features_32_l)
        # print(pt_features_16_l.size())  # torch.Size([batch_size, 64, 16, 16, 16])
        pt_features_8_l = self.conv3(pt_features_16_l)
        # print(pt_features_8_l.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        pt_features_4_l = self.conv4(pt_features_8_l)
        # print(pt_features_4_l.size())   # torch.Size([batch_size, 256, 4, 4, 4])
        features = self.fc5(pt_features_4_l.view(-1, 16384))
        # print(features.size())          # torch.Size([batch_size, 2048])
        pt_features_4_r = self.fc6(features).view(-1, 256, 4, 4, 4) + pt_features_4_l
        # print(pt_features_4_r.size())   # torch.Size([batch_size, 256, 4, 4, 4])
        pt_features_8_r = self.dconv7(pt_features_4_r) + pt_features_8_l
        # print(pt_features_8_r.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        pt_features_16_r = self.dconv8(pt_features_8_r) + pt_features_16_l
        # print(pt_features_16_r.size())  # torch.Size([batch_size, 64, 16, 16, 16])
        pt_features_32_r = self.dconv9(pt_features_16_r) + pt_features_32_l
        # print(pt_features_32_r.size())  # torch.Size([batch_size, 32, 32, 32, 32])
        pt_features_64_r = self.dconv10(pt_features_32_r) + pt_features_64_l
        # print(pt_features_64_r.size())  # torch.Size([batch_size, 1, 64, 64, 64])
        sparse_cloud = self.gridding_rev(pt_features_64_r.squeeze(dim=1))
        # print(sparse_cloud.size())      # torch.Size([batch_size, 262144, 3])
        sparse_cloud = self.point_sampling(sparse_cloud, partial_cloud)
        # print(sparse_cloud.size())      # torch.Size([batch_size, num_pred//8, 3])
        point_features_32 = self.feature_sampling(sparse_cloud, pt_features_32_r).view(-1, self.num_pred//8, 256)
        # print(point_features_32.size()) # torch.Size([batch_size, num_pred//8, 256])
        point_features_16 = self.feature_sampling(sparse_cloud, pt_features_16_r).view(-1, self.num_pred//8, 512)
        # print(point_features_16.size()) # torch.Size([batch_size, num_pred//8, 512])
        point_features_8 = self.feature_sampling(sparse_cloud, pt_features_8_r).view(-1, self.num_pred//8, 1024)
        # print(point_features_8.size())  # torch.Size([batch_size, num_pred//8, 1024])
        point_features = torch.cat([point_features_32, point_features_16, point_features_8], dim=2)
        # print(point_features.size())    # torch.Size([batch_size, num_pred//8, 1792])
        point_features = self.fc11(point_features)
        # print(point_features.size())    # torch.Size([batch_size, num_pred//8, 1792])
        point_features = self.fc12(point_features)
        # print(point_features.size())    # torch.Size([batch_size, num_pred//8, 448])
        point_features = self.fc13(point_features)
        # print(point_features.size())    # torch.Size([batch_size, num_pred//8, 112])
        point_offset = self.fc14(point_features).view(-1, self.num_pred, 3)
        # point_offset = self.fc14(point_features).view(-1, num_crop, 3)
        dense_cloud = sparse_cloud.unsqueeze(dim=2).repeat(1, 1, 8, 1).reshape(-1,self.num_pred,3) + point_offset

        ret = (sparse_cloud * 2.0, dense_cloud * 2.0)
        # ret = (sparse_cloud, dense_cloud)
        return ret
