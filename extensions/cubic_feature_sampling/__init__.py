# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-19 16:55:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-26 13:15:14
# @Email:  cshzxie@gmail.com

import torch

import cubic_feature_sampling


class CubicFeatureSamplingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ptcloud, cubic_features, neighborhood_size=1):
        scale = cubic_features.size(2)
        point_features, grid_pt_indexes = cubic_feature_sampling.forward(scale, neighborhood_size, ptcloud,
                                                                         cubic_features)
        ctx.save_for_backward(torch.Tensor([scale]), torch.Tensor([neighborhood_size]), grid_pt_indexes)
        return point_features

    @staticmethod
    def backward(ctx, grad_point_features):
        scale, neighborhood_size, grid_pt_indexes = ctx.saved_tensors
        scale = int(scale.item())
        neighborhood_size = int(neighborhood_size.item())
        grad_point_features = grad_point_features.contiguous()
        grad_ptcloud, grad_cubic_features = cubic_feature_sampling.backward(scale, neighborhood_size,
                                                                            grad_point_features, grid_pt_indexes)
        return grad_ptcloud, grad_cubic_features, None


class CubicFeatureSampling(torch.nn.Module):
    def __init__(self):
        super(CubicFeatureSampling, self).__init__()

    def forward(self, ptcloud, cubic_features, neighborhood_size=1):
        h_scale = cubic_features.size(2) / 2
        ptcloud = ptcloud * h_scale + h_scale
        return CubicFeatureSamplingFunction.apply(ptcloud, cubic_features, neighborhood_size)
