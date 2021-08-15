# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-20 11:50:50
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-26 13:52:33
# @Email:  cshzxie@gmail.com
#
# Note:
# - Replace float -> double, kFloat -> kDouble in cubic_feature_sampling.cu

import os
import sys
import torch
import unittest

from torch.autograd import gradcheck

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from extensions.cubic_feature_sampling import CubicFeatureSamplingFunction


class CubicFeatureSamplingTestCase(unittest.TestCase):
    def test_neighborhood_size_1(self):
        ptcloud = torch.rand(2, 64, 3) * 2 - 1
        cubic_features = torch.rand(2, 4, 8, 8, 8)
        ptcloud.requires_grad = True
        cubic_features.requires_grad = True
        self.assertTrue(
            gradcheck(CubicFeatureSamplingFunction.apply,
                      [ptcloud.double().cuda(), cubic_features.double().cuda()]))

    def test_neighborhood_size_2(self):
        ptcloud = torch.rand(2, 32, 3) * 2 - 1
        cubic_features = torch.rand(2, 2, 8, 8, 8)
        ptcloud.requires_grad = True
        cubic_features.requires_grad = True
        self.assertTrue(
            gradcheck(CubicFeatureSamplingFunction.apply,
                      [ptcloud.double().cuda(), cubic_features.double().cuda(), 2]))

    def test_neighborhood_size_3(self):
        ptcloud = torch.rand(1, 32, 3) * 2 - 1
        cubic_features = torch.rand(1, 2, 16, 16, 16)
        ptcloud.requires_grad = True
        cubic_features.requires_grad = True
        self.assertTrue(
            gradcheck(CubicFeatureSamplingFunction.apply,
                      [ptcloud.double().cuda(), cubic_features.double().cuda(), 3]))


if __name__ == '__main__':
    unittest.main()
