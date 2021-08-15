# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-10 10:48:55
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-26 14:20:42
# @Email:  cshzxie@gmail.com
#
# Note:
# - Replace float -> double, kFloat -> kDouble in gridding.cu and gridding_reverse.cu

import os
import sys
import torch
import unittest

from torch.autograd import gradcheck

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from extensions.gridding import GriddingFunction, GriddingReverseFunction


class GriddingTestCase(unittest.TestCase):
    def test_gridding_reverse_function_4(self):
        x = torch.rand(2, 4, 4, 4)
        x.requires_grad = True
        self.assertTrue(gradcheck(GriddingReverseFunction.apply, [4, x.double().cuda()]))

    def test_gridding_reverse_function_8(self):
        x = torch.rand(4, 8, 8, 8)
        x.requires_grad = True
        self.assertTrue(gradcheck(GriddingReverseFunction.apply, [8, x.double().cuda()]))

    def test_gridding_reverse_function_16(self):
        x = torch.rand(1, 16, 16, 16)
        x.requires_grad = True
        self.assertTrue(gradcheck(GriddingReverseFunction.apply, [16, x.double().cuda()]))

    def test_gridding_function_32pts(self):
        x = torch.rand(1, 32, 3)
        x.requires_grad = True
        self.assertTrue(gradcheck(GriddingFunction.apply, [x.double().cuda()]))


if __name__ == '__main__':
    unittest.main()
